from typing import List, Optional

import torch
import torch.nn as nn


class DynamicDenseCRF(nn.Module):
    """
    Dense CRF over K edge-states with edge embedding-based transition scoring.
    All transitions are allowed (no masking); transition scores are computed
    from edge embedding pairs.

    emissions: (B, T, K)
    edge_embeds: (B, T, K, D)
    tags: (B, T) compressed edge-state IDs in [0..K-1]
    """

    def __init__(
        self,
        comp_src: torch.Tensor,
        comp_dst: torch.Tensor,
        edge_embed_dim: int = 16,
        team_size: int = 11,
    ):
        super().__init__()
        self.K = comp_src.numel()
        self.comp_src = comp_src  # (K,)
        self.comp_dst = comp_dst  # (K,)
        self.edge_embed_dim = edge_embed_dim
        self.team_size = team_size

        self.linear = nn.Linear(edge_embed_dim * 2, 1)

    def _pairwise_trans_score(self, prev_embed: torch.Tensor, curr_embed: torch.Tensor) -> torch.Tensor:
        """
        Compute dense transition scores for all (prev, curr) pairs using the
        linear layer over concatenated edge embeddings, without materializing
        the full concatenation tensor.

        prev_embed: (B, K, D)
        curr_embed: (B, K, D)
        returns: (B, K_curr, K_prev)
        """
        D = prev_embed.size(-1)
        weight = self.linear.weight  # (1, 2D)
        bias = self.linear.bias  # (1,) or None

        w_prev = weight[:, :D]  # (1, D)
        w_curr = weight[:, D:]  # (1, D)

        prev_score = torch.matmul(prev_embed, w_prev.t()).squeeze(-1)  # (B, K_prev)
        curr_score = torch.matmul(curr_embed, w_curr.t()).squeeze(-1)  # (B, K_curr)

        trans_score = curr_score.unsqueeze(2) + prev_score.unsqueeze(1)  # (B, K_curr, K_prev)
        if bias is not None:
            trans_score = trans_score + bias.view(1, 1, 1)
        return trans_score

    def _compute_logZ(
        self,
        emissions: torch.Tensor,
        edge_embeds: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward algorithm in log-space (dense transitions).
        emissions: (B, T, K)
        edge_embeds: (B, T, K, D)
        """
        B, T, K = emissions.shape
        if mask is None:
            mask = emissions.new_ones((B, T), dtype=torch.bool)
        mask = mask.bool()

        alpha = emissions[:, 0]  # (B, K)

        for t in range(1, T):
            emit_score = emissions[:, t]  # (B, K_curr)

            prev_embed = edge_embeds[:, t - 1]  # (B, K_prev, D)
            curr_embed = edge_embeds[:, t]  # (B, K_curr, D)
            trans_score = self._pairwise_trans_score(prev_embed, curr_embed)  # (B, K_curr, K_prev)

            score = torch.logsumexp(alpha.unsqueeze(1) + trans_score, dim=2)  # (B, K_curr)
            new_alpha = score + emit_score  # (B, K_curr)
            alpha = torch.where(mask[:, t].unsqueeze(1), new_alpha, alpha)

        logZ = torch.logsumexp(alpha, dim=1)  # (B,)
        return logZ

    def _compute_gold_score(
        self,
        emissions: torch.Tensor,
        edge_embeds: torch.Tensor,
        tags: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, K = emissions.shape
        D = self.edge_embed_dim

        if mask is None:
            mask = emissions.new_ones((B, T), dtype=torch.bool)
        mask = mask.bool()

        score = emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)  # (B,)

        for t in range(1, T):
            prev = tags[:, t - 1]  # (B,)
            curr = tags[:, t]  # (B,)
            emit_score = emissions[:, t].gather(1, curr.unsqueeze(1)).squeeze(1)  # (B,)

            prev_embed = edge_embeds[:, t - 1]  # (B, K, D)
            curr_embed = edge_embeds[:, t]  # (B, K, D)
            prev_embed = prev_embed.gather(1, prev.view(B, 1, 1).expand(-1, 1, D)).squeeze(1)  # (B, D)
            curr_embed = curr_embed.gather(1, curr.view(B, 1, 1).expand(-1, 1, D)).squeeze(1)  # (B, D)
            trans_score = self.linear(torch.cat([prev_embed, curr_embed], dim=-1)).squeeze(-1)  # (B,)

            score += (emit_score + trans_score) * mask[:, t].float()

        return score  # (B,)

    def neg_log_likelihood(
        self,
        emissions: torch.Tensor,
        edge_embeds: torch.Tensor,
        tags: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        logZ = self._compute_logZ(emissions, edge_embeds, mask)
        gold = self._compute_gold_score(emissions, edge_embeds, tags, mask)
        nll = logZ - gold
        if reduction == "mean":
            return nll.mean()
        elif reduction == "sum":
            return nll.sum()
        else:
            return nll

    @torch.no_grad()
    def decode(
        self,
        emissions: torch.Tensor,
        edge_embeds: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        last_node_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Viterbi decode with dense transitions.
        """
        B, T, K = emissions.shape
        if mask is None:
            mask = emissions.new_ones((B, T), dtype=torch.bool)
        mask = mask.bool()

        delta = emissions[:, 0]  # (B, K)
        backpointers: List[torch.Tensor] = []

        for t in range(1, T):
            emit_score = emissions[:, t]  # (B, K_curr)
            trans_score = self._pairwise_trans_score(edge_embeds[:, t - 1], edge_embeds[:, t])  # (B, K, K)

            score = delta.unsqueeze(1) + trans_score  # (B, K_curr, K_prev)
            best_score, best_prev = score.max(dim=2)  # (B, K_curr), (B, K_curr)
            new_delta = emit_score + best_score  # (B, K_curr)

            delta = torch.where(mask[:, t].unsqueeze(1), new_delta, delta)
            backpointers.append(best_prev)

        if last_node_idx is not None:
            assert B == 1
            last_node_idx = last_node_idx.to(device=delta.device, dtype=torch.long)
            if last_node_idx.dim() == 0:
                last_node_idx = last_node_idx.view(1)
            comp_src = self.comp_src.to(device=delta.device, dtype=last_node_idx.dtype)
            comp_dst = self.comp_dst.to(device=delta.device, dtype=last_node_idx.dtype)
            best_last_tag = torch.where((comp_src == last_node_idx[0]) & (comp_dst == last_node_idx[0]))[0][0]
        else:
            best_last_tag = delta.argmax(dim=1)  # (B,)

        best_tags = emissions.new_zeros((B, T), dtype=torch.long)
        best_tags[:, -1] = best_last_tag

        for t in reversed(range(1, T)):
            curr = best_tags[:, t]  # (B,)
            prev = backpointers[t - 1].gather(1, curr.unsqueeze(1)).squeeze(1)  # (B,)
            best_tags[:, t - 1] = prev

        return best_tags

    @torch.no_grad()
    def dump_gold_score(
        self,
        emissions: torch.Tensor,
        edge_embeds: torch.Tensor,
        tags: torch.Tensor,
        sample_idx: int = 0,
    ):
        """
        Dump per-timestep components on the gold path for one sample,
        so you can see if trans or emission is computed differently in logZ vs gold.
        """
        print("==== GOLD PATH DEBUG ====")
        path = tags[sample_idx].tolist()
        print("path[:10] =", path[:10])

        emit_score = emissions[sample_idx, 0, path[0]].item()
        total_score = emit_score
        print(f"t=0 emit={emit_score:.6f}")

        for t in range(1, emissions.shape[1]):
            prev = path[t - 1]
            curr = path[t]
            emit_score = emissions[sample_idx, t, curr].item()

            prev_embed = edge_embeds[sample_idx, t - 1, prev]  # (D,)
            curr_embed = edge_embeds[sample_idx, t, curr]  # (D,)
            trans_score = self.linear(torch.cat([prev_embed, curr_embed], dim=-1)).item()

            total_score += emit_score + trans_score
            print(f"t={t} emit={emit_score:.6f} trans={trans_score:.6f}")

        print("total gold score =", total_score)
