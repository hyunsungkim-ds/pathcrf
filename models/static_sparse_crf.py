from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class StaticSparseCRF(nn.Module):
    """
    Sparse CRF over K edge-states with static transition parameters.
    Uses the same allowed-transition rules as EdgeEmbedCRF/TiedCRF, but
    learns a free parameter for each allowed (prev, curr) pair.

    emissions: (B, T, K)
    tags: (B, T) compressed edge-state IDs in [0..K-1]
    """

    def __init__(
        self,
        comp_src: torch.Tensor,
        comp_dst: torch.Tensor,
        team_size: int = 11,
        forbid_score: float = -1e4,
    ):
        super().__init__()
        self.K = comp_src.numel()
        self.comp_src = comp_src  # (K,)
        self.comp_dst = comp_dst  # (K,)
        self.team_size = team_size
        self.forbid_score = forbid_score

        inc_idx, inc_mask = self._build_incoming()
        self.register_buffer("inc_idx", inc_idx)  # (K, M) long, -1 padded
        self.register_buffer("inc_mask", inc_mask)  # (K, M) bool

        # One parameter per allowed incoming transition (aligned to inc_idx/inc_mask)
        self.transitions = nn.Parameter(torch.zeros(self.K, inc_idx.size(1)))

    def _same_team(self, a: int, b: int) -> bool:
        ts = self.team_size
        if a >= 2 * ts or b >= 2 * ts:
            return False
        else:
            return (a < ts and b < ts) or (a >= ts and b >= ts)

    def _edge_type(self, sender: int, receiver: int) -> str:
        """
        For transitions like A->X, classify X as teammate/opponent/outside relative to A.
        Assumes the sender is a player.
        Returns one of: "self", "teammate", "opponent", "out"
        """
        if receiver == sender:
            return "self"
        elif receiver >= 2 * self.team_size:
            return "out"
        else:
            return "teammate" if self._same_team(sender, receiver) else "opponent"

    def _build_allowed_next(self) -> List[List[Tuple[int, int]]]:
        """
        Build the allowed list of next states for each prev state i:
        allowed_next[i] = list of (j, _) according to the 12-parameter rules.
        """
        K = self.K
        src = self.comp_src.tolist()
        dst = self.comp_dst.tolist()

        allowed_next: List[List[Tuple[int, int]]] = [[] for _ in range(K)]

        for i in range(K):
            src_idx = src[i]
            dst_idx = dst[i]

            if src_idx >= 2 * self.team_size and src_idx == dst_idx:
                # Case 4: self-loop of an outside edge O (22..25)
                # Only the same self-loop allowed
                allowed_next[i].append((i, 11))
                continue

            # Non-self edge from an outside edge should not exist in K
            # but if it did, no outgoing allowed
            if src_idx >= 2 * self.team_size and src_idx != dst_idx:
                continue

            if src_idx == dst_idx:
                # Case 1: self-loop of a player A (0..21)
                # The next edge must originate from A.
                for j in range(K):
                    if src[j] != src_idx:
                        continue
                    rel = self._edge_type(src_idx, dst[j])
                    if rel == "self":
                        allowed_next[i].append((j, 0))
                    elif rel == "teammate":
                        allowed_next[i].append((j, 1))
                    elif rel == "opponent":
                        allowed_next[i].append((j, 2))
                    elif rel == "out":
                        allowed_next[i].append((j, 3))
                continue

            if dst_idx < 2 * self.team_size:
                # Case 2: non-self edge from a player A (0..21) to another player B (0..21)
                # The next edge must be either the same edge A->B or originate from B
                allowed_next[i].append((i, 4))
                for j in range(K):
                    if src[j] != dst_idx:
                        continue
                    rel = self._edge_type(dst_idx, dst[j])
                    if rel == "self":
                        allowed_next[i].append((j, 5))
                    elif rel == "teammate":
                        allowed_next[i].append((j, 6))
                    elif rel == "opponent":
                        allowed_next[i].append((j, 7))
                    elif rel == "out":
                        allowed_next[i].append((j, 8))
            else:
                # Case 3: non-self edge from a player A (0..21) to an outside node O (22..25)
                # The next edge must be either A->O or O->O
                allowed_next[i].append((i, 9))
                for j in range(K):
                    if src[j] == dst_idx and dst[j] == dst_idx:
                        allowed_next[i].append((j, 10))
                        break

        return allowed_next

    def _build_incoming(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert allowed_next into incoming padded tensors:
            inc_idx[j] = list of prev i that can go to j
        """
        allowed_next = self._build_allowed_next()
        K = self.K

        incoming = [[] for _ in range(K)]
        for i in range(K):
            for j, _ in allowed_next[i]:
                incoming[j].append(i)

        M = max(len(v) for v in incoming) if K > 0 else 1
        inc_idx = torch.full((K, M), -1, dtype=torch.long)
        inc_mask = torch.zeros((K, M), dtype=torch.bool)

        for j in range(K):
            prevs = incoming[j]
            if len(prevs) == 0:
                continue
            inc_idx[j, : len(prevs)] = torch.tensor(prevs, dtype=torch.long)
            inc_mask[j, : len(prevs)] = True

        return inc_idx, inc_mask

    def _logsumexp_masked(self, x: torch.Tensor, mask: torch.Tensor, dim=-1) -> torch.Tensor:
        neg = torch.full_like(x, self.forbid_score)
        x = torch.where(mask, x, neg)
        return torch.logsumexp(x, dim=dim)

    def _compute_logZ(self, emissions: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward algorithm in log-space using incoming lists.
        emissions: (B, T, K)
        """
        B, T, K = emissions.shape
        if mask is None:
            mask = emissions.new_ones((B, T), dtype=torch.bool)
        mask = mask.bool()

        alpha = emissions[:, 0]  # (B, K)

        inc_idx: torch.Tensor = self.inc_idx.clamp(min=0)  # (K, M)
        inc_idx_alpha = inc_idx.unsqueeze(0).expand(B, -1, -1)  # (B, K, M)
        inc_mask: torch.Tensor = self.inc_mask.unsqueeze(0).expand(B, -1, -1)  # (B, K, M)
        trans_score = self.transitions.unsqueeze(0).expand(B, -1, -1)  # (B, K, M)

        for t in range(1, T):
            emit_score = emissions[:, t]  # (B, K)

            prev_alpha = alpha.unsqueeze(1).expand(-1, K, -1)  # (B, K_curr, K_prev)
            prev_alpha = prev_alpha.gather(2, inc_idx_alpha)  # (B, K, M)

            score = self._logsumexp_masked(prev_alpha + trans_score, inc_mask, dim=2)  # (B, K)
            new_alpha = score + emit_score  # (B, K)
            alpha = torch.where(mask[:, t].unsqueeze(1), new_alpha, alpha)

        logZ = torch.logsumexp(alpha, dim=1)  # (B,)
        return logZ

    def _compute_gold_score(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, K = emissions.shape

        if mask is None:
            mask = emissions.new_ones((B, T), dtype=torch.bool)
        mask = mask.bool()

        score = emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)  # (B,)

        inc_idx: torch.Tensor = self.inc_idx.clamp(min=0)  # (K, M)
        inc_mask: torch.Tensor = self.inc_mask  # (K, M)
        trans = self.transitions  # (K, M)
        forbid_score = emissions.new_full((B,), self.forbid_score)

        for t in range(1, T):
            prev = tags[:, t - 1]  # (B,)
            curr = tags[:, t]  # (B,)

            emit_score = emissions[:, t].gather(1, curr.unsqueeze(1)).squeeze(1)

            inc_idx_t = inc_idx[curr]  # (B, M)
            inc_mask_t = inc_mask[curr]  # (B, M)
            trans_t = trans[curr]  # (B, M)

            match = (inc_idx_t == prev.unsqueeze(1)) & inc_mask_t  # (B, M)
            trans_score = torch.where(
                match,
                trans_t,
                trans_t.new_full(trans_t.shape, self.forbid_score),
            ).max(dim=1).values

            score += (emit_score + trans_score) * mask[:, t].float()

        return score  # (B,)

    def neg_log_likelihood(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        logZ = self._compute_logZ(emissions, mask)
        gold = self._compute_gold_score(emissions, tags, mask)
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
        mask: Optional[torch.Tensor] = None,
        last_node_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Viterbi decode using incoming lists.
        """
        B, T, K = emissions.shape
        if mask is None:
            mask = emissions.new_ones((B, T), dtype=torch.bool)
        mask = mask.bool()

        inc_idx: torch.Tensor = self.inc_idx.clamp(min=0)  # (K, M)
        inc_idx_delta = inc_idx.unsqueeze(0).expand(B, -1, -1)  # (B, K, M)
        inc_mask: torch.Tensor = self.inc_mask.unsqueeze(0).expand(B, -1, -1)  # (B, K, M)
        trans_score = self.transitions.unsqueeze(0).expand(B, -1, -1)  # (B, K, M)

        delta = emissions[:, 0]  # (B, K)
        forbid_score = emissions.new_full((B, K, inc_idx.size(1)), self.forbid_score)  # (B, K, M)
        backpointers: List[torch.Tensor] = []

        for t in range(1, T):
            emit_score = emissions[:, t]  # (B, K)

            prev_delta = delta.unsqueeze(1).expand(-1, K, -1)  # (B, K_curr, K_prev)
            prev_delta = prev_delta.gather(2, inc_idx_delta)  # (B, K, M)

            score = torch.where(inc_mask, prev_delta + trans_score, forbid_score)  # (B, K, M)
            best_score, best_m = score.max(dim=2)  # (B, K), (B, K)
            new_delta = emit_score + best_score  # (B, K)

            delta = torch.where(mask[:, t].unsqueeze(1), new_delta, delta)
            backpointers.append(best_m)

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
            best_m = backpointers[t - 1].gather(1, curr.unsqueeze(1)).squeeze(1)  # (B,)
            prev = inc_idx[curr, best_m]  # (B,)
            best_tags[:, t - 1] = prev

        return best_tags

    @torch.no_grad()
    def dump_gold_score(
        self,
        emissions: torch.Tensor,
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

            inc_idx_t = self.inc_idx[curr]  # (M,)
            inc_mask_t = self.inc_mask[curr]
            trans_t = self.transitions[curr]
            match = (inc_idx_t == prev) & inc_mask_t
            if not match.any().item():
                print(f"t={t} FORBIDDEN prev={prev} curr={curr}")
                trans_score = self.forbid_score
            else:
                trans_score = trans_t[match][0].item()

            total_score += emit_score + trans_score
            print(f"t={t} emit={emit_score:.6f} trans={trans_score:.6f}")

        print("total gold score =", total_score)
