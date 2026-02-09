import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data


def num_trainable_params(model: nn.Module) -> int:
    total = 0
    for p in model.parameters():
        count = 1
        for s in p.size():
            count *= s
        total += count
    return total


def collate_window_batch(
    batch: Sequence[Tuple[List[Data], Optional[torch.Tensor], torch.Tensor, dict]],
) -> Tuple[List[Batch], torch.Tensor, Optional[torch.Tensor], Sequence[dict]]:
    data_list, macro_list, micro_list, meta_list = zip(*batch)
    T = len(data_list[0])

    batch_list = []
    for t in range(T):
        batch_list.append(Batch.from_data_list([data_list[i][t] for i in range(len(data_list))]))

    macro_target = None if macro_list[0] is None else torch.stack(macro_list, dim=0)  # (B, T)
    micro_target = torch.stack(micro_list, dim=0)  # (B, T, 2)
    return batch_list, macro_target, micro_target, meta_list


def printlog(line: str, save_path: str):
    print(line)
    with open(save_path + "/log.txt", "a") as file:
        file.write(line + "\n")


def loss_str(losses: dict) -> str:
    ret = ""
    for key, value in losses.items():
        ret += f" {key}: {np.mean(value):.4f} |"
    return ret[:-2]


def calc_dist(pred_xy: torch.Tensor, target_xy: torch.Tensor, aggfunc="mean") -> float:
    if aggfunc == "mean":
        return torch.norm(pred_xy - target_xy, dim=-1).mean().item()
    else:  # if aggfunc == "sum":
        return torch.norm(pred_xy - target_xy, dim=-1).sum().item()


def calc_class_acc(pred_poss: torch.Tensor, target_poss: torch.Tensor, aggfunc="mean") -> float:
    if aggfunc == "mean":
        return (torch.argmax(pred_poss, dim=1) == target_poss).float().mean().item()
    else:  # if aggfunc == "sum":
        return (torch.argmax(pred_poss, dim=1) == target_poss).float().sum().item()


def calc_real_loss(
    pred_xy: torch.Tensor,
    model_input: torch.Tensor,
    player_mask: Optional[torch.Tensor] = None,
    eps: torch.Tensor = torch.tensor(1e-6),
    aggfunc: str = "mean",
) -> torch.Tensor:
    eps = eps.to(pred_xy.device)
    if len(pred_xy.shape) == 2:
        pred_xy = pred_xy.unsqueeze(0).clone()  # (T, 2) to (1, T, 2)

    if len(model_input.shape) == 3:
        model_input = model_input.unsqueeze(0).clone()  # (T, N, F) to (1, T, N, F)

    # Calculate the angle between two consecutive velocity vectors
    # We skip the division by time difference, which is eventually reduced
    vels = pred_xy.diff(dim=1)
    speeds = torch.linalg.norm(vels, dim=-1)
    cos_num = torch.sum(vels[:, :-1] * vels[:, 1:], dim=-1) + eps
    cos_denom = speeds[:, :-1] * speeds[:, 1:] + eps
    cosines = torch.clamp(cos_num / cos_denom, -1 + eps, 1 - eps)
    angles = torch.acos(cosines)

    # Compute the distance between the ball and the nearest valid node
    pred_xy = torch.unsqueeze(pred_xy, dim=2)
    player_xy = model_input[..., 2:4] if model_input.size(-1) >= 4 else model_input[..., :2]
    ball_dists = torch.linalg.norm(pred_xy - player_xy, dim=-1)
    if player_mask is not None:
        if len(player_mask.shape) == 2:
            player_mask = player_mask.unsqueeze(0)  # (T, N) to (1, T, N)
        ball_dists = ball_dists.masked_fill(player_mask == 0, 1e6)
    nearest_dists = torch.min(ball_dists, dim=-1).values[:, 1:-1]

    # Either course angle must be close to 0 or the ball must be close to a player
    if aggfunc == "mean":
        return (torch.tanh(angles) * nearest_dists).mean()
    else:  # if aggfunc == "sum"
        return (torch.tanh(angles) * nearest_dists).sum()


def l1_regularizer(model: nn.Module) -> torch.Tensor:
    l1_loss = 0
    for model_param_name, model_param_value in model.named_parameters():
        if model_param_name.endswith("weight"):
            l1_loss += model_param_value.abs().sum()
    return l1_loss


def load_trial_args(save_path: str) -> Dict:
    args_path = os.path.join(save_path, "args.json")
    if not os.path.isfile(args_path):
        raise FileNotFoundError(f"args.json not found at {args_path}")
    with open(args_path, "r") as f:
        return json.load(f)


def resolve_model_path(save_path: str, model_file: str) -> str:
    if os.path.isfile(model_file):
        return model_file
    candidate = os.path.join(save_path, "model", model_file)
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(f"Model file not found: {model_file} or {candidate}")


def build_model(trial_args: Dict, device: torch.device) -> nn.Module:
    agent_model = trial_args.get("agent_model")
    macro_type = trial_args.get("macro_type", None)

    coarse_dim = trial_args.get("coarse_dim", trial_args.get("macro_fpe_dim", 16))
    fine_dim = trial_args.get("fine_dim", trial_args.get("micro_conv_dim", 64))
    seq_dim = trial_args.get("seq_dim", trial_args.get("rnn_dim", 128))
    att_heads = trial_args.get("att_heads", trial_args.get("gnn_heads", 4))

    if agent_model == "gat":
        from .gat_lstm import GATLSTM

        model = GATLSTM(
            macro_type=macro_type,
            micro_type=trial_args.get("micro_type", "ball"),
            seq_model=trial_args.get("seq_model", "bi_lstm"),
            node_in_dim=trial_args.get("node_in_dim", 8),
            edge_in_dim=trial_args.get("edge_in_dim", 0),
            team_size=trial_args.get("team_size", 11),
            gat_heads=att_heads,
            macro_node_dim=coarse_dim,
            macro_graph_dim=coarse_dim,
            micro_gnn_dim=fine_dim,
            macro_rnn_dim=seq_dim,
            micro_rnn_dim=seq_dim,
            dropout=trial_args.get("dropout", 0.1),
        ).to(device)

    elif agent_model and agent_model.startswith("set"):
        from .set_lstm import SetLSTM

        model = SetLSTM(
            macro_type=macro_type,
            micro_type=trial_args.get("micro_type", "ball"),
            seq_model_type=trial_args.get("seq_model", "bi_lstm"),
            crf_model_type=trial_args.get("crf_model", None),
            feat_dim=trial_args.get("node_in_dim", 8),
            team_size=trial_args.get("team_size", 11),
            macro_ppe_dim=coarse_dim,
            macro_fpe_dim=coarse_dim,
            micro_enc_dim=fine_dim,
            macro_seq_dim=seq_dim,
            micro_seq_dim=seq_dim,
            crf_edge_dim=trial_args.get("crf_edge_dim", 16),
            dropout=trial_args.get("dropout", 0.1),
        ).to(device)

    elif agent_model == "transportmer":
        from .transportmer import TranSPORTmer

        model = TranSPORTmer(
            macro_type=macro_type,
            micro_type=trial_args.get("micro_type", "ball"),
            crf_model_type=trial_args.get("crf_model", None),
            feat_dim=trial_args.get("node_in_dim", 8),
            team_size=trial_args.get("team_size", 11),
            sab_heads=att_heads,
            coarse_dim=coarse_dim,
            fine_dim=fine_dim,
            crf_edge_dim=trial_args.get("crf_edge_dim", 16),
            dropout=trial_args.get("dropout", 0.1),
        ).to(device)

    else:
        raise ValueError(f"Unsupported agent_model: {agent_model}")

    return model


def build_edge_compression_maps(
    n_players: int = 22, n_outsides: int = 4, device=None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Original nodes: 0..(n_players-1) players, n_players..n_players+n_outside-1 outside
    Original edge_id: src*N + dst where N = n_players + n_outside (26)

    Returns:
      valid_orig_edge_ids: (K,) long tensor of original edge_ids that are kept (K=576)
      orig2comp: (N*N,) long tensor mapping original edge_id -> compressed_id or -1
      comp_src: (K,) long tensor of src node id (0..25)
      comp_dst: (K,) long tensor of dst node id (0..25)
    """
    n_nodes = n_players + n_outsides  # 26
    K = n_players * n_nodes + n_outsides  # 22 * 26 + 4 = 576

    valid_orig_edge_ids = []
    comp_src = []
    comp_dst = []

    # Include all edges from a player (22 * 26 = 572)
    for s in range(n_players):
        for d in range(n_nodes):
            valid_orig_edge_ids.append(s * n_nodes + d)
            comp_src.append(s)
            comp_dst.append(d)

    # Include self-loops of outside nodes (4)
    for o in range(n_players, n_nodes):
        valid_orig_edge_ids.append(o * n_nodes + o)
        comp_src.append(o)
        comp_dst.append(o)

    valid_orig_edge_ids = torch.tensor(valid_orig_edge_ids, dtype=torch.long)
    comp_src = torch.tensor(comp_src, dtype=torch.long)
    comp_dst = torch.tensor(comp_dst, dtype=torch.long)

    orig2comp = torch.full((n_nodes * n_nodes,), -1, dtype=torch.long)
    orig2comp[valid_orig_edge_ids] = torch.arange(K, dtype=torch.long)

    if device is not None:
        valid_orig_edge_ids = valid_orig_edge_ids.to(device)
        comp_src = comp_src.to(device)
        comp_dst = comp_dst.to(device)
        orig2comp = orig2comp.to(device)

    return valid_orig_edge_ids, orig2comp, comp_src, comp_dst


def build_allowed_mask(src_list, dst_list, team_size: int) -> torch.Tensor:
    K = len(src_list)
    allowed = torch.zeros((K, K), dtype=torch.bool)
    limit = 2 * team_size

    for i in range(K):
        src_i = src_list[i]
        dst_i = dst_list[i]

        if src_i >= limit:
            if src_i == dst_i:
                allowed[i, i] = True
            continue

        if src_i == dst_i:
            # Case 1: self-loop of a player A -> next edge must originate from A
            for j in range(K):
                if src_list[j] == src_i:
                    allowed[i, j] = True
            continue

        if dst_i < limit:
            # Case 2: A->B (player) -> same edge or edges from B
            allowed[i, i] = True
            for j in range(K):
                if src_list[j] == dst_i:
                    allowed[i, j] = True
        else:
            # Case 3: A->O -> same edge or O->O
            allowed[i, i] = True
            for j in range(K):
                if src_list[j] == dst_i and dst_list[j] == dst_i:
                    allowed[i, j] = True
                    break

    return allowed


def forbid_rate(pred_edges: torch.Tensor, allowed_mask: torch.Tensor) -> torch.Tensor:
    if pred_edges.size(1) < 2:
        return torch.tensor(0.0, device=pred_edges.device)
    prev = pred_edges[:, :-1]
    nxt = pred_edges[:, 1:]
    valid = (prev >= 0) & (nxt >= 0)

    safe_prev = prev.clone()
    safe_nxt = nxt.clone()
    safe_prev[~valid] = 0
    safe_nxt[~valid] = 0

    allowed = allowed_mask[safe_prev, safe_nxt]
    forbid = (~valid) | (~allowed)
    return forbid.float().mean()
