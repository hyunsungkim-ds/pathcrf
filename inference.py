import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from datatools import config, postprocess, utils
from models.dynamic_dense_crf import DynamicDenseCRF
from models.dynamic_sparse_crf import DynamicSparseCRF
from models.edge_embed_crf import EdgeEmbedCRF
from models.set_lstm import SetLSTM
from models.static_dense_crf import StaticDenseCRF
from models.static_sparse_crf import StaticSparseCRF
from models.utils import build_allowed_mask, forbid_rate


def _add_outside_nodes(tracking: pd.DataFrame) -> pd.DataFrame:
    outside_xy = {
        "out_left": (0.0, config.PITCH_Y / 2),
        "out_right": (config.PITCH_X, config.PITCH_Y / 2),
        "out_bottom": (config.PITCH_X / 2, 0.0),
        "out_top": (config.PITCH_X / 2, config.PITCH_Y),
    }

    tracking = tracking.copy()
    for label, (x, y) in outside_xy.items():
        tracking[f"{label}_x"] = x
        tracking[f"{label}_y"] = y
        tracking[[f"{label}_vx", f"{label}_vy", f"{label}_speed", f"{label}_accel"]] = 0.0

    return tracking


def _build_node_indicators(
    node_order: List[str], left_team: str, right_team: str, left_gk: str, right_gk: str
) -> np.ndarray:
    indicators = np.zeros((len(node_order), 2), dtype=np.float32)
    for i, p in enumerate(node_order):
        if p.startswith(left_team):
            indicators[i, 0] = -1.0
        elif p.startswith(right_team):
            indicators[i, 0] = 1.0
        indicators[i, 1] = 1.0 if p in [left_gk, right_gk] else 0.0
    return indicators


def _pad_nodes(
    input_np: np.ndarray,
    node_order: List[str],
    left_team: str,
    right_team: str,
    team_size: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    T, N, F = input_np.shape
    max_nodes = team_size * 2 + 4
    if N >= max_nodes:
        player_mask = np.ones((T, max_nodes), dtype=np.float32)
        return input_np[:, :max_nodes, :], player_mask, node_order[:max_nodes]

    input_padded = np.zeros((T, max_nodes, F), dtype=np.float32)
    player_mask = np.zeros((T, max_nodes), dtype=np.float32)

    left_mask = np.array([p.startswith(left_team) for p in node_order], dtype=bool)
    right_mask = np.array([p.startswith(right_team) for p in node_order], dtype=bool)
    outside_mask = ~(left_mask | right_mask)

    left_idx = np.where(left_mask)[0]
    right_idx = np.where(right_mask)[0]
    outside_idx = np.where(outside_mask)[0]

    left_keep = min(len(left_idx), team_size)
    right_keep = min(len(right_idx), team_size)
    outside_keep = min(len(outside_idx), 4)

    if left_keep > 0:
        input_padded[:, :left_keep, :] = input_np[:, left_idx[:left_keep], :]
        player_mask[:, :left_keep] = 1.0

    if right_keep > 0:
        start = team_size
        input_padded[:, start : start + right_keep, :] = input_np[:, right_idx[:right_keep], :]
        player_mask[:, start : start + right_keep] = 1.0

    if outside_keep > 0:
        start = team_size * 2
        input_padded[:, start : start + outside_keep, :] = input_np[:, outside_idx[:outside_keep], :]
        player_mask[:, start : start + outside_keep] = 1.0

    left_names = [node_order[i] for i in left_idx[:left_keep]]
    right_names = [node_order[i] for i in right_idx[:right_keep]]
    outside_names = [node_order[i] for i in outside_idx[:outside_keep]]

    left_padded = left_names + [f"{left_team}_pad_{i}" for i in range(left_keep, team_size)]
    right_padded = right_names + [f"{right_team}_pad_{i}" for i in range(right_keep, team_size)]
    outside_padded = outside_names + [f"out_pad_{i}" for i in range(outside_keep, 4)]

    padded_order = left_padded + right_padded + outside_padded
    return input_padded, player_mask, padded_order


def _downsample_possession(
    ep_tracking: pd.DataFrame, down_idx: np.ndarray, poss_dict: Dict[str, int]
) -> Tuple[np.ndarray, np.ndarray]:
    poss_series = ep_tracking["player_id"].reset_index(drop=True)
    poss_series_down = []
    for i, curr_idx in enumerate(down_idx):
        prev_idx = down_idx[i - 1] if i > 0 else -1
        step_poss = poss_series.iloc[prev_idx + 1 : curr_idx + 1].dropna()
        last_poss = step_poss.iloc[-1] if not step_poss.empty else np.nan
        poss_series_down.append(last_poss)

    poss_player = pd.Series(poss_series_down)
    poss_prev = poss_player.ffill().bfill().map(poss_dict).to_numpy()
    poss_next = poss_player.bfill().ffill().map(poss_dict).to_numpy()

    if poss_prev.size > 1:
        is_loop_prev = poss_prev[:-1] == poss_next[:-1]
        is_loop_next = poss_prev[1:] == poss_next[1:]
        edge_change = poss_prev[:-1] != poss_prev[1:]
        forbidden_mask = is_loop_prev & is_loop_next & edge_change
        if np.any(forbidden_mask):
            poss_next[:-1] = np.where(forbidden_mask, poss_prev[1:], poss_next[:-1])

    return poss_prev, poss_next


def _detect_first_event(
    tracking: pd.DataFrame,
    edge_seq: pd.DataFrame,
    min_dur: int = 10,
) -> Optional[Dict[str, object]]:
    if edge_seq is None or edge_seq.empty:
        return None
    if not {"edge_src", "edge_dst"}.issubset(edge_seq.columns):
        return None

    tracking = tracking.copy()
    if "frame_id" in tracking.columns:
        tracking = tracking.set_index("frame_id").sort_index()

    edge_seq = edge_seq.copy()
    if "frame_id" in edge_seq.columns:
        edge_seq = edge_seq.set_index("frame_id").sort_index()
    edge_seq = edge_seq.loc[edge_seq.index.intersection(tracking.index)]
    if edge_seq.empty:
        return None

    edge_tuples = pd.Series(list(zip(edge_seq["edge_src"], edge_seq["edge_dst"])), index=edge_seq.index)
    edge_tuples = postprocess._overwrite_short_self_loops(edge_tuples, min_dur)
    if edge_tuples.empty:
        return None

    edge_codes = pd.Series(pd.factorize(edge_tuples)[0], index=edge_tuples.index)
    change_mask = edge_codes.diff().fillna(0) != 0
    first_edge = edge_tuples.iloc[0]
    change_mask.iloc[0] = first_edge[0] != first_edge[1]
    change_frames = edge_tuples.index[change_mask]

    for frame_id in change_frames:
        prev_edge = edge_tuples.shift(1).at[frame_id]
        curr_edge = edge_tuples.at[frame_id]
        if curr_edge == prev_edge:
            continue
        if curr_edge[0] == curr_edge[1] and prev_edge is None:
            continue

        if curr_edge[0] == curr_edge[1]:
            player_id = curr_edge[0] if postprocess._is_player(curr_edge[0]) else str(curr_edge[0]).split("_")[-1]
            event_type = "control" if postprocess._is_player(curr_edge[0]) else "out"
        elif postprocess._is_player(curr_edge[0]):
            player_id = curr_edge[0]
            event_type = "pass"
        else:
            continue

        start_x, start_y = postprocess._get_node_xy(tracking, frame_id, player_id)
        return {
            "frame_id": frame_id,
            "episode_id": tracking.at[frame_id, "episode_id"] if "episode_id" in tracking.columns else None,
            "player_id": player_id,
            "event_type": event_type,
            "start_x": start_x,
            "start_y": start_y,
        }

    return None


def _find_last_node_from_next_event(event: Optional[Dict[str, object]]) -> Optional[str]:
    if event is None:
        return None

    event_type = event["event_type"]
    is_home = event["player_id"].startswith("home_")
    is_away = event["player_id"].startswith("away_")

    if event_type == "goalkick":
        return "out_left" if is_home else "out_right"
    elif event_type == "corner":
        return "out_left" if is_away else "out_right"
    elif event_type == "throw_in":
        return "out_top" if float(event["start_y"]) >= config.PITCH_Y / 2.0 else "out_bottom"
    else:
        return None


def _upsample_xy(full_index: pd.Index, down_index: pd.Index, down_xy: np.ndarray) -> pd.DataFrame:
    pred_full = pd.DataFrame(index=full_index, columns=["ball_x", "ball_y"], dtype=float)
    pred_full.loc[down_index, ["ball_x", "ball_y"]] = down_xy
    pred_full = pred_full.interpolate(method="index").ffill().bfill()

    if len(down_index) == 0:
        return pred_full

    last_frame = down_index[-1]
    last_pos = pred_full.index.get_loc(last_frame)
    if last_pos < len(pred_full.index) - 1:
        if len(down_index) >= 2:
            prev_frame = down_index[-2]
            dt = float(last_frame - prev_frame)
            if dt <= 0:
                vel = np.array([0.0, 0.0], dtype=np.float32)
            else:
                vel = (down_xy[-1] - down_xy[-2]) / dt
        else:
            vel = np.array([0.0, 0.0], dtype=np.float32)

        last_val = pred_full.loc[last_frame, ["ball_x", "ball_y"]].to_numpy(dtype=float)
        for step, frame_id in enumerate(pred_full.index[last_pos + 1 :], start=1):
            pred_full.at[frame_id, "ball_x"] = last_val[0] + vel[0] * step
            pred_full.at[frame_id, "ball_y"] = last_val[1] + vel[1] * step

    return pred_full


def _run_model(
    model: torch.nn.Module,
    window_input_padded: np.ndarray,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
    input_pt = torch.tensor(window_input_padded, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        macro_out, micro_out, edge_embeds = model(input_pt)
        micro_out = micro_out.squeeze(0).detach().cpu()
        if macro_out is not None:
            macro_out = macro_out.squeeze(0).detach().cpu()
        if edge_embeds is not None:
            edge_embeds = edge_embeds.detach().cpu()
    return macro_out, micro_out, edge_embeds


def _window_weights(windows: List[Tuple[int, int]], idx: int) -> torch.Tensor:
    start, end = windows[idx]
    length = end - start
    weights = torch.ones(length, dtype=torch.float32)
    if idx > 0:
        overlap = windows[idx - 1][1] - start
        if overlap > 0:
            overlap = min(overlap, length)
            ramp = torch.linspace(0.0, 1.0, steps=overlap)
            weights[:overlap] = ramp
    if idx < len(windows) - 1:
        overlap = end - windows[idx + 1][0]
        if overlap > 0:
            overlap = min(overlap, length)
            ramp = torch.linspace(1.0, 0.0, steps=overlap)
            weights[-overlap:] = ramp
    return weights


def _build_allowed_next(allowed_mask: torch.Tensor) -> List[torch.Tensor]:
    allowed_next: List[torch.Tensor] = []
    for i in range(allowed_mask.size(0)):
        allowed_next.append(allowed_mask[i].nonzero(as_tuple=False).squeeze(1))
    return allowed_next


def _build_allowed_prev(allowed_mask: torch.Tensor) -> List[torch.Tensor]:
    allowed_prev: List[torch.Tensor] = []
    for i in range(allowed_mask.size(0)):
        allowed_prev.append(allowed_mask[:, i].nonzero(as_tuple=False).squeeze(1))
    return allowed_prev


def _greedy_constrained_decode(edge_logits: torch.Tensor, allowed_next: List[torch.Tensor]) -> torch.Tensor:
    if edge_logits.dim() != 2:
        raise ValueError(f"edge_logits should be 2D (T, K), got shape {tuple(edge_logits.shape)}")

    T = edge_logits.size(0)
    if T == 0:
        return torch.empty((0,), dtype=torch.long, device=edge_logits.device)

    if allowed_next and allowed_next[0].device != edge_logits.device:
        allowed_next = [c.to(edge_logits.device) for c in allowed_next]

    preds = torch.empty((T,), dtype=torch.long, device=edge_logits.device)
    preds[0] = torch.argmax(edge_logits[0])

    for t in range(1, T):
        prev = int(preds[t - 1].item())
        if prev < 0 or prev >= len(allowed_next):
            preds[t] = torch.argmax(edge_logits[t])
            continue

        candidates = allowed_next[prev]
        if candidates.numel() == 0:
            preds[t] = torch.argmax(edge_logits[t])
        else:
            best = torch.argmax(edge_logits[t].index_select(0, candidates))
            preds[t] = candidates[best]

    return preds


def _viterbi_constrained_decode(edge_logits: torch.Tensor, allowed_prev: List[torch.Tensor]) -> torch.Tensor:
    if edge_logits.dim() != 2:
        raise ValueError(f"edge_logits should be 2D (T, K), got shape {tuple(edge_logits.shape)}")

    T, K = edge_logits.shape
    if T == 0:
        return torch.empty((0,), dtype=torch.long, device=edge_logits.device)

    if allowed_prev and allowed_prev[0].device != edge_logits.device:
        allowed_prev = [c.to(edge_logits.device) for c in allowed_prev]

    neg_inf = float("-inf")
    dp = torch.full((T, K), neg_inf, dtype=edge_logits.dtype, device=edge_logits.device)
    back = torch.full((T, K), -1, dtype=torch.long, device=edge_logits.device)
    dp[0] = edge_logits[0]

    for t in range(1, T):
        prev_scores = dp[t - 1]
        for k in range(K):
            candidates = allowed_prev[k]
            if candidates.numel() == 0:
                continue
            scores = prev_scores.index_select(0, candidates)
            best_val, best_idx = scores.max(0)
            if torch.isneginf(best_val).item():
                continue
            dp[t, k] = edge_logits[t, k] + best_val
            back[t, k] = candidates[best_idx]

    last_scores = dp[T - 1]
    if torch.isneginf(last_scores).all().item():
        return edge_logits.argmax(dim=1)

    path = torch.empty((T,), dtype=torch.long, device=edge_logits.device)
    path[T - 1] = int(torch.argmax(last_scores).item())
    for t in range(T - 2, -1, -1):
        prev = back[t + 1, path[t + 1]]
        if int(prev.item()) < 0:
            prev = torch.argmax(edge_logits[t])
        path[t] = prev

    return path


def inference_episode(
    model: SetLSTM,
    ep_tracking: pd.DataFrame,
    left_gk: str = None,
    right_gk: str = None,
    use_crf: bool = True,
    decode: str = "indep",
    last_node: Optional[str] = None,
    evaluate: bool = True,
    window_seconds: Optional[float] = None,
    fps: float = 25.0,
    sample_freq: int = 5,
    device: torch.device = "cuda",
) -> Tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    List[str],
    Dict[str, float],
]:
    macro_type = getattr(model, "macro_type", None)
    micro_type = getattr(model, "micro_type", None)

    ep_tracking = ep_tracking.copy()
    if "out_left_x" not in ep_tracking.columns:
        ep_tracking = _add_outside_nodes(ep_tracking)

    if left_gk is None or right_gk is None:
        left_gk, right_gk = utils.detect_keepers(ep_tracking, left_first=True)

    valid_cols = ep_tracking.dropna(axis=1, how="all").columns
    left_team, right_team = left_gk.split("_")[0], right_gk.split("_")[0]
    left_players = sorted({c[:-2] for c in valid_cols if re.match(rf"{left_team}_.*_x", c)})
    right_players = sorted({c[:-2] for c in valid_cols if re.match(rf"{right_team}_.*_x", c)})
    outside_nodes = ["out_left", "out_right", "out_bottom", "out_top"]
    node_order = left_players + right_players + outside_nodes

    feat_dim = getattr(model, "feat_dim", None) or getattr(model, "node_in_dim", None) or 8
    feat_types = ["_x", "_y", "_vx", "_vy", "_speed", "_accel"][: max(feat_dim - 2, 0)]
    input_cols = [f"{p}{ft}" for p in node_order for ft in feat_types]

    if input_cols and ep_tracking[input_cols].isna().any().any():
        return (None, None, None, node_order, dict())

    n_nodes = len(node_order)

    indicators = _build_node_indicators(node_order, left_team, right_team, left_gk, right_gk)
    ep_indicators = np.broadcast_to(indicators[None, :, :], (len(ep_tracking), n_nodes, 2))
    ep_features = ep_tracking[input_cols].to_numpy(dtype=np.float32).reshape(len(ep_tracking), n_nodes, -1)
    ep_input = np.concatenate([ep_indicators, ep_features], axis=-1)

    down_idx = np.arange(0, len(ep_tracking), sample_freq, dtype=int)
    if len(down_idx) == 0 or down_idx[-1] != len(ep_tracking) - 1:
        down_idx = np.append(down_idx, len(ep_tracking) - 1)
    tracking_down = ep_tracking.iloc[down_idx]
    input_down = ep_input[down_idx]

    input_padded_down, _, padded_order = _pad_nodes(input_down, node_order, left_team, right_team, model.team_size)
    n_nodes_padded = len(padded_order)

    poss_prev = None
    poss_next = None

    if evaluate and (micro_type == "poss_edge" or macro_type in ["poss_prev", "poss_next", "poss_edge"]):
        poss_dict = {name: i for i, name in enumerate(padded_order)}
        if "out_left" in poss_dict:
            poss_dict["goal_left"] = poss_dict["out_left"]
        if "out_right" in poss_dict:
            poss_dict["goal_right"] = poss_dict["out_right"]
        poss_prev, poss_next = _downsample_possession(ep_tracking, down_idx, poss_dict)

    allowed_mask_orig = None
    allowed_next = None
    allowed_prev = None
    if micro_type == "poss_edge":
        src_list = [i // n_nodes_padded for i in range(n_nodes_padded * n_nodes_padded)]
        dst_list = [i % n_nodes_padded for i in range(n_nodes_padded * n_nodes_padded)]
        allowed_mask_orig = build_allowed_mask(src_list, dst_list, model.team_size)
        if (not use_crf) and decode == "greedy":
            allowed_next = _build_allowed_next(allowed_mask_orig)
        if (not use_crf) and decode == "viterbi":
            allowed_prev = _build_allowed_prev(allowed_mask_orig)

    if window_seconds is None:
        macro_out, micro_out, edge_embeds = _run_model(model, input_padded_down, device)
        window_acc_sum = 0.0
        window_count = 0

    else:
        window_len_ds = int(round(window_seconds * fps / sample_freq))
        stride_ds = int(round(0.5 * window_seconds * fps / sample_freq))
        window_len_ds = max(1, window_len_ds)
        stride_ds = max(1, stride_ds)

        total_len = input_down.shape[0]
        if total_len == 0:
            raise ValueError("Empty episode after downsampling.")
        if total_len <= window_len_ds:
            windows = [(0, total_len)]
        else:
            starts = list(range(0, total_len - window_len_ds + 1, stride_ds))
            if not starts:
                starts = [0]
            windows = []
            for i, start in enumerate(starts):
                end = total_len if i == len(starts) - 1 else start + window_len_ds
                windows.append((start, end))

        window_eval = evaluate and micro_type == "poss_edge" and poss_prev is not None and poss_next is not None
        if window_eval and use_crf:
            valid_edge_ids_window = model.valid_orig_edge_ids.to(device)
            orig2comp_window = model.orig2comp.detach().cpu().numpy()

        macro_accum = None
        macro_weights = None
        micro_accum = None
        micro_weights = None
        edge_accum = None
        edge_weights = None
        window_acc_sum = 0.0
        window_count = 0

        for idx, (start, end) in enumerate(windows):
            win_input_padded = input_padded_down[start:end]
            win_macro, win_micro, win_edge = _run_model(model, win_input_padded, device)
            weights = _window_weights(windows, idx)

            if window_eval:
                win_prev = poss_prev[start:end]
                win_next = poss_next[start:end]
                if not (pd.isna(win_prev).any() or pd.isna(win_next).any()):
                    edge_labels = win_prev.astype(np.int64) * len(padded_order) + win_next.astype(np.int64)
                    if use_crf:
                        win_micro_dev = win_micro.to(device)
                        emissions = win_micro_dev.index_select(dim=1, index=valid_edge_ids_window).unsqueeze(0)

                        if isinstance(model.crf, (EdgeEmbedCRF, DynamicDenseCRF, DynamicSparseCRF)):
                            if win_edge is None:
                                raise ValueError("CRF edge embeddings are missing for dynamic CRF.")
                            win_edge_dev = win_edge.to(device)
                            edge_embeds_comp = win_edge_dev.index_select(dim=2, index=valid_edge_ids_window)
                            decoded_window = model.crf.decode(emissions, edge_embeds_comp)
                        else:
                            assert isinstance(model.crf, (StaticDenseCRF, StaticSparseCRF))
                            decoded_window = model.crf.decode(emissions)

                        decoded_window = decoded_window.squeeze(0).cpu().numpy()
                        edge_labels_comp = orig2comp_window[edge_labels]
                        valid_mask = edge_labels_comp >= 0
                        if valid_mask.any():
                            window_acc_sum += float((decoded_window[valid_mask] == edge_labels_comp[valid_mask]).mean())
                            window_count += 1
                    else:
                        if decode == "greedy":
                            assert allowed_next is not None
                            pred_edges = _greedy_constrained_decode(win_micro, allowed_next).detach().cpu().numpy()
                        elif decode == "viterbi":
                            assert allowed_prev is not None
                            pred_edges = _viterbi_constrained_decode(win_micro, allowed_prev).detach().cpu().numpy()
                        else:
                            pred_edges = win_micro.argmax(dim=1).numpy()
                        window_acc_sum += float((pred_edges == edge_labels).mean())
                        window_count += 1

            if win_macro is not None:
                if macro_accum is None:
                    if win_macro.dim() == 3:
                        macro_size = (total_len, win_macro.shape[1], win_macro.shape[2])
                        macro_accum = torch.zeros(macro_size, dtype=win_macro.dtype)
                    else:
                        macro_size = (total_len, win_macro.shape[1])
                        macro_accum = torch.zeros(macro_size, dtype=win_macro.dtype)
                    macro_weights = torch.zeros(total_len, dtype=win_macro.dtype)
                if win_macro.dim() == 3:
                    macro_accum[start:end] += win_macro * weights[:, None, None]
                else:
                    macro_accum[start:end] += win_macro * weights[:, None]
                macro_weights[start:end] += weights

            if win_micro is not None:
                if micro_accum is None:
                    micro_accum = torch.zeros((total_len, win_micro.shape[1]), dtype=win_micro.dtype)
                    micro_weights = torch.zeros(total_len, dtype=win_micro.dtype)
                micro_accum[start:end] += win_micro * weights[:, None]
                micro_weights[start:end] += weights

            if win_edge is not None:
                if edge_accum is None:
                    if win_edge.dim() == 4:
                        edge_shape = (win_edge.shape[0], total_len, win_edge.shape[2], win_edge.shape[3])
                    else:
                        edge_shape = (total_len, win_edge.shape[1], win_edge.shape[2])
                    edge_accum = torch.zeros(edge_shape, dtype=win_edge.dtype)
                    edge_weights = torch.zeros(total_len, dtype=win_edge.dtype)
                if win_edge.dim() == 4:
                    edge_accum[:, start:end] += win_edge * weights.view(1, -1, 1, 1)
                else:
                    edge_accum[start:end] += win_edge * weights.view(-1, 1, 1)
                edge_weights[start:end] += weights

        if macro_accum is None:
            macro_out = None
        else:
            denom = torch.where(macro_weights == 0, torch.ones_like(macro_weights), macro_weights)
            if macro_accum.dim() == 3:
                macro_out = macro_accum / denom[:, None, None]
            else:
                macro_out = macro_accum / denom[:, None]

        if micro_accum is None:
            raise ValueError("Windowed inference produced no micro outputs.")
        denom = torch.where(micro_weights == 0, torch.ones_like(micro_weights), micro_weights)
        micro_out = micro_accum / denom[:, None]

        if edge_accum is None:
            edge_embeds = None
        else:
            denom = torch.where(edge_weights == 0, torch.ones_like(edge_weights), edge_weights)
            if edge_accum.dim() == 4:
                edge_embeds = edge_accum / denom.view(1, -1, 1, 1)
            else:
                edge_embeds = edge_accum / denom.view(-1, 1, 1)

    macro_prev_full = None
    macro_next_full = None

    def _macro_logits_to_full(macro_logits: torch.Tensor) -> pd.DataFrame:
        macro_probs = F.softmax(macro_logits, dim=-1).numpy()
        macro_down = pd.DataFrame(macro_probs, index=tracking_down.index, columns=padded_order)
        macro_full = macro_down.reindex(ep_tracking.index).interpolate(method="index").ffill().bfill()
        macro_full = macro_full.reindex(columns=padded_order)
        row_sums = macro_full.sum(axis=1).replace(0, 1.0)
        return macro_full.div(row_sums, axis=0)

    if macro_out is not None and macro_type in ["poss_prev", "poss_next"]:
        macro_full = _macro_logits_to_full(macro_out)
        if macro_type == "poss_prev":
            macro_prev_full = macro_full
        else:
            macro_next_full = macro_full
    elif macro_out is not None and macro_type == "poss_edge":
        if macro_out.dim() != 3 or macro_out.shape[-1] != 2:
            raise ValueError(f"Unexpected macro_out shape for poss_edge: {tuple(macro_out.shape)}")
        macro_prev_full = _macro_logits_to_full(macro_out[:, :, 0])
        macro_next_full = _macro_logits_to_full(macro_out[:, :, 1])

    if micro_type == "ball":
        micro_xy = micro_out.numpy()
        micro_full = _upsample_xy(ep_tracking.index, tracking_down.index, micro_xy)
    else:
        if use_crf:
            required = ["valid_orig_edge_ids", "orig2comp", "comp_src", "comp_dst", "crf"]
            missing = [name for name in required if not hasattr(model, name)]
            if missing:
                raise ValueError(f"CRF inference requires model.{', '.join(missing)}")

            valid_edge_ids = model.valid_orig_edge_ids.to(micro_out.device)
            emissions = micro_out.index_select(dim=1, index=valid_edge_ids).to(device).unsqueeze(0)

            if last_node in padded_order:
                last_node_idx = torch.tensor([padded_order.index(last_node)], device=device, dtype=torch.long)
            else:
                last_node_idx = None

            if isinstance(model.crf, (DynamicDenseCRF, DynamicSparseCRF)):
                if edge_embeds is None:
                    raise ValueError("CRF edge embeddings are missing for dynamic CRF.")
                edge_embeds_comp = edge_embeds.index_select(dim=2, index=valid_edge_ids).to(device)
                decoded = model.crf.decode(emissions, edge_embeds_comp, last_node_idx=last_node_idx).squeeze(0).cpu()
            else:
                assert isinstance(model.crf, (StaticDenseCRF, StaticSparseCRF))
                decoded = model.crf.decode(emissions, last_node_idx=last_node_idx).squeeze(0).cpu()
            decoded_np = decoded.numpy()
            comp_src = model.comp_src.detach().cpu().numpy()
            comp_dst = model.comp_dst.detach().cpu().numpy()

            micro_down = pd.DataFrame(index=tracking_down.index)
            micro_down["edge_src"] = [padded_order[i] for i in comp_src[decoded_np]]
            micro_down["edge_dst"] = [padded_order[i] for i in comp_dst[decoded_np]]

            micro_full = micro_down.reindex(ep_tracking.index).ffill().bfill()
        else:
            if decode in ["greedy", "viterbi"]:
                if decode == "greedy":
                    assert allowed_next is not None
                    pred_edges = _greedy_constrained_decode(micro_out, allowed_next).detach().cpu().numpy()
                else:  # decode == "viterbi"
                    assert allowed_prev is not None
                    pred_edges = _viterbi_constrained_decode(micro_out, allowed_prev).detach().cpu().numpy()
                src_pred = pred_edges // len(padded_order)
                dst_pred = pred_edges % len(padded_order)
                micro_down = pd.DataFrame(index=tracking_down.index)
                micro_down["edge_src"] = [padded_order[i] for i in src_pred]
                micro_down["edge_dst"] = [padded_order[i] for i in dst_pred]
                micro_full = micro_down.reindex(ep_tracking.index).ffill().bfill()
            else:
                edge_cols = [f"{src}-{dst}" for src in padded_order for dst in padded_order]
                edge_probs = F.softmax(micro_out, dim=-1).numpy()
                micro_down = pd.DataFrame(edge_probs, index=tracking_down.index, columns=edge_cols)
                micro_full = micro_down.reindex(ep_tracking.index).interpolate(method="index").ffill().bfill()
                row_sums = micro_full.sum(axis=1).replace(0, 1.0)
                micro_full = micro_full.div(row_sums, axis=0)

    stats = {
        "n_frames": 0,
        "correct_macro": 0,
        "correct_edges": 0,
        "correct_src": 0,
        "correct_dst": 0,
        "sum_pos_error": 0.0,
        "forbid_trans": 0.0,
        "total_trans": 0,
        "window_acc_sum": window_acc_sum,
        "window_count": window_count,
    }

    if evaluate:
        if macro_out is not None and macro_type in ["poss_prev", "poss_next", "poss_edge"]:
            if macro_type in ["poss_prev", "poss_next"]:
                poss_series = poss_prev if macro_type == "poss_prev" else poss_next
                if not pd.isna(poss_series).any():
                    target_nodes = poss_series.astype(np.int64)
                    stats["correct_macro"] = int((macro_out.argmax(dim=1).numpy() == target_nodes).sum())
            else:  # macro_type == "poss_edge"
                if not (pd.isna(poss_prev).any() or pd.isna(poss_next).any()):
                    if macro_out.dim() == 3 and macro_out.shape[-1] == 2:
                        src_logits = macro_out[:, :, 0]
                        dst_logits = macro_out[:, :, 1]
                        src_pred = src_logits.argmax(dim=1).numpy()
                        dst_pred = dst_logits.argmax(dim=1).numpy()
                        correct_mask = (src_pred == poss_prev) & (dst_pred == poss_next)
                        if use_crf and hasattr(model, "orig2comp"):
                            edge_labels = poss_prev * len(padded_order) + poss_next
                            edge_labels_comp = (
                                model.orig2comp.detach().cpu()[torch.tensor(edge_labels, dtype=torch.long)].numpy()
                            )
                            valid_mask = edge_labels_comp >= 0
                            correct_mask = correct_mask & valid_mask
                        stats["correct_macro"] = int(correct_mask.sum())

        pred_edges_noncrf = None
        if (not use_crf) and micro_type == "poss_edge":
            if decode == "greedy" and allowed_next is not None:
                pred_edges_noncrf = _greedy_constrained_decode(micro_out, allowed_next)
            elif decode == "viterbi" and allowed_prev is not None:
                pred_edges_noncrf = _viterbi_constrained_decode(micro_out, allowed_prev)
            else:
                pred_edges_noncrf = micro_out.argmax(dim=1)

        if micro_type == "ball":
            if {"ball_x", "ball_y"}.issubset(tracking_down.columns):
                target_xy = tracking_down[["ball_x", "ball_y"]].to_numpy(dtype=np.float32)
                stats["sum_pos_error"] = float(np.linalg.norm(micro_xy - target_xy, axis=1).sum())
                stats["n_frames"] = int(micro_out.shape[0])
        else:
            if not (pd.isna(poss_prev).any() or pd.isna(poss_next).any()):
                edge_labels = poss_prev * len(padded_order) + poss_next
                if use_crf:
                    orig2comp = model.orig2comp.detach().cpu()
                    edge_labels_comp = orig2comp[torch.tensor(edge_labels, dtype=torch.long)].numpy()
                    valid_mask = edge_labels_comp >= 0
                    if valid_mask.any():
                        decoded_np = decoded.numpy()
                        stats["correct_edges"] = int((decoded_np[valid_mask] == edge_labels_comp[valid_mask]).sum())
                        stats["n_frames"] = int(valid_mask.sum())
                        comp_src = model.comp_src.detach().cpu().numpy()
                        comp_dst = model.comp_dst.detach().cpu().numpy()
                        src_pred = comp_src[decoded_np]
                        dst_pred = comp_dst[decoded_np]
                        src_labels = poss_prev.astype(np.int64)
                        dst_labels = poss_next.astype(np.int64)
                        stats["correct_src"] = int((src_pred[valid_mask] == src_labels[valid_mask]).sum())
                        stats["correct_dst"] = int((dst_pred[valid_mask] == dst_labels[valid_mask]).sum())
                else:
                    stats["n_frames"] = int(len(edge_labels))
                    if pred_edges_noncrf is None:
                        pred_edges_noncrf = micro_out.argmax(dim=1)
                    pred_edges = pred_edges_noncrf.detach().cpu().numpy()
                    stats["correct_edges"] = int((pred_edges == edge_labels).sum())
                    src_pred = pred_edges // len(padded_order)
                    dst_pred = pred_edges % len(padded_order)
                    src_labels = poss_prev.astype(np.int64)
                    dst_labels = poss_next.astype(np.int64)
                    stats["correct_src"] = int((src_pred == src_labels).sum())
                    stats["correct_dst"] = int((dst_pred == dst_labels).sum())

            if micro_type == "poss_edge":
                allowed_mask_comp = None
                if hasattr(model, "comp_src") and hasattr(model, "comp_dst"):
                    src_list = model.comp_src.detach().cpu().tolist()
                    dst_list = model.comp_dst.detach().cpu().tolist()
                    allowed_mask_comp = build_allowed_mask(src_list, dst_list, model.team_size)

                if use_crf and allowed_mask_comp is not None:
                    pred_edges = decoded.cpu()
                    allowed_mask = allowed_mask_comp
                else:
                    if pred_edges_noncrf is None:
                        pred_edges_noncrf = micro_out.argmax(dim=1)
                    pred_edges = pred_edges_noncrf.cpu()
                    if allowed_mask_comp is not None and hasattr(model, "orig2comp"):
                        orig2comp = model.orig2comp.detach().cpu()
                        pred_edges = orig2comp[pred_edges]
                        allowed_mask = allowed_mask_comp
                    else:
                        allowed_mask = allowed_mask_orig

                pred_edges = pred_edges.view(1, -1)
                rate = forbid_rate(pred_edges, allowed_mask).item()
                total_trans = max(pred_edges.size(1) - 1, 0)
                stats["forbid_trans"] = rate * total_trans
                stats["total_trans"] = total_trans

    return macro_prev_full, macro_next_full, micro_full, padded_order, stats


def inference(
    model: SetLSTM,
    tracking: pd.DataFrame,
    use_crf: bool = True,
    decode: str = "indep",  # choices: [indep, greedy, viterbi]
    correct_episode_lasts: bool = False,
    evaluate: bool = True,
    window_seconds: Optional[float] = None,
    fps: float = 25.0,
    sample_freq: int = 5,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], pd.DataFrame, dict]:
    """
    Downsample to (fps / sample_freq), run the model, then upsample to original fps.
    If window_seconds is set, run sliding windows with 50% overlap and blend outputs linearly.
    Returns (macro_prev, macro_next, micro_pred, stats).
    For poss_prev, macro_next is None. For poss_next, macro_prev is None.
    For poss_edge, macro_prev and macro_next are both returned.
    For poss_edge, micro_pred_df columns are "src-dst" values per frame.
    If use_crf is True, the output stores decoded edge_src/edge_dst labels per frame (bfill upsampling).
    If use_crf is False, decode controls independent argmax ("indep") or constrained decoding ("greedy" or "viterbi"),
    Greedy/Viterbi decoding return edge_src/edge_dst labels per frame, just like CRF decoding.
    CRF inference is only supported for SetLSTM-style models that expose CRF buffers.
    """
    macro_type = getattr(model, "macro_type", None)
    micro_type = getattr(model, "micro_type", None)
    device = next(model.parameters()).device

    if macro_type not in [None, "poss_prev", "poss_next", "poss_edge"] or micro_type not in ["ball", "poss_edge"]:
        raise ValueError(f"Unsupported model types: macro={macro_type}, micro={micro_type}")
    if use_crf and getattr(model, "crf", None) is None:
        raise ValueError("use_crf=True requires a model with a configured CRF.")
    if not use_crf and decode not in ["indep", "greedy", "viterbi"]:
        raise ValueError(f"Unsupported decode={decode} when use_crf=False (use 'indep', 'greedy', or 'viterbi').")

    model.eval()
    tracking = tracking.copy().dropna(axis=1, how="all")

    if "episode_id" not in tracking.columns or "frame_id" not in tracking.columns:
        tracking, _ = utils.label_frames_and_episodes(tracking, fps=fps)
    if "phase_id" not in tracking.columns:
        tracking = utils.label_phases(tracking)

    tracking = tracking.set_index("frame_id").sort_index()
    tracking = _add_outside_nodes(tracking)

    macro_prev_df = None
    macro_next_df = None
    if macro_type in ["poss_prev", "poss_next", "poss_edge"]:
        home_players = sorted({c[:-2] for c in tracking.columns if re.match(r"home_.*_x", c)})
        away_players = sorted({c[:-2] for c in tracking.columns if re.match(r"away_.*_x", c)})
        outside_nodes = ["out_left", "out_right", "out_bottom", "out_top"]
        nodes = home_players + away_players + outside_nodes
        if macro_type in ["poss_prev", "poss_edge"]:
            macro_prev_df = pd.DataFrame(index=tracking.index, columns=nodes, dtype=float)
        if macro_type in ["poss_next", "poss_edge"]:
            macro_next_df = pd.DataFrame(index=tracking.index, columns=nodes, dtype=float)

    if micro_type == "ball":
        micro_pred_df = pd.DataFrame(index=tracking.index, columns=["ball_x", "ball_y"], dtype=float)
    else:
        micro_pred_df = pd.DataFrame(index=tracking.index, dtype=float)

    stats = {"n_frames": 0}
    correct_macro = 0
    correct_edges = 0
    correct_src = 0
    correct_dst = 0
    sum_pos_error = 0.0
    window_acc_sum = 0.0
    window_count = 0
    forbid_trans = 0.0
    total_trans = 0

    for phase in sorted(tracking["phase_id"].unique(), reverse=True):
        phase_tracking = tracking[tracking["phase_id"] == phase]
        if phase == 0 or phase_tracking.empty:
            continue

        left_gk, right_gk = utils.detect_keepers(phase_tracking, left_first=True)

        episodes = sorted([e for e in phase_tracking["episode_id"].unique() if e > 0], reverse=True)
        next_ep_event = None

        for episode in tqdm(episodes, desc=f"Phase {phase}"):
            ep_tracking: pd.DataFrame = phase_tracking[phase_tracking["episode_id"] == episode].sort_index()
            last_node = _find_last_node_from_next_event(next_ep_event) if correct_episode_lasts else None
            macro_prev_full, macro_next_full, micro_full, _, ep_stats = inference_episode(
                model=model,
                ep_tracking=ep_tracking,
                left_gk=left_gk,
                right_gk=right_gk,
                use_crf=use_crf,
                decode=decode,
                last_node=last_node,
                evaluate=evaluate,
                window_seconds=window_seconds,
                fps=fps,
                sample_freq=sample_freq,
                device=device,
            )

            if macro_prev_full is not None and macro_prev_df is not None:
                missing_cols = [c for c in macro_prev_full.columns if c not in macro_prev_df.columns]
                if missing_cols:
                    new_block = pd.DataFrame(np.nan, index=macro_prev_df.index, columns=missing_cols)
                    macro_prev_df = pd.concat([macro_prev_df, new_block], axis=1)
                    macro_prev_df = macro_prev_df.copy()
                macro_prev_df.loc[ep_tracking.index, macro_prev_full.columns] = macro_prev_full.values

            if macro_next_full is not None and macro_next_df is not None:
                missing_cols = [c for c in macro_next_full.columns if c not in macro_next_df.columns]
                if missing_cols:
                    new_block = pd.DataFrame(np.nan, index=macro_next_df.index, columns=missing_cols)
                    macro_next_df = pd.concat([macro_next_df, new_block], axis=1)
                    macro_next_df = macro_next_df.copy()
                macro_next_df.loc[ep_tracking.index, macro_next_full.columns] = macro_next_full.values

            if micro_full is not None:
                if micro_type == "ball":
                    micro_pred_df.loc[ep_tracking.index, micro_full.columns] = micro_full.values
                elif use_crf or decode in ["greedy", "viterbi"]:
                    for col in micro_full.columns:
                        if col not in micro_pred_df.columns:
                            micro_pred_df[col] = pd.Series(index=micro_pred_df.index, dtype=object)
                    micro_pred_df.loc[ep_tracking.index, micro_full.columns] = micro_full.values
                else:
                    new_cols = [c for c in micro_full.columns if c not in micro_pred_df.columns]
                    if new_cols:
                        new_block = pd.DataFrame(np.nan, index=micro_pred_df.index, columns=new_cols)
                        micro_pred_df = pd.concat([micro_pred_df, new_block], axis=1)
                        micro_pred_df = micro_pred_df.copy()
                    micro_pred_df.loc[ep_tracking.index, micro_full.columns] = micro_full.values

            if evaluate and len(ep_tracking) > 0:
                stats["n_frames"] += ep_stats["n_frames"]
                correct_macro += ep_stats["correct_macro"]
                correct_edges += ep_stats["correct_edges"]
                correct_src += ep_stats["correct_src"]
                correct_dst += ep_stats["correct_dst"]
                sum_pos_error += ep_stats["sum_pos_error"]
                window_acc_sum += ep_stats["window_acc_sum"]
                window_count += ep_stats["window_count"]
                forbid_trans += ep_stats["forbid_trans"]
                total_trans += ep_stats["total_trans"]

            if correct_episode_lasts:
                if micro_full is not None and {"edge_src", "edge_dst"}.issubset(micro_full.columns):
                    first_event = _detect_first_event(ep_tracking, micro_full)
                    if first_event is not None:
                        first_event_df = postprocess.classify_setpieces(pd.DataFrame([first_event]))
                        first_event["event_type"] = first_event_df.iloc[0]["event_type"]
                else:
                    first_event = None
                next_ep_event = first_event

    if evaluate and stats["n_frames"] > 0:
        if macro_type in ["poss_prev", "poss_next", "poss_edge"]:
            stats["macro_acc"] = round(correct_macro / stats["n_frames"], 4)
        if micro_type == "poss_edge":
            stats["edge_acc"] = round(correct_edges / stats["n_frames"], 4)
            stats["src_acc"] = round(correct_src / stats["n_frames"], 4)
            stats["dst_acc"] = round(correct_dst / stats["n_frames"], 4)
            if total_trans > 0:
                stats["forbid_rate"] = round(forbid_trans / total_trans, 4)
        if micro_type == "ball":
            stats["pos_error"] = round(sum_pos_error / stats["n_frames"], 4)
    if evaluate and window_seconds is not None and window_count > 0:
        stats["edge_acc_window"] = round(window_acc_sum / window_count, 4)

    return macro_prev_df, macro_next_df, micro_pred_df, stats
