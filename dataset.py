import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm

from datatools import config, utils


def _complete_edge_index(N: int, self_loops: bool = False) -> torch.Tensor:
    """
    Fully-connected directed edge_index (2, E).
    """
    if self_loops:
        src, dst = np.where(np.ones((N, N), dtype=bool))
    else:
        mask = ~np.eye(N, dtype=bool)
        src, dst = np.where(mask)
    return torch.tensor(np.vstack([src, dst]).astype(np.int64), dtype=torch.long)


class SoccerWindowDataset(Dataset):
    """
    Abstract dataset class that builds and stores both raw and padded samples.
    Subclasses decide whether to return graph batches or padded tensors.
    """

    def __init__(
        self,
        data_paths: List[str],
        node_in_dim: int = 8,
        edge_in_dim: int = 0,
        team_size: int = 11,
        fps: float = 25.0,
        sample_freq: int = 5,
        window_seconds: float = 10.0,
        window_stride: int = 1,
        self_loops: bool = True,
        flip_pitch: bool = False,
        verbose: bool = True,
    ):
        self.data_paths = list(data_paths)
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.team_size = team_size if team_size is not None else 11
        self.fps = fps
        self.sample_freq = sample_freq
        self.window_size = int(round(window_seconds * fps / sample_freq))
        self.window_stride = int(window_stride)
        self.self_loops = self_loops
        self.flip_pitch = flip_pitch

        self.feature_types = ["_x", "_y", "_vx", "_vy", "_speed", "_accel"]
        used_feature_types = self.feature_types[: node_in_dim - 2]  # Two node indicators will be attached later
        self.used_feature_types = used_feature_types

        # Offsets for positional/velocity features inside node feature vectors
        self._idx_x = 2 + used_feature_types.index("_x") if "_x" in used_feature_types else None
        self._idx_y = 2 + used_feature_types.index("_y") if "_y" in used_feature_types else None
        self._idx_vx = 2 + used_feature_types.index("_vx") if "_vx" in used_feature_types else None
        self._idx_vy = 2 + used_feature_types.index("_vy") if "_vy" in used_feature_types else None

        self.outside_nodes = {
            "out_left": (0, config.PITCH_Y / 2),
            "out_right": (config.PITCH_X, config.PITCH_Y / 2),
            "out_bottom": (config.PITCH_X / 2, 0),
            "out_top": (config.PITCH_X / 2, config.PITCH_Y),
        }

        self.samples: List[dict] = []
        self._edge_cache_pt = {}  # N -> edge_index tensor
        self._edge_cache_np = {}  # N -> edge_index numpy

        for f in tqdm(self.data_paths, desc="Building window samples", disable=not verbose):
            tracking = pd.read_parquet(f)
            phases = utils.summarize_phases(tracking)

            for k, xy in self.outside_nodes.items():
                tracking[f"{k}_x"] = xy[0]
                tracking[f"{k}_y"] = xy[1]
                tracking[[f"{k}_vx", f"{k}_vy", f"{k}_speed", f"{k}_accel"]] = 0

            for phase, row in phases.iterrows():
                active_players: List[str] = row["active_players"]
                if len(active_players) == 0:
                    continue

                phase_tracking = tracking[tracking["phase_id"] == phase].copy()

                # Reorder teams so that the left team comes first
                left_gk, right_gk = utils.detect_keepers(phase_tracking, left_first=True)
                left_team, right_team = left_gk.split("_")[0], right_gk.split("_")[0]
                left_players = [p for p in active_players if p.startswith(left_team)]
                right_players = [p for p in active_players if p.startswith(right_team)]
                node_order = left_players + right_players + list(self.outside_nodes.keys())

                poss_dict = dict(zip(node_order, np.arange(len(node_order)).astype(np.int16)))
                poss_dict["goal_left"] = np.int16(len(node_order) - 4)  # Same as out_left
                poss_dict["goal_right"] = np.int16(len(node_order) - 3)  # Same as out_right

                node_cols = [f"{p}{ft}" for p in node_order for ft in used_feature_types]
                n_nodes = len(node_order)

                node_indicators = np.zeros((n_nodes, 2), dtype=np.float32)
                for i, p in enumerate(node_order):
                    if p.startswith(left_team):
                        node_indicators[i, 0] = -1.0
                    elif p.startswith(right_team):
                        node_indicators[i, 0] = 1.0
                    node_indicators[i, 1] = 1.0 if p in [left_gk, right_gk] else 0.0

                for episode in phase_tracking["episode_id"].unique():
                    ep_tracking = tracking[tracking["episode_id"] == episode].copy()
                    if episode == 0 or len(ep_tracking) < window_seconds * fps:
                        continue

                    ep_node_indicators = np.broadcast_to(node_indicators[None, :, :], (len(ep_tracking), n_nodes, 2))
                    ep_features = ep_tracking[node_cols].values.reshape(len(ep_tracking), n_nodes, -1)
                    ep_features = np.concatenate([ep_node_indicators, ep_features], axis=-1)  # (T_ep, N, F)
                    ep_ball = ep_tracking[["ball_x", "ball_y"]].to_numpy(dtype=np.float32)  # (T_ep, 2)

                    if ep_tracking["player_id"].dropna().empty:
                        # Skip episodes where player_id is missing for the entire clip
                        continue

                    ep_poss = ep_tracking["player_id"].reset_index(drop=True)
                    down_idx = np.arange(0, len(ep_tracking), sample_freq, dtype=np.int64)
                    if len(down_idx) == 0 or down_idx[-1] != len(ep_tracking) - 1:
                        down_idx = np.append(down_idx, len(ep_tracking) - 1)

                    poss_player = []
                    for i, curr_idx in enumerate(down_idx):
                        prev_idx = down_idx[i - 1] if i > 0 else -1
                        step_poss = ep_poss.iloc[prev_idx + 1 : curr_idx + 1].dropna()
                        last_poss = step_poss.iloc[-1] if not step_poss.empty else np.nan
                        poss_player.append(last_poss)

                    poss_player = pd.Series(poss_player)
                    poss_prev = poss_player.ffill().bfill().map(poss_dict).to_numpy()
                    poss_next = poss_player.bfill().ffill().map(poss_dict).to_numpy()
                    poss_team = ep_tracking["ball_owning_team_id"].iloc[down_idx] == right_team
                    poss_team = poss_team.astype(np.int16).to_numpy()

                    is_loop_prev = poss_prev[:-1] == poss_next[:-1]
                    is_loop_next = poss_prev[1:] == poss_next[1:]
                    edge_change = poss_prev[:-1] != poss_prev[1:]
                    forbidden_mask = is_loop_prev & is_loop_next & edge_change
                    if np.any(forbidden_mask):
                        # Replace A->A followed by B->B with A->B to obey the CRF edge transition rule
                        poss_next[:-1] = np.where(forbidden_mask, poss_prev[1:], poss_next[:-1])

                    ep_features_down = ep_features[down_idx]
                    ep_poss_down = np.stack([poss_prev, poss_next, poss_team], axis=1)
                    ep_ball_down = ep_ball[down_idx]

                    if len(ep_features_down) < self.window_size:
                        continue

                    starts_down = list(range(0, len(ep_features_down) - self.window_size + 1, self.window_stride))
                    starts_down = starts_down[:-1] + [len(ep_features_down) - self.window_size]

                    for start_down in starts_down:
                        end_down = start_down + self.window_size
                        input_seq = ep_features_down[start_down:end_down]  # (T, N, F)
                        poss_seq = ep_poss_down[start_down:end_down]  # (T, 3)
                        ball_seq = ep_ball_down[start_down:end_down]  # (T, 2)

                        start = int(down_idx[start_down])
                        end = int(down_idx[end_down - 1]) + 1

                        self.samples.append(
                            {
                                "input": input_seq,  # np (T, N, F)
                                "poss": poss_seq,  # np (T, 3)
                                "ball": ball_seq,  # np (T, 2)
                                "meta": {
                                    "file": f,
                                    "phase_id": int(phase),
                                    "episode_id": int(episode),
                                    "start_in_episode": int(start),
                                    "end_in_episode": int(end),
                                    "start_frame_id": int(ep_tracking.iloc[start]["frame_id"]),
                                    "end_frame_id": int(ep_tracking.iloc[end - 1]["frame_id"]),
                                    "left_team": left_team,
                                    "right_team": right_team,
                                    "node_order": node_order,
                                },
                            }
                        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        raise NotImplementedError("Use SoccerWindowGraphs or SoccerWindowTensors.")


class SoccerWindowGraphs(SoccerWindowDataset):
    """Returns per-time-step PyG graphs (for DGNN)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx: int) -> Tuple[List[Data], Optional[torch.Tensor], torch.Tensor, dict]:
        s = self.samples[idx]
        input_np = s["input"].copy()  # (T, N, F) unpadded
        poss_np = s["poss"]  # (T, 3)
        ball_np = s["ball"].copy()  # (T, 2)

        input_np, ball_np = self._random_flip(input_np, ball_np)

        T, N, _ = input_np.shape

        # edge_index: fully-connected (cached by N)
        if N not in self._edge_cache_pt:
            self._edge_cache_pt[N] = _complete_edge_index(N, self_loops=self.self_loops)
            self._edge_cache_np[N] = self._edge_cache_pt[N].cpu().numpy()

        edge_index_pt = self._edge_cache_pt[N]
        edge_index_np = self._edge_cache_np[N]

        # Feature graphs: List[Data] of length T
        input_list = []
        for t in range(T):
            node_attr = torch.tensor(input_np[t], dtype=torch.float32)

            if self.edge_in_dim > 0:
                team_flag = input_np[t, :, 0]
                pos_xy = input_np[t, :, 2:4]

                src_idx = edge_index_np[0]
                dst_idx = edge_index_np[1]
                same_team = (team_flag[src_idx] * team_flag[dst_idx] > 0).astype(np.float32)
                distances = np.linalg.norm(pos_xy[src_idx] - pos_xy[dst_idx], axis=1).astype(np.float32)
                edge_attr = torch.tensor(np.stack([same_team, distances], axis=1), dtype=torch.float32)

                input_list.append(Data(x=node_attr, edge_index=edge_index_pt, edge_attr=edge_attr))

            else:
                input_list.append(Data(x=node_attr, edge_index=edge_index_pt))

        poss_pt = torch.tensor(poss_np, dtype=torch.long) if poss_np is not None else None  # (T,)
        ball_pt = torch.tensor(ball_np, dtype=torch.float32)  # (T, 2)

        return input_list, poss_pt, ball_pt, s["meta"]

    def _random_flip(self, input_np: np.ndarray, ball_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random pitch flips to coordinates only (used by graph dataset)."""
        if not self.flip_pitch:
            return input_np, ball_np

        flip_x = bool(np.random.randint(0, 2))
        flip_y = bool(np.random.randint(0, 2))

        if flip_x and self._idx_x is not None:
            input_np[:, :, self._idx_x] = config.PITCH_X - input_np[:, :, self._idx_x]
            if self._idx_vx is not None:
                input_np[:, :, self._idx_vx] = -input_np[:, :, self._idx_vx]
            ball_np[:, 0] = config.PITCH_X - ball_np[:, 0]

        if flip_y and self._idx_y is not None:
            input_np[:, :, self._idx_y] = config.PITCH_Y - input_np[:, :, self._idx_y]
            if self._idx_vy is not None:
                input_np[:, :, self._idx_vy] = -input_np[:, :, self._idx_vy]
            ball_np[:, 1] = config.PITCH_Y - ball_np[:, 1]

        return input_np, ball_np


class SoccerWindowTensors(SoccerWindowDataset):
    """Returns padded tensor inputs with masks for set/sequence models."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Pad samples post-construction so graph subclass stays untouched
        padded_samples = []
        for s in self.samples:
            input = s["input"]
            poss = s["poss"]
            meta = s["meta"]
            input_padded, poss_padded, player_mask = self._pad_sample(input, poss, meta)

            s["input"] = input_padded  # Overwrite with padded version
            s["poss"] = poss_padded
            s["player_mask"] = player_mask
            padded_samples.append(s)

        self.samples = padded_samples

    def _pad_sample(
        self, input: np.ndarray, poss: Optional[np.ndarray], meta: Dict[str, Any]
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        node_order = meta["node_order"]
        left_team = meta["left_team"]
        right_team = meta["right_team"]

        T, N, F = input.shape
        target_size = self.team_size * 2 + len(self.outside_nodes)

        if N >= target_size:
            player_mask = np.ones((T, N), dtype=np.float32)
            return input, poss, player_mask

        input_padded = np.zeros((T, target_size, F), dtype=np.float32)
        player_mask = np.zeros((T, target_size), dtype=np.float32)

        left_mask = np.array([p.startswith(left_team) for p in node_order], dtype=bool)
        right_mask = np.array([p.startswith(right_team) for p in node_order], dtype=bool)
        outside_mask = ~(left_mask | right_mask)

        left_idx = np.where(left_mask)[0]
        right_idx = np.where(right_mask)[0]
        outside_idx = np.where(outside_mask)[0]

        left_count = len(left_idx)
        right_count = len(right_idx)
        outside_count = len(outside_idx)

        left_keep = min(left_count, self.team_size)
        right_keep = min(right_count, self.team_size)
        outside_keep = min(outside_count, len(self.outside_nodes))

        if left_keep > 0:
            input_padded[:, :left_keep, :] = input[:, left_idx[:left_keep], :]
            player_mask[:, :left_keep] = 1.0

        if right_keep > 0:
            start = self.team_size
            input_padded[:, start : start + right_keep, :] = input[:, right_idx[:right_keep], :]
            player_mask[:, start : start + right_keep] = 1.0

        if outside_keep > 0:
            start = self.team_size * 2
            input_padded[:, start : start + outside_keep, :] = input[:, outside_idx[:outside_keep], :]
            player_mask[:, start : start + outside_keep] = 1.0

        def _remap_idx(idx: int) -> int:
            if idx < left_count:
                return idx
            elif idx < left_count + right_count:
                return idx - left_count + self.team_size
            else:
                return idx - (left_count + right_count) + self.team_size * 2

        poss_padded = poss
        if poss is not None:
            poss_padded = poss.copy()
            poss_padded[:, :2] = np.vectorize(_remap_idx)(poss_padded[:, :2])

        return input_padded, poss_padded, player_mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        input_np = s["input"].copy()  # (T, N, F), padded in __init__
        poss_np = s["poss"].copy()  # (T, 3)
        ball_np = s["ball"].copy()  # (T, 2)
        player_mask_np = s["player_mask"].copy()  # (T, N)

        input_np, poss_np, ball_np, player_mask_np = self._random_flip(input_np, poss_np, ball_np, player_mask_np)

        input_pt = torch.tensor(input_np, dtype=torch.float32)  # (T, N, F)
        poss_pt = torch.tensor(poss_np, dtype=torch.long)  # (T, 3)
        ball_pt = torch.tensor(ball_np, dtype=torch.float32)  # (T, 2)
        player_mask_pt = torch.tensor(player_mask_np, dtype=torch.float32)  # (T, N)

        return input_pt, poss_pt, ball_pt, player_mask_pt

    def _random_flip(
        self,
        input_np: np.ndarray,
        poss_np: Optional[np.ndarray],
        ball_np: np.ndarray,
        player_mask_np: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
        """
        Flip coordinates and keep left team in front by swapping team blocks and poss labels when flip_x is True.
        """
        if not self.flip_pitch:
            return input_np, poss_np, ball_np, player_mask_np

        flip_x = bool(np.random.randint(0, 2))
        flip_y = bool(np.random.randint(0, 2))

        if flip_x:
            ball_np[:, 0] = config.PITCH_X - ball_np[:, 0]

            if self._idx_x is not None:
                input_np[:, :, self._idx_x] = config.PITCH_X - input_np[:, :, self._idx_x]
            if self._idx_vx is not None:
                input_np[:, :, self._idx_vx] = -input_np[:, :, self._idx_vx]

            left_block = input_np[:, : self.team_size, :]
            right_block = input_np[:, self.team_size : self.team_size * 2, :]
            outside_block = input_np[:, self.team_size * 2 :, :]
            input_np = np.concatenate([right_block, left_block, outside_block], axis=1)

            left_mask = player_mask_np[:, : self.team_size]
            right_mask = player_mask_np[:, self.team_size : self.team_size * 2]
            outside_mask = player_mask_np[:, self.team_size * 2 :]
            player_mask_np = np.concatenate([right_mask, left_mask, outside_mask], axis=1)

            if poss_np is not None:
                # Flip poss_team
                poss_np[:, 2] = 1 - poss_np[:, 2]

                # Flip poss_prev and poss_next
                poss_player = poss_np[:, :2]
                left_team_mask = poss_player < self.team_size
                right_team_mask = (poss_player >= self.team_size) & (poss_player < self.team_size * 2)
                left_permuted = np.where(left_team_mask, poss_player + self.team_size, 0)
                right_permuted = np.where(right_team_mask, poss_player - self.team_size, 0)

                out_left = self.team_size * 2
                out_right = self.team_size * 2 + 1
                out_bottom = self.team_size * 2 + 2
                out_top = self.team_size * 2 + 3

                out_l_to_r = np.where(poss_player == out_left, out_right, 0)
                out_r_to_l = np.where(poss_player == out_right, out_left, 0)
                out_bt = np.where(np.isin(poss_player, [out_bottom, out_top]), poss_player, 0)
                poss_np[:, :2] = left_permuted + right_permuted + out_l_to_r + out_r_to_l + out_bt

        if flip_y:
            ball_np[:, 1] = config.PITCH_Y - ball_np[:, 1]

            if self._idx_y is not None:
                input_np[:, :, self._idx_y] = config.PITCH_Y - input_np[:, :, self._idx_y]
            if self._idx_vy is not None:
                input_np[:, :, self._idx_vy] = -input_np[:, :, self._idx_vy]

            if poss_np is not None:
                poss_player = poss_np[:, :2]
                out_bottom = self.team_size * 2 + 2
                out_top = self.team_size * 2 + 3

                out_lr = np.where(poss_player < out_bottom, poss_player, 0)
                out_b_to_t = np.where(poss_player == out_bottom, out_top, 0)
                out_t_to_b = np.where(poss_player == out_top, out_bottom, 0)
                poss_np[:, :2] = out_lr + out_b_to_t + out_t_to_b

        return input_np, poss_np, ball_np, player_mask_np
