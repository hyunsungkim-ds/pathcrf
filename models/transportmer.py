import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from datatools.config import PITCH_X, PITCH_Y
from set_transformer.blocks import RFF, SAB

from .dynamic_dense_crf import DynamicDenseCRF
from .dynamic_sparse_crf import DynamicSparseCRF
from .static_dense_crf import StaticDenseCRF
from .static_sparse_crf import StaticSparseCRF
from .utils import build_edge_compression_maps


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:, 0:-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TranSPORTmer(nn.Module):
    def __init__(
        self,
        macro_type: Optional[str] = None,  # choices: None, poss_team, poss_prev, poss_next, poss_edge
        micro_type: str = "ball",  # choices: ball, poss_edge
        crf_model_type: str = None,  # choices: None, {static|dynamic}_{dense|sparse}_crf
        feat_dim: int = 8,
        team_size: int = 11,
        sab_heads: int = 4,
        coarse_dim: int = 128,
        fine_dim: int = 128,
        crf_edge_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.macro_type = None if macro_type in ["none", None] else macro_type
        self.micro_type = micro_type
        self.crf_model_type = crf_model_type

        self.team_size = team_size  # number of players per team (usually 11)
        self.n_nodes = team_size * 2 + 4  # number of total players + 4 outside labels (usually 26)
        self.macro_out_dim = 0 if self.macro_type is None else (2 if self.macro_type == "poss_edge" else 1)

        self.register_buffer("pitch_size", torch.tensor([PITCH_X, PITCH_Y], dtype=torch.float32))

        self.coarse_dim = coarse_dim
        self.fine_dim = fine_dim
        self.sab_heads = sab_heads
        self.dropout = dropout

        # Coarse encoder
        self.coarse_proj = nn.Sequential(nn.Linear(feat_dim, coarse_dim), nn.ReLU(), nn.Dropout(dropout))
        self.coarse_pe = PositionalEncoding(coarse_dim)
        self.coarse_sab_t1 = SAB(coarse_dim, self.sab_heads, RFF(coarse_dim))
        self.coarse_sab_t2 = SAB(coarse_dim, self.sab_heads, RFF(coarse_dim))
        self.coarse_sab_s = SAB(coarse_dim, self.sab_heads, RFF(coarse_dim))

        # Intermediate possessor prediction head
        if self.macro_type is not None:
            self.macro_head = nn.Sequential(
                nn.Linear(coarse_dim, coarse_dim),
                nn.ReLU(),
                nn.Linear(coarse_dim, 2 * self.macro_out_dim),
                nn.GLU(),
            )
            self.fine_proj = nn.Linear(feat_dim + coarse_dim + self.macro_out_dim, fine_dim)
        else:
            self.fine_proj = nn.Linear(feat_dim + coarse_dim, fine_dim)

        # Fine encoder
        self.fine_pe = PositionalEncoding(fine_dim)
        self.fine_sab_t1 = SAB(fine_dim, self.sab_heads, RFF(fine_dim))
        self.fine_sab_t2 = SAB(fine_dim, self.sab_heads, RFF(fine_dim))
        self.fine_sab_s = SAB(fine_dim, self.sab_heads, RFF(fine_dim))

        # Ball trajectory prediction head
        if self.micro_type == "ball":
            self.ball_head = nn.Sequential(
                nn.Linear(fine_dim, fine_dim),
                nn.ReLU(),
                nn.Linear(fine_dim, 4),
                nn.GLU(),
            )

        # Possession edge prediction head
        elif self.micro_type == "poss_edge":
            self.edge_proj = nn.Sequential(nn.Linear(fine_dim * 2, fine_dim), nn.ReLU())
            self.edge_head = nn.Sequential(nn.Linear(fine_dim, 2), nn.GLU())

            valid_orig_edge_ids, orig2comp, comp_src, comp_dst = build_edge_compression_maps(self.team_size * 2, 4)
            self.register_buffer("valid_orig_edge_ids", valid_orig_edge_ids)  # (576,)
            self.register_buffer("orig2comp", orig2comp)  # (676,)
            self.register_buffer("comp_src", comp_src)  # (576,)
            self.register_buffer("comp_dst", comp_dst)  # (576,)

            if crf_model_type == "static_dense_crf":
                self.crf = StaticDenseCRF(comp_src, comp_dst, self.team_size)
            elif crf_model_type == "static_sparse_crf":
                self.crf = StaticSparseCRF(comp_src, comp_dst, self.team_size)
            elif crf_model_type == "dynamic_dense_crf":
                self.crf_edge_proj = nn.Sequential(nn.Linear(fine_dim, crf_edge_dim), nn.ReLU())
                self.crf_edge_norm = nn.LayerNorm(crf_edge_dim)
                self.crf = DynamicDenseCRF(comp_src, comp_dst, crf_edge_dim, team_size=self.team_size)
            elif crf_model_type == "dynamic_sparse_crf":
                self.crf_edge_proj = nn.Sequential(nn.Linear(fine_dim, crf_edge_dim), nn.ReLU())
                self.crf_edge_norm = nn.LayerNorm(crf_edge_dim)
                self.crf = DynamicSparseCRF(comp_src, comp_dst, crf_edge_dim, team_size=self.team_size)
            else:
                self.crf = None

    def _temporal_sab(self, x: torch.Tensor, block: SAB, pos_encoder: PositionalEncoding = None) -> torch.Tensor:
        """
        Apply SAB along the temporal dimension.
        Input x: (B, T, N, D)
        Output x: (B, T, N, D)
        """
        B, T, N, D = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
        if pos_encoder is not None:
            x = x.transpose(0, 1)  # (T, B*N, D)
            x = pos_encoder(x)
            x = x.transpose(0, 1)  # (B*N, T, D)
        x = block(x)
        x = x.reshape(B, N, T, D).permute(0, 2, 1, 3)  # (B, T, N, D)
        return x

    def _social_sab(self, x: torch.Tensor, block: SAB) -> torch.Tensor:
        """
        Apply SAB along the social (agent) dimension.
        Input x: (B, T, N, D)
        Output x: (B, T, N, D)
        """
        B, T, N, D = x.shape
        x = x.reshape(B * T, N, D)
        x = block(x)
        x = x.reshape(B, T, N, D)
        return x

    def forward(
        self,
        input: torch.Tensor,
        macro_target: torch.Tensor = None,
        micro_target: torch.Tensor = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        B, T, N, _ = input.shape
        x = self.coarse_proj(input)  # (B, T, N, coarse)

        if self.micro_type == "ball":
            ball_token = torch.ones((B, T, 1, self.coarse_dim)).to(input.device)
            x = torch.cat([x, ball_token], dim=2)  # (B, T, N+1, coarse)

        # Coarse Encoder
        x = self._temporal_sab(x, self.coarse_sab_t1, self.coarse_pe)
        x = self._temporal_sab(x, self.coarse_sab_t2, pos_encoder=None)
        x = self._social_sab(x, self.coarse_sab_s)  # (B, T, N, coarse)

        macro_out = None
        if self.macro_type is None:
            x = torch.cat([input, x], -1)  # (B, T, N, F+coarse)
        else:
            macro_out = self.macro_head(x)  # (B, T, N, M)
            x = torch.cat([input, x, macro_out], -1)  # (B, T, N, F+coarse+M)

        # Fine Encoder
        x = self.fine_proj(x)  # (B, T, N, fine)
        x = self._temporal_sab(x, self.fine_sab_t1, self.fine_pe)
        x = self._temporal_sab(x, self.fine_sab_t2, pos_encoder=None)
        x = self._social_sab(x, self.fine_sab_s)  # (B, T, N, fine)

        # Output Layer
        if self.micro_type == "ball":
            micro_out = self.ball_head(x[:, :, -1, :])  # (B, T, 2)
            micro_out = micro_out * self.pitch_size
            macro_out_return = None
            if self.macro_type is not None:
                macro_out_return = macro_out.squeeze(-1) if self.macro_out_dim == 1 else macro_out
            return macro_out_return, micro_out, None

        elif self.micro_type == "poss_edge":
            node_h = x.reshape(B * T, N, -1)  # (B*T, N, fine)
            src_h = node_h.unsqueeze(2).expand(-1, N, N, -1)  # (B*T, N, N, fine)
            dst_h = node_h.unsqueeze(1).expand(-1, N, N, -1)  # (B*T, N, N, fine)
            edge_h = self.edge_proj(torch.cat([src_h, dst_h], -1))  # (B*T, N, N, fine)
            edge_logits = self.edge_head(edge_h).squeeze(-1)  # (B*T, N, N)
            edge_logits = edge_logits.reshape(B, T, -1)  # (B, T, N*N)

            if isinstance(self.crf, (DynamicDenseCRF, DynamicSparseCRF)):
                edge_h = edge_h.reshape(B, T, N * N, -1)  # (B, T, N*N, fine)
                edge_h = self.crf_edge_proj(edge_h)  # (B, T, N*N, edge)
                edge_h = self.crf_edge_norm(edge_h)  # (B, T, N*N, edge)
            else:
                edge_h = None

            if self.macro_type is None:
                return None, edge_logits, edge_h
            else:
                macro_out_return = macro_out.squeeze(-1) if self.macro_out_dim == 1 else macro_out
                return macro_out_return, edge_logits, edge_h

        else:
            raise ValueError(f"Unknown micro_type: {self.micro_type}")
