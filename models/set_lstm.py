import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from datatools.config import PITCH_X, PITCH_Y
from set_transformer.blocks import RFF, SAB
from set_transformer.model import SetTransformer

from .dynamic_dense_crf import DynamicDenseCRF
from .dynamic_sparse_crf import DynamicSparseCRF
from .static_dense_crf import StaticDenseCRF
from .static_sparse_crf import StaticSparseCRF
from .utils import build_edge_compression_maps


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
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

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class SetLSTM(nn.Module):
    def __init__(
        self,
        macro_type: Optional[str] = None,  # choices: None, poss_team, poss_prev, poss_next, poss_edge
        micro_type: str = "ball",  # choices: ball, poss_edge
        seq_model_type: str = "bi_lstm",  # choices: lstm, bi_lstm, sab
        crf_model_type: str = None,  # choices: None, {static|dynamic}_{dense|sparse}_crf
        feat_dim: int = 8,
        team_size: int = 11,
        macro_ppe_dim: int = 16,
        macro_fpe_dim: int = 16,
        micro_enc_dim: int = 64,
        macro_seq_dim: int = 128,
        micro_seq_dim: int = 128,
        crf_edge_dim: int = 16,
        dropout: float = 0,
    ):
        super().__init__()

        self.macro_type = None if macro_type in ["none", None] else macro_type
        self.micro_type = micro_type
        self.seq_model_type = seq_model_type
        self.crf_model_type = crf_model_type

        self.feat_dim = feat_dim
        self.team_size = team_size  # number of players per team (usually 11)
        self.n_nodes = team_size * 2 + 4  # number of total players + 4 outside labels (usually 26)

        self.register_buffer("pitch_size", torch.tensor([PITCH_X, PITCH_Y], dtype=torch.float32))

        self.macro_ppe_dim = macro_ppe_dim
        self.macro_fpe_dim = macro_fpe_dim
        self.macro_seq_dim = macro_seq_dim

        self.micro_seq_dim = micro_seq_dim
        self.micro_out_dim = 2
        self.macro_out_dim = 0 if self.macro_type is None else (2 if self.macro_type == "poss_edge" else 1)
        self.macro_mask_dim = 2 if self.macro_type == "poss_edge" else 1

        n_layers = 2

        assert macro_ppe_dim > 0 or macro_fpe_dim > 0
        macro_z_dim = feat_dim + self.macro_mask_dim

        if macro_ppe_dim > 0:
            self.macro_team1_enc = SetTransformer(feat_dim + self.macro_mask_dim, macro_ppe_dim, embed_type="e")
            self.macro_team2_enc = SetTransformer(feat_dim + self.macro_mask_dim, macro_ppe_dim, embed_type="e")
            self.macro_outside_enc = nn.Linear(feat_dim + self.macro_mask_dim, macro_ppe_dim)
            macro_z_dim += macro_ppe_dim

        if macro_fpe_dim > 0:
            self.macro_fpe_enc = SetTransformer(feat_dim + self.macro_mask_dim, macro_fpe_dim, embed_type="e")
            macro_z_dim += macro_fpe_dim

        if seq_model_type == "sab":
            self.macro_seq_proj = nn.Sequential(nn.Linear(macro_z_dim, macro_seq_dim), nn.ReLU())
            self.macro_pe = PositionalEncoding(macro_seq_dim, dropout)
            self.macro_sab_t1 = SAB(macro_seq_dim, 4, RFF(macro_seq_dim))
            self.macro_sab_t2 = SAB(macro_seq_dim, 4, RFF(macro_seq_dim))
        else:
            self.macro_seq = nn.LSTM(
                input_size=macro_z_dim,
                hidden_size=macro_seq_dim,
                num_layers=n_layers,
                dropout=dropout,
                bidirectional=seq_model_type == "bi_lstm",
            )

        macro_seq_out_dim = macro_seq_dim * 2 if seq_model_type == "bi_lstm" else macro_seq_dim
        if self.macro_type is not None:
            self.macro_head = nn.Sequential(nn.Linear(macro_seq_out_dim, 2 * self.macro_out_dim), nn.GLU())

        micro_in_dim = feat_dim + macro_seq_out_dim + self.macro_out_dim
        micro_st_type = "e" if micro_type == "poss_edge" else "i"
        self.micro_team1_enc = SetTransformer(micro_in_dim, micro_enc_dim, embed_type=micro_st_type)
        self.micro_team2_enc = SetTransformer(micro_in_dim, micro_enc_dim, embed_type=micro_st_type)

        if micro_type == "ball":
            self.micro_outside_enc = nn.Sequential(nn.Linear(micro_in_dim * 4, micro_enc_dim), nn.ReLU())
            self.micro_pool = nn.Sequential(nn.Linear(micro_enc_dim * 3, micro_enc_dim), nn.ReLU())
        elif micro_type == "poss_edge":
            self.micro_outside_enc = nn.Sequential(nn.Linear(micro_in_dim, micro_enc_dim), nn.ReLU())
            self.micro_fpe_enc = SetTransformer(micro_in_dim, micro_enc_dim, embed_type="e")
            self.micro_node_proj = nn.Sequential(nn.Linear(micro_enc_dim * 2, micro_enc_dim), nn.ReLU())
        else:
            raise ValueError(f"Unknown micro_type: {self.micro_type}")

        if seq_model_type == "sab":
            self.micro_seq_proj = nn.Sequential(nn.Linear(micro_enc_dim + self.micro_out_dim, micro_seq_dim), nn.ReLU())
            self.micro_pe = PositionalEncoding(micro_seq_dim, dropout)
            self.micro_sab_t1 = SAB(micro_seq_dim, 4, RFF(micro_seq_dim))
            self.micro_sab_t2 = SAB(micro_seq_dim, 4, RFF(micro_seq_dim))
        else:
            self.micro_seq = nn.LSTM(
                input_size=micro_enc_dim + self.micro_out_dim,
                hidden_size=micro_seq_dim,
                num_layers=n_layers,
                dropout=dropout,
                bidirectional=seq_model_type == "bi_lstm",
            )

        micro_seq_out_dim = micro_seq_dim * 2 if seq_model_type == "bi_lstm" else micro_seq_dim
        if micro_type == "ball":
            self.micro_ball_head = nn.Sequential(nn.Linear(micro_seq_out_dim, self.micro_out_dim * 2), nn.GLU())
            self.crf = None

        elif micro_type == "poss_edge":
            self.micro_edge_proj = nn.Sequential(nn.Linear(micro_seq_out_dim * 2, micro_seq_dim), nn.ReLU())
            self.micro_edge_head = nn.Sequential(nn.Linear(micro_seq_dim, 2), nn.GLU())

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
                self.crf_edge_proj = nn.Sequential(nn.Linear(micro_seq_dim, crf_edge_dim), nn.ReLU())
                self.crf_edge_norm = nn.LayerNorm(crf_edge_dim)
                self.crf = DynamicDenseCRF(comp_src, comp_dst, crf_edge_dim, team_size=self.team_size)
            elif crf_model_type == "dynamic_sparse_crf":
                self.crf_edge_proj = nn.Sequential(nn.Linear(micro_seq_dim, crf_edge_dim), nn.ReLU())
                self.crf_edge_norm = nn.LayerNorm(crf_edge_dim)
                self.crf = DynamicSparseCRF(comp_src, comp_dst, crf_edge_dim, team_size=self.team_size)
            else:
                self.crf = None

    def _temporal_sab(
        self, x: torch.Tensor, block: SAB, pos_encoder: Optional[PositionalEncoding] = None
    ) -> torch.Tensor:
        """
        Apply SAB along the temporal dimension.
        Input x: (T, B, D)
        Output x: (T, B, D)
        """
        if pos_encoder is not None:
            x = pos_encoder(x)
        x = x.transpose(0, 1)  # (B, T, D)
        x = block(x)
        x = x.transpose(0, 1)  # (T, B, D)
        return x

    def forward(
        self,
        input: torch.Tensor,
        macro_target: torch.Tensor = None,
        micro_target: torch.Tensor = None,
        random_mask: torch.Tensor = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        if self.seq_model_type in ["lstm", "bi_lstm"]:  # for DataParallel
            self.macro_seq.flatten_parameters()
            self.micro_seq.flatten_parameters()

        input = input.transpose(0, 1)  # (B, T, N, F) to (T, B, N, F)
        T, B, N, feat_dim = input.shape

        team_size = self.team_size
        assert feat_dim == self.feat_dim

        random_mask = random_mask.transpose(0, 1) if random_mask is not None else None  # (B, T) to (T, B)

        if macro_target is not None and random_mask is not None:
            if self.macro_type == "poss_edge":
                src_onehot = F.one_hot(macro_target[:, :, 0], N).transpose(0, 1)  # (B, T, N) to (T, B, N)
                dst_onehot = F.one_hot(macro_target[:, :, 1], N).transpose(0, 1)  # (B, T, N) to (T, B, N)
                macro_onehot = torch.stack([src_onehot, dst_onehot], dim=-1).float()  # (T, B, N, 2)
                masked_macro_target = macro_onehot * random_mask.unsqueeze(-1)
                masked_macro_target = masked_macro_target.reshape(T * B, N, self.macro_mask_dim)
            else:
                macro_onehot = F.one_hot(macro_target, N).transpose(0, 1).float()  # (B, T, N) to (T, B, N)
                masked_macro_target = (macro_onehot * random_mask).reshape(T * B, N, 1)
        else:
            masked_macro_target = torch.zeros(T * B, N, self.macro_mask_dim).to(input.device)

        if self.micro_type == "ball":
            if micro_target is not None and random_mask is not None:
                masked_micro_target = micro_target.transpose(0, 1) * random_mask  # (T, B, 2)
            else:
                masked_micro_target = torch.zeros(T, B, self.micro_out_dim).to(input.device)
        elif self.micro_type == "poss_edge":
            if micro_target is not None and random_mask is not None:
                src_onehot = F.one_hot(micro_target[:, :, 0], N).transpose(0, 1)  # (B, T, N) to (T, B, N)
                dst_onehot = F.one_hot(micro_target[:, :, 1], N).transpose(0, 1)  # (B, T, N) to (T, B, N)
                micro_onehot = torch.stack([src_onehot, dst_onehot], dim=-1).float()  # (T, B, N, 2)
                masked_micro_target = micro_onehot * random_mask.unsqueeze(-1)
            else:
                masked_micro_target = torch.zeros(T, B, N, self.micro_out_dim).to(input.device)
            masked_micro_target = masked_micro_target.reshape(T, B * N, -1)  # (T, B*N, 2)

        team1_x = input[:, :, :team_size, :].reshape(T * B, team_size, feat_dim)  # (T*B, N_team, F)
        team1_x_expanded = torch.cat([team1_x, masked_macro_target[:, :team_size]], -1)  # (T*B, N_team, F+1)

        team2_x = input[:, :, team_size : team_size * 2, :].reshape(T * B, team_size, feat_dim)
        team2_x_expanded = torch.cat([team2_x, masked_macro_target[:, team_size:-4]], -1)  # (T*B, N_team, F+1)

        outside_x = input[:, :, -4:, :].reshape(T * B, 4, feat_dim)  # (T*B, 4, F)
        outside_x_expanded = torch.cat([outside_x, masked_macro_target[:, -4:]], -1)  # (T*B, 4, F+1)

        x = torch.cat([team1_x_expanded, team2_x_expanded, outside_x_expanded], 1)  # (T*B, N, F+1)

        if self.macro_ppe_dim > 0 or self.macro_fpe_dim > 0:
            macro_z = x

            if self.macro_ppe_dim > 0:
                team1_z = self.macro_team1_enc(team1_x_expanded)  # (T*B, N_team, PPE)
                team2_z = self.macro_team2_enc(team2_x_expanded)  # (T*B, N_team, PPE)
                outsize_z = self.macro_outside_enc(outside_x_expanded)  # (T*B, 4, PPE)
                macro_ppe_z = torch.cat([team1_z, team2_z, outsize_z], 1)  # (T*B, N, PPE)
                macro_z = torch.cat([x, macro_ppe_z], -1)  # (T*B, N, F+1+PPE)

            if self.macro_fpe_dim > 0:
                macro_fpe_z = self.macro_fpe_enc(x)  # (T*B, N, FPE)
                macro_z = torch.cat([macro_z, macro_fpe_z], -1)

            macro_z = macro_z.reshape(T, B * N, -1)  # (T, B*N, Z)
            if self.seq_model_type == "sab":
                macro_z = self.macro_seq_proj(macro_z)  # (T, B*N, 2H)
                macro_h = self._temporal_sab(macro_z, self.macro_sab_t1, self.macro_pe)
                macro_h = self._temporal_sab(macro_h, self.macro_sab_t2, None)
            else:
                macro_h, _ = self.macro_seq(macro_z)  # (T, B*N, 2H)

            macro_h = macro_h.reshape(T * B, N, -1)  # (T*B, N, 2H)
            macro_out = self.macro_head(macro_h) if self.macro_type is not None else None  # (T*B, N, M)

        else:  # Use raw inputs (with randomly ordered players) without permutation-equivariant embedding
            macro_z = x.reshape(T, B, -1)  # (T, B, N*(F+1))
            macro_h, _ = self.macro_seq(macro_z)  # (T, B, 2H)
            macro_h = macro_h.reshape(T * B, -1)
            if self.macro_type is not None:
                macro_out = self.macro_head(macro_h).reshape(T * B, N, self.macro_out_dim)  # (T*B, N, M)
            else:
                macro_out = None
            macro_h = macro_h.unsqueeze(1).expand(-1, N, -1)  # (T*B, N, 2H)

        if self.macro_type is None:
            # team_micro_in: (T*B, N_team, F+2H), outside_micro_in: (T*B, 4, F+2H)
            team1_micro_in = torch.cat([team1_x, macro_h[:, :team_size]], -1)
            team2_micro_in = torch.cat([team2_x, macro_h[:, team_size:-4]], -1)
            outside_micro_in = torch.cat([outside_x, macro_h[:, -4:]], -1)
        else:
            # team_micro_in: (T*B, N_team, F+2H+M), outside_micro_in: (T*B, 4, F+2H+M)
            team1_micro_in = torch.cat([team1_x, macro_h[:, :team_size], macro_out[:, :team_size]], -1)
            team2_micro_in = torch.cat([team2_x, macro_h[:, team_size:-4], macro_out[:, team_size:-4]], -1)
            outside_micro_in = torch.cat([outside_x, macro_h[:, -4:], macro_out[:, -4:]], -1)

        if self.micro_type == "ball":
            assert self.micro_type.startswith("ball")

            micro_team1_z = self.micro_team1_enc(team1_micro_in)  # (T*B, Z)
            micro_team2_z = self.micro_team2_enc(team2_micro_in)  # (T*B, Z)
            outside_micro_in = outside_micro_in.reshape(T * B, -1)  # (T*B, 4*(F+2H+1))
            micro_outside_z = self.micro_outside_enc(outside_micro_in)  # (T*B, Z)

            micro_z = torch.cat([micro_team1_z, micro_team2_z, micro_outside_z], -1)  # (T*B, 3Z)
            micro_z = self.micro_pool(micro_z).reshape(T, B, -1)  # (T, B, Z)
            micro_z = torch.cat([micro_z, masked_micro_target], -1)  # (T, B, Z+2)

            if self.seq_model_type == "sab":
                micro_z = self.micro_seq_proj(micro_z)  # (T, B, 2H)
                micro_h = self._temporal_sab(micro_z, self.micro_sab_t1, self.micro_pe)
                micro_h = self._temporal_sab(micro_h, self.micro_sab_t2, None)
            else:
                micro_h, _ = self.micro_seq(micro_z)  # (T, B, 2H)

            micro_out = self.micro_ball_head(micro_h).transpose(0, 1)  # (B, T, 2)
            micro_out = micro_out * self.pitch_size
            macro_out_return = None
            if self.macro_type is not None:
                macro_out_return = macro_out.reshape(T, B, N, self.macro_out_dim).transpose(0, 1)
                if self.macro_out_dim == 1:
                    macro_out_return = macro_out_return.squeeze(-1)  # (B, T, N)
            return macro_out_return, micro_out, None  # (B, T, N or N,2), (B, T, 2), None

        elif self.micro_type == "poss_edge":
            team1_node_z = self.micro_team1_enc(team1_micro_in)  # (T*B, N_team, Z)
            team2_node_z = self.micro_team2_enc(team2_micro_in)  # (T*B, N_team, Z)
            outside_node_z = self.micro_outside_enc(outside_micro_in)  # (T*B, 4, Z)
            micro_ppe_z = torch.cat([team1_node_z, team2_node_z, outside_node_z], 1)  # (T*B, N, Z)

            micro_fpe_in = torch.cat([team1_micro_in, team2_micro_in, outside_micro_in], 1)  # (T*B, N, F+2H+1)
            micro_fpe_z = self.micro_fpe_enc(micro_fpe_in)  # (T*B, N, Z)

            micro_z = torch.cat([micro_ppe_z, micro_fpe_z], -1)  # (T*B, N, 2Z)
            micro_z = self.micro_node_proj(micro_z).reshape(T, B * N, -1)  # (T, B*N, Z)

            if masked_micro_target.dim() == 4:
                masked_micro_target = masked_micro_target.reshape(T, B * N, -1)
            micro_z = torch.cat([micro_z, masked_micro_target], -1)  # (T, B*N, Z+2)
            if self.seq_model_type == "sab":
                micro_z = self.micro_seq_proj(micro_z)  # (T, B*N, 2H)
                micro_h = self._temporal_sab(micro_z, self.micro_sab_t1, self.micro_pe)
                micro_h = self._temporal_sab(micro_h, self.micro_sab_t2, None)
            else:
                micro_h, _ = self.micro_seq(micro_z)  # (T, B*N, 2H)

            micro_h = micro_h.reshape(T * B, N, -1)  # (T*B, N, 2H)
            src_h = micro_h.unsqueeze(2).expand(-1, N, N, -1)  # (T*B, N, N, 2H)
            dst_h = micro_h.unsqueeze(1).expand(-1, N, N, -1)  # (T*B, N, N, 2H)
            edge_h = self.micro_edge_proj(torch.cat([src_h, dst_h], -1))  # (T*B, N, N, H)
            edge_logits = self.micro_edge_head(edge_h).squeeze(-1)  # (T*B, N, N)
            edge_logits = edge_logits.reshape(T, B, -1).transpose(0, 1)  # (B, T, N*N)

            if isinstance(self.crf, (DynamicDenseCRF, DynamicSparseCRF)):
                edge_h = edge_h.reshape(T, B, N * N, -1)  # (T, B, N*N, H)
                edge_h = self.crf_edge_proj(edge_h)  # (T, B, N*N, D)
                edge_h = self.crf_edge_norm(edge_h).transpose(0, 1)  # (B, T, N*N, D)
            else:
                edge_h = None

            macro_out_return = None
            if self.macro_type is not None:
                macro_out_return = macro_out.reshape(T, B, N, self.macro_out_dim).transpose(0, 1)
                if self.macro_out_dim == 1:
                    macro_out_return = macro_out_return.squeeze(-1)  # (B, T, N)
            return macro_out_return, edge_logits, edge_h

        else:
            raise ValueError(f"Unknown micro_type: {self.micro_type}")
