import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv, global_mean_pool

from datatools.config import PITCH_X, PITCH_Y


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
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


class GATEncoder(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        out_dim: int,
        heads: int = 4,
        dropout: float = 0.0,
        pooling: bool = False,
    ):
        super().__init__()
        self.edge_dim = edge_dim
        self.gat = GATConv(
            node_dim,
            out_dim,
            heads=heads,
            concat=False,
            dropout=dropout,
            add_self_loops=True,
            edge_dim=edge_dim if edge_dim and edge_dim > 0 else None,
        )
        self.norm = nn.LayerNorm(out_dim)
        self.pooling = pooling

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if edge_attr is not None and self.edge_dim:
            x = self.gat(x, edge_index, edge_attr=edge_attr)
        else:
            x = self.gat(x, edge_index)
        x = self.norm(x)
        x = F.elu(x)
        return global_mean_pool(x, batch) if self.pooling else x


class GATLSTM(nn.Module):
    def __init__(
        self,
        macro_type: Optional[str] = None,  # choices: None, poss_team, poss_prev, poss_next, poss_edge
        micro_type: str = "ball",  # choices: ball, poss_edge
        node_in_dim: int = 8,
        edge_in_dim: int = 0,
        team_size: int = 11,
        gat_heads: int = 4,
        macro_node_dim: int = 16,
        macro_graph_dim: int = 16,
        micro_gnn_dim: int = 128,
        macro_rnn_dim: int = 256,
        micro_rnn_dim: int = 256,
        dropout: float = 0.0,
        seq_model: str = "bi_lstm",  # choices: lstm, bi_lstm, transformer
    ):
        super().__init__()

        self.macro_type = None if macro_type in ["none", None] else macro_type
        self.micro_type = micro_type
        self.seq_model = seq_model

        self.register_buffer("pitch_size", torch.tensor([PITCH_X, PITCH_Y], dtype=torch.float32))

        self.team_size = team_size  # Maximum number of nodes in a team
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.macro_node_dim = macro_node_dim
        self.macro_graph_dim = macro_graph_dim
        self.micro_gnn_dim = micro_gnn_dim
        self.macro_rnn_dim = macro_rnn_dim
        self.micro_rnn_dim = micro_rnn_dim
        self.micro_out_dim = 2
        self.macro_out_dim = 1 if self.macro_type is not None else 0

        self.macro_node_gnn = GATEncoder(
            node_in_dim,
            edge_in_dim,
            macro_node_dim,
            heads=gat_heads,
            dropout=dropout,
            pooling=False,
        )
        if macro_graph_dim > 0:
            self.macro_graph_gnn = GATEncoder(
                node_in_dim,
                edge_in_dim,
                macro_graph_dim,
                heads=gat_heads,
                dropout=dropout,
                pooling=True,
            )

        macro_emb_dim = node_in_dim + macro_node_dim + macro_graph_dim
        if seq_model == "transformer":
            self.macro_trans_fc = nn.Sequential(nn.Linear(macro_emb_dim, macro_rnn_dim * 2), nn.ReLU())
            self.macro_pos_encoder = PositionalEncoding(macro_rnn_dim * 2, dropout)
            encoder_layers = TransformerEncoderLayer(macro_rnn_dim * 2, 4, macro_rnn_dim * 4, dropout)
            self.macro_transformer = TransformerEncoder(encoder_layers, 2)
        else:
            self.macro_rnn = nn.LSTM(
                input_size=macro_emb_dim,
                hidden_size=macro_rnn_dim,
                num_layers=2,
                dropout=dropout,
                bidirectional=seq_model == "bi_lstm",
            )

        if self.macro_type is not None:
            macro_out_fc_dim = macro_rnn_dim * 2 if seq_model in ["bi_lstm", "transformer"] else macro_rnn_dim
            self.macro_out_fc = nn.Sequential(nn.Linear(macro_out_fc_dim, 2), nn.GLU())

        micro_in_dim = node_in_dim + macro_rnn_dim * 2 + self.macro_out_dim
        self.micro_node_gnn = GATEncoder(
            micro_in_dim,
            edge_in_dim,
            micro_gnn_dim,
            heads=gat_heads,
            dropout=dropout,
            pooling=False,
        )
        self.micro_graph_gnn = GATEncoder(
            micro_in_dim,
            edge_in_dim,
            micro_gnn_dim,
            heads=gat_heads,
            dropout=dropout,
            pooling=True,
        )

        if micro_type == "poss_edge":
            self.micro_fc1 = nn.Sequential(nn.Linear(node_in_dim + micro_gnn_dim * 2, micro_gnn_dim), nn.ReLU())
            micro_z_dim = micro_rnn_dim * (4 if seq_model in ["bi_lstm", "transformer"] else 2)
            self.micro_fc2 = nn.Sequential(
                nn.Linear(micro_z_dim, micro_rnn_dim), nn.ReLU(), nn.Linear(micro_rnn_dim, 2), nn.GLU()
            )
        else:
            self.micro_fc1 = nn.Sequential(nn.Linear(micro_gnn_dim, micro_gnn_dim), nn.ReLU())
            micro_z_dim = micro_rnn_dim * (2 if seq_model in ["bi_lstm", "transformer"] else 1)
            self.micro_fc2 = nn.Sequential(nn.Linear(micro_z_dim, self.micro_out_dim * 2), nn.GLU())

        if seq_model == "transformer":
            self.micro_trans_fc = nn.Sequential(nn.Linear(micro_gnn_dim, micro_rnn_dim * 2), nn.ReLU())
            self.micro_pos_encoder = PositionalEncoding(micro_rnn_dim * 2, dropout)
            encoder_layers = TransformerEncoderLayer(micro_rnn_dim * 2, 4, micro_rnn_dim * 4, dropout)
            self.micro_transformer = TransformerEncoder(encoder_layers, 2)
        else:
            self.micro_rnn = nn.LSTM(
                input_size=micro_gnn_dim,
                hidden_size=micro_rnn_dim,
                num_layers=2,
                dropout=dropout,
                bidirectional=seq_model == "bi_lstm",
            )

    def forward(self, batch_seq: List[Batch]) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        if self.seq_model in ["lstm", "bi_lstm"]:  # for DataParallel
            self.macro_rnn.flatten_parameters()
            self.micro_rnn.flatten_parameters()

        seq_len = len(batch_seq)  # T
        if seq_len == 0:
            raise ValueError("batch_seq must contain at least one Batch.")

        total_nodes = batch_seq[0].x.size(0)  # N
        max_nodes = self.team_size * 2 + 4

        macro_z_list = []
        for batch in batch_seq:
            edge_attr = batch.edge_attr if self.edge_in_dim > 0 else None
            macro_node_z = self.macro_node_gnn(batch.x, batch.edge_index, batch.batch, edge_attr=edge_attr)  # (N, Z)

            if self.macro_graph_dim > 0:
                macro_graph_z = self.macro_graph_gnn(batch.x, batch.edge_index, batch.batch, edge_attr=edge_attr)
                macro_graph_z_tile = macro_graph_z[batch.batch]  # (B, Z) to (N, Z)
                macro_z = torch.cat([batch.x, macro_node_z, macro_graph_z_tile], dim=-1)  # (N, F+2Z)
            else:
                macro_z = torch.cat([batch.x, macro_node_z], dim=-1)  # (N, F+Z)

            macro_z_list.append(macro_z)

        macro_z = torch.stack(macro_z_list, dim=0)  # (T, N, F+2Z)

        if self.seq_model == "transformer":
            macro_z = self.macro_trans_fc(macro_z)  # (T, N, 2H)
            macro_z = self.macro_pos_encoder(macro_z * math.sqrt(self.macro_rnn_dim))
            macro_h = self.macro_transformer(macro_z)  # (T, N, 2H)
        else:
            macro_h, _ = self.macro_rnn(macro_z)  # (T, N, 2H)

        if self.macro_type is not None:
            macro_out_flat = self.macro_out_fc(macro_h.reshape(seq_len * total_nodes, -1))  # (T*N, 1)
            macro_out = macro_out_flat.view(seq_len, total_nodes, 1)  # (T, N, 1)
        else:
            macro_out = None

        micro_z_list = []
        if self.micro_type == "poss_edge":
            edge_logits_list = []

            for t, batch in enumerate(batch_seq):
                if macro_out is None:
                    micro_in = torch.cat([batch.x, macro_h[t]], dim=-1)  # (N, F+2H)
                else:
                    micro_in = torch.cat([batch.x, macro_h[t], macro_out[t]], dim=-1)  # (N, F+2H+1)

                edge_attr = batch.edge_attr if self.edge_in_dim > 0 else None
                micro_node_z = self.micro_node_gnn(micro_in, batch.edge_index, batch.batch, edge_attr=edge_attr)
                micro_z_t = self.micro_graph_gnn(micro_in, batch.edge_index, batch.batch, edge_attr=edge_attr)
                micro_graph_z_tile = micro_z_t[batch.batch]  # (B, Z) to (N, Z)

                micro_z_t = torch.cat([batch.x, micro_node_z, micro_graph_z_tile], dim=-1)  # (N, F+2Z)
                micro_z_t = self.micro_fc1(micro_z_t)  # (N, Z)
                micro_z_list.append(micro_z_t)

            micro_z = torch.stack(micro_z_list, dim=0)  # (T, N, Z)
            if self.seq_model == "transformer":
                micro_z = self.micro_trans_fc(micro_z)  # (T, N, 2H)
                micro_z = self.micro_pos_encoder(micro_z * math.sqrt(self.micro_rnn_dim) * 2)
                micro_h = self.micro_transformer(micro_z)  # (T, N, 2H)
            else:
                micro_h, _ = self.micro_rnn(micro_z)  # (T, N, 2H)

            for t, batch in enumerate(batch_seq):
                ptr = batch.ptr
                edge_logits_t = []

                for b in range(batch.num_graphs):
                    start, end = ptr[b].item(), ptr[b + 1].item()
                    h_g = micro_h[t][start:end]  # (N_g, 2H)

                    n_g = h_g.size(0)
                    assert n_g <= max_nodes
                    src_h = h_g.unsqueeze(1).expand(-1, n_g, -1)  # (N_g, N_g, 2H)
                    dst_h = h_g.unsqueeze(0).expand(n_g, -1, -1)  # (N_g, N_g, 2H)
                    edge_h = torch.cat([src_h, dst_h], dim=-1)  # (N_g, N_g, 4H)
                    logits_g = self.micro_fc2(edge_h).squeeze(-1)  # (N_g, N_g)

                    padded_logits_g = logits_g.new_zeros((max_nodes, max_nodes))  # (26, 26)
                    padded_logits_g[:n_g, :n_g] = logits_g
                    edge_logits_t.append(padded_logits_g.reshape(-1))  # (26*26,)

                edge_logits_list.append(torch.stack(edge_logits_t, dim=0))  # (B, 26*26)

            macro_out = macro_out.squeeze(-1) if macro_out is not None else None  # (T, N, 1) to (T, N)
            micro_out = torch.stack(edge_logits_list, dim=0)  # (T, B, 26*26)
            return macro_out, micro_out

        else:
            assert self.micro_type.startswith("ball")

            for t, batch in enumerate(batch_seq):
                if macro_out is None:
                    micro_in = torch.cat([batch.x, macro_h[t]], dim=-1)  # (N, F+2H)
                else:
                    micro_in = torch.cat([batch.x, macro_h[t], macro_out[t]], dim=-1)  # (N, F+2H+1)

                edge_attr = batch.edge_attr if self.edge_in_dim > 0 else None
                micro_z_t = self.micro_graph_gnn(micro_in, batch.edge_index, batch.batch, edge_attr=edge_attr)
                micro_z_t = self.micro_fc1(micro_z_t)  # (B, Z)
                micro_z_list.append(micro_z_t)

            micro_z = torch.stack(micro_z_list, dim=0)  # (T, B, Z)
            if self.seq_model == "transformer":
                micro_z = self.micro_trans_fc(micro_z)  # (T, B, 2H)
                micro_z = self.micro_pos_encoder(micro_z * math.sqrt(self.micro_rnn_dim) * 2)
                micro_h = self.micro_transformer(micro_z)  # (T, B, 2H)
            else:
                micro_h, _ = self.micro_rnn(micro_z)  # (T, B, 2H)

            macro_out = macro_out.squeeze(-1) if macro_out is not None else None  # (T, N, 1) to (T, N)
            micro_out = self.micro_fc2(micro_h) * self.pitch_size  # (T, B, 2)
            return macro_out, micro_out
