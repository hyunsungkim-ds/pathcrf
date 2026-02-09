import argparse
import json
import os
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import Batch

from dataset import SoccerWindowGraphs, SoccerWindowTensors
from models import utils
from models.dynamic_dense_crf import DynamicDenseCRF
from models.dynamic_sparse_crf import DynamicSparseCRF
from models.set_lstm import SetLSTM
from models.transportmer import TranSPORTmer


def _init_distributed() -> Tuple[bool, int, int, int]:
    """
    Initialize torch.distributed from torchrun env vars.

    Returns: (distributed, rank, world_size, local_rank)
    """
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return False, 0, 1, 0

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available, but WORLD_SIZE > 1 was set.")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return True, rank, world_size, local_rank


def _maybe_all_reduce_mean(value: float, device: torch.device) -> float:
    if not dist.is_available() or not dist.is_initialized():
        return float(value)
    t = torch.tensor([value], dtype=torch.float32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return float(t.item())


def _macro_loss_and_acc(
    macro_out: torch.Tensor, macro_target: torch.Tensor, batch_seq: list[Batch]
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Compute possessor classification loss/accuracy per graph using Batch.ptr to slice nodes.
    - macro_out: (T, N_total) logits
    - macro_target: (T, B) int targets (index within each graph)
    """
    macro_losses = []
    macro_correct = []

    for t in range(macro_out.shape[0]):
        logits_t = macro_out[t]  # (N_total,)
        batch_t = batch_seq[t]
        ptr = batch_t.ptr  # (B+1,)

        for b in range(batch_t.num_graphs):
            start, end = ptr[b].item(), ptr[b + 1].item()
            target_idx = macro_target[t, b]

            logits_slice = logits_t[start:end].unsqueeze(0)  # (1, N_graph)
            target_slice = target_idx.unsqueeze(0)
            macro_losses.append(F.cross_entropy(logits_slice, target_slice))

            pred = logits_slice.argmax(dim=1)
            macro_correct.append((pred == target_slice).float())

    if len(macro_losses) == 0:
        return None, None

    macro_loss = torch.stack(macro_losses).mean()
    macro_acc = torch.stack(macro_correct).mean()
    return macro_loss, macro_acc


def run_epoch(
    model: DistributedDataParallel,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
    macro_weight: float = 0,
    rloss_weight: float = 0,
    crf_weight: float = 0,
    use_emit_loss: bool = False,
    use_src_loss: bool = False,
    use_dst_loss: bool = False,
    clip: float = None,
    train: bool = False,
    is_main: bool = True,
    print_batch: int = 50,
):
    model.train() if train else model.eval()
    module = model.module if hasattr(model, "module") else model

    if module.macro_type in ["poss_prev", "poss_next", "poss_edge"]:
        loss_dict = {"macro_loss": [], "macro_acc": []}
    else:
        loss_dict = dict()

    if module.micro_type == "poss_edge":
        loss_dict = loss_dict | {"micro_loss": [], "crf_loss": [], "edge_acc": [], "src_acc": [], "dst_acc": []}
        if not train:
            loss_dict["forbid_rate"] = []
    elif isinstance(module, SetLSTM) and rloss_weight > 0:
        loss_dict = loss_dict | {"micro_loss": [], "real_loss": [], "pos_error": []}
    else:  # module.micro_type == "ball" and (isinstance(module, GATLSTM) or rloss_weight == 0)
        loss_dict = loss_dict | {"micro_loss": [], "pos_error": []}

    n_batches = len(loader)
    n_nodes = module.team_size * 2 + 4

    allowed_mask_comp = None
    allowed_mask_orig = None

    if (not train) and module.micro_type == "poss_edge":
        if hasattr(module, "comp_src") and hasattr(module, "comp_dst"):
            src_list = module.comp_src.detach().cpu().tolist()
            dst_list = module.comp_dst.detach().cpu().tolist()
            allowed_mask_comp = utils.build_allowed_mask(src_list, dst_list, module.team_size).to(device)

        src_list = [i // n_nodes for i in range(n_nodes * n_nodes)]
        dst_list = [i % n_nodes for i in range(n_nodes * n_nodes)]
        allowed_mask_orig = utils.build_allowed_mask(src_list, dst_list, module.team_size).to(device)

    for batch_idx, batch in enumerate(loader):
        batch_input, batch_poss, batch_ball, _ = batch
        if module.macro_type is not None:
            assert batch_poss.numel() > 0
            if module.macro_type == "poss_prev":
                macro_target = batch_poss[:, :, 0].to(device)  # (B, T)
            elif module.macro_type == "poss_next":
                macro_target = batch_poss[:, :, 1].to(device)  # (B, T)
            elif module.macro_type == "poss_team":
                macro_target = batch_poss[:, :, 2].to(device)  # (B, T)
            else:  # module.macro_type == "poss_edge"
                macro_target = batch_poss[:, :, :2].to(device)  # (B, T, 2)
        else:
            macro_target = None

        if module.micro_type == "poss_edge":
            assert batch_poss.numel() > 0
            micro_target = batch_poss[:, :, :2].to(device)  # (B, T, 2)
        else:  # module.micro_type == "ball"
            micro_target = batch_ball.to(device)  # (B, T, 2)

        if isinstance(batch[0], list):
            batch_input = [b.to(device) for b in batch_input]
            if macro_target is not None:
                if macro_target.dim() == 2:
                    macro_target = macro_target.permute(1, 0)  # (T, B)
                else:  # module.macro_type == "poss_edge"
                    macro_target = macro_target.permute(1, 0, 2)  # (T, B, 2)
            micro_target = micro_target.permute(1, 0, 2)  # (T, B, 2)

            with torch.set_grad_enabled(train):
                model_out = model(batch_input)
                if isinstance(model_out, tuple) and len(model_out) == 3:
                    macro_out, micro_out, _ = model_out
                else:
                    macro_out, micro_out = model_out  # (T, N_total) or None, (T, B, 2) or (T, B, 26*26)
                if module.micro_type == "poss_edge":
                    edge_labels = micro_target[:, :, 0].long() * n_nodes + micro_target[:, :, 1].long()  # (T, B)
                    micro_logits = micro_out.reshape(-1, micro_out.size(-1))  # (T*B, max_nodes*max_nodes)
                    micro_loss = micro_logits.new_tensor(0.0)
                    if use_emit_loss:
                        micro_loss += F.cross_entropy(micro_logits, edge_labels.reshape(-1))

                    micro_pred = micro_logits.argmax(dim=1)
                    micro_acc = (micro_pred == edge_labels.reshape(-1)).float().mean()
                    loss_dict["edge_acc"].append(micro_acc.item())

                    if (not train) and (allowed_mask_comp is not None or allowed_mask_orig is not None):
                        pred_edges = micro_pred.view(edge_labels.shape).transpose(0, 1)  # (B, T)
                        if allowed_mask_comp is not None and hasattr(module, "orig2comp"):
                            pred_edges = module.orig2comp[pred_edges]
                            forbid_rate = utils.forbid_rate(pred_edges, allowed_mask_comp)
                        else:
                            forbid_rate = utils.forbid_rate(pred_edges, allowed_mask_orig)
                        loss_dict["forbid_rate"].append(forbid_rate.item())

                    src_pred = micro_pred // n_nodes
                    src_labels = micro_target[:, :, 0].long().reshape(-1)
                    src_acc = (src_pred == src_labels).float().mean()
                    loss_dict["src_acc"].append(src_acc.item())

                    dst_pred = micro_pred % n_nodes
                    dst_labels = micro_target[:, :, 1].long().reshape(-1)
                    dst_acc = (dst_pred == dst_labels).float().mean()
                    loss_dict["dst_acc"].append(dst_acc.item())

                else:
                    micro_loss = nn.MSELoss()(micro_out, micro_target)

                macro_loss = 0.0
                macro_acc = None
                if macro_out is not None and macro_target is not None and macro_target.dim() == 2:
                    macro_loss, macro_acc = _macro_loss_and_acc(macro_out, macro_target, batch_input)

                loss = macro_weight * macro_loss + micro_loss
        else:
            batch_input = batch_input.to(device)  # (B, T, N, F)
            player_mask = batch[-1].to(device)  # (B, T, N)

            with torch.set_grad_enabled(train):
                macro_out, micro_out, edge_embeds = model(batch_input, macro_target, micro_target)
                # None or (B, T, N) or (B, T, N, 2), (B, T, ...), (B, T, K, D)

                macro_loss = 0.0
                macro_acc = None

                if module.macro_type in ["poss_prev", "poss_next"] and macro_target is not None:
                    macro_logits = macro_out.reshape(-1, macro_out.size(-1))  # (B*T, N)
                    macro_labels = macro_target.reshape(-1)  # (B*T,)
                    macro_loss = F.cross_entropy(macro_logits, macro_labels)
                    macro_acc = (macro_logits.argmax(dim=1) == macro_labels).float().mean()

                elif module.macro_type == "poss_edge" and macro_target is not None:
                    src_logits = macro_out[:, :, :, 0].reshape(-1, n_nodes)  # (B*T, N)
                    dst_logits = macro_out[:, :, :, 1].reshape(-1, n_nodes)  # (B*T, N)
                    src_labels = macro_target[:, :, 0].reshape(-1)  # (B*T,)
                    dst_labels = macro_target[:, :, 1].reshape(-1)  # (B*T,)
                    macro_loss = F.cross_entropy(src_logits, src_labels) + F.cross_entropy(dst_logits, dst_labels)
                    src_pred = src_logits.argmax(dim=1)
                    dst_pred = dst_logits.argmax(dim=1)
                    macro_acc = ((src_pred == src_labels) & (dst_pred == dst_labels)).float().mean()

                if module.micro_type == "poss_edge":
                    assert micro_target is not None and micro_target.dim() == 3
                    assert isinstance(module, (SetLSTM, TranSPORTmer))

                    edge_labels = micro_target[:, :, 0].long() * n_nodes + micro_target[:, :, 1].long()  # (B, T)

                    micro_logits = micro_out.reshape(-1, micro_out.size(-1))  # (B*T, N*N)
                    micro_loss = micro_logits.new_tensor(0.0)
                    if use_emit_loss:
                        micro_loss += F.cross_entropy(micro_logits, edge_labels.reshape(-1))

                    edge_logits = micro_logits.reshape(-1, n_nodes, n_nodes)  # (B*T, N, N)
                    src_logits = torch.logsumexp(edge_logits, dim=2)  # (B*T, N)
                    src_labels = micro_target[:, :, 0].long().reshape(-1)  # (B*T,)
                    if use_src_loss:
                        micro_loss += F.cross_entropy(src_logits, src_labels)

                    dst_logits = torch.logsumexp(edge_logits, dim=1)  # (B*T, N)
                    dst_labels = micro_target[:, :, 1].long().reshape(-1)  # (B*T,)
                    if use_dst_loss:
                        micro_loss += F.cross_entropy(dst_logits, dst_labels)

                    if crf_weight > 0:
                        assert module.crf is not None

                        # Filter out invalid edges (non-self loops of outside nodes) and reindex valid edges
                        edge_labels_comp = module.orig2comp[edge_labels]  # (0..675) to (0..575)
                        emissions = micro_out.index_select(dim=2, index=module.valid_orig_edge_ids).float()
                        if (edge_labels_comp < 0).any():
                            raise ValueError("CRF target includes invalid edges.")

                        if isinstance(module.crf, (DynamicDenseCRF, DynamicSparseCRF)):
                            edge_embeds = edge_embeds.index_select(dim=2, index=module.valid_orig_edge_ids).float()
                            if train and batch_idx == 0:
                                with torch.no_grad():
                                    logZ = module.crf._compute_logZ(emissions, edge_embeds)
                                    gold = module.crf._compute_gold_score(emissions, edge_embeds, edge_labels_comp)
                                    if (logZ < gold).any():
                                        bad_idx = (logZ < gold).nonzero(as_tuple=False).view(-1)[0].item()
                                        module.crf.dump_gold_score(emissions, edge_embeds, edge_labels_comp, bad_idx)
                                        raise ValueError("logZ < gold detected.")
                            crf_loss = module.crf.neg_log_likelihood(emissions, edge_embeds, edge_labels_comp)
                            crf_pred = module.crf.decode(emissions, edge_embeds)  # (B, T)
                        else:
                            crf_loss = module.crf.neg_log_likelihood(emissions, edge_labels_comp)
                            crf_pred = module.crf.decode(emissions)  # (B, T)

                        crf_loss = crf_loss / emissions.size(1)  # Normalize by sequence length (T)
                        micro_acc = (crf_pred == edge_labels_comp).float().mean()
                        src_acc = (module.comp_src[crf_pred].reshape(-1) == src_labels).float().mean()
                        dst_acc = (module.comp_dst[crf_pred].reshape(-1) == dst_labels).float().mean()

                        if (not train) and allowed_mask_comp is not None:
                            forbid_rate = utils.forbid_rate(crf_pred, allowed_mask_comp)
                            loss_dict["forbid_rate"].append(forbid_rate.item())

                    else:
                        crf_loss = torch.tensor(0.0)
                        micro_acc = (micro_logits.argmax(dim=1) == edge_labels.reshape(-1)).float().mean()
                        src_acc = (src_logits.argmax(dim=1) == src_labels).float().mean()
                        dst_acc = (dst_logits.argmax(dim=1) == dst_labels).float().mean()

                        if (not train) and (allowed_mask_comp is not None or allowed_mask_orig is not None):
                            pred_edges = micro_logits.argmax(dim=1).view(edge_labels.shape)  # (B, T)
                            if allowed_mask_comp is not None and hasattr(module, "orig2comp"):
                                pred_edges = module.orig2comp[pred_edges]
                                forbid_rate = utils.forbid_rate(pred_edges, allowed_mask_comp)
                            else:
                                forbid_rate = utils.forbid_rate(pred_edges, allowed_mask_orig)
                            loss_dict["forbid_rate"].append(forbid_rate.item())

                    loss_dict["src_acc"].append(src_acc.item())
                    loss_dict["dst_acc"].append(dst_acc.item())

                    loss = macro_weight * macro_loss + micro_loss + crf_weight * crf_loss

                else:  # module.micro_type == "ball"
                    micro_loss = nn.MSELoss()(micro_out, micro_target)
                    real_loss = utils.calc_real_loss(micro_out, batch_input, player_mask)
                    loss = macro_weight * macro_loss + micro_loss + rloss_weight * real_loss
                    if rloss_weight > 0:
                        loss_dict["real_loss"].append(real_loss.item())

        if train:
            optimizer.zero_grad()
            loss.backward()
            if clip is not None:
                nn.utils.clip_grad_norm_(module.parameters(), clip)
            optimizer.step()

        if macro_loss > 0:
            loss_dict["macro_loss"].append(macro_loss.item())
        if macro_acc is not None:
            loss_dict["macro_acc"].append(macro_acc.item())

        loss_dict["micro_loss"].append(micro_loss.item())
        if module.micro_type == "poss_edge":
            loss_dict["edge_acc"].append(micro_acc.item())
            loss_dict["crf_loss"].append(crf_loss.item())
        else:
            loss_dict["pos_error"].append(utils.calc_dist(micro_out, micro_target))

        if is_main and train and batch_idx % print_batch == 0:
            print(f"[{batch_idx:>{len(str(n_batches))}d}/{n_batches}]  {utils.loss_str(loss_dict)}")

    for key, value in loss_dict.items():
        loss_dict[key] = np.mean(value)

    for key in list(loss_dict.keys()):
        loss_dict[key] = _maybe_all_reduce_mean(loss_dict[key], device=device)

    return loss_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--trial", type=int, required=False)
    macro_choices = ["none", "poss_team", "poss_prev", "poss_next", "poss_player", "poss_edge"]
    parser.add_argument("--macro_type", type=str, choices=macro_choices)
    parser.add_argument("--micro_type", type=str, default="ball", choices=["ball", "poss_edge"])

    parser.add_argument("--data_dir", type=str, default="data/sportec/tracking_processed")
    parser.add_argument("--node_in_dim", type=int, default=8, help="Number of node features to use")
    parser.add_argument("--edge_in_dim", type=int, default=0, help="Number of edge features to use in GNNLSTM")
    parser.add_argument("--team_size", type=int, default=11, help="Max number of players in a team")
    parser.add_argument("--fps", type=float, default=25.0, help="Tracking frame rate")
    parser.add_argument("--sample_freq", type=int, default=5, help="Downsampling frequency")
    parser.add_argument("--window_seconds", type=float, default=10.0, help="Window length in seconds")
    parser.add_argument("--window_stride", type=int, default=1, help="Step size between windows")
    parser.add_argument("--no_self_loops", action="store_true", help="Disable self-loops in the graph")
    parser.add_argument("--flip_pitch", action="store_true", default=False, help="Augment data by flipping the pitch")

    parser.add_argument("--agent_model", type=str, required=True, choices=["gat", "set_tf", "transportmer"])
    parser.add_argument("--seq_model", type=str, required=True, choices=["lstm", "bi_lstm", "sab"])
    crf_choices = [None, "static_dense_crf", "static_sparse_crf", "dynamic_dense_crf", "dynamic_sparse_crf"]
    parser.add_argument("--crf_model", type=str, default=None, choices=crf_choices)
    parser.add_argument("--att_heads", type=int, default=4)
    parser.add_argument("--coarse_dim", type=int, default=16)
    parser.add_argument("--fine_dim", type=int, default=64)
    parser.add_argument("--seq_dim", type=int, default=128)
    parser.add_argument("--crf_edge_dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")

    parser.add_argument("--emit_loss", action="store_true", default=False)
    parser.add_argument("--src_loss", action="store_true", default=False)
    parser.add_argument("--dst_loss", action="store_true", default=False)
    parser.add_argument("--team_loss", action="store_true", default=False)
    parser.add_argument("--macro_weight", type=float, default=0.0, help="Weight for the macro loss")
    parser.add_argument("--rloss_weight", type=float, default=0.0, help="Weight for the reality loss")
    parser.add_argument("--crf_weight", type=float, default=0.0, help="Weight for the CRF loss")
    parser.add_argument("--l1_weight", type=float, required=False, default=0, help="Weight for the L1 loss")

    parser.add_argument("--n_epochs", type=int, default=50, help="Num epochs")
    parser.add_argument("--ce_epochs", type=int, default=0, help="Only use CE loss for the first N epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--start_lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-4, help="Minimum learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="L2 weight decay for Adam")
    parser.add_argument("--clip", type=float, default=None, help="Gradient clipping (L2 norm)")
    parser.add_argument("--seed", type=int, default=128, help="PyTorch random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers per process")

    parser.add_argument("--print_batch", type=int, default=100, help="Periodically print performance")
    parser.add_argument("--save_epoch", type=int, default=10, help="Periodically save model")
    parser.add_argument("--resume_file", type=str, default=None, help="Model file name under saved/{trial}/model")
    parser.add_argument("--best_loss", type=float, default=None, help="Best total loss to resume from")
    parser.add_argument("--best_pos_error", type=float, default=None, help="Best position error to resume from")
    parser.add_argument("--best_acc", type=float, default=None, help="Best micro accuracy to resume from")

    args, _ = parser.parse_known_args()
    args_dict = vars(args).copy()

    distributed, rank, world_size, local_rank = _init_distributed()
    is_main = rank == 0

    # Set device and manual seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if distributed else "cuda:0")
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load model
    model = utils.build_model(args_dict, device)

    if distributed:
        if torch.cuda.is_available():
            model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        else:
            model = DistributedDataParallel(model)
        module: nn.Module = model.module
    else:
        module: nn.Module = model

    args_dict["distributed"] = distributed
    args_dict["world_size"] = world_size
    args_dict["total_params"] = utils.num_trainable_params(module)

    # Create save path and saving parameters
    save_path = f"saved/{args.trial:03d}"
    if is_main:
        print("Generating datasets...")
        os.makedirs(f"{save_path}/model", exist_ok=True)
        with open(f"{save_path}/args.json", "w") as f:
            json.dump(args_dict, f, indent=4)

    data_paths = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith(".parquet")]
    data_paths.sort()
    train_paths = data_paths[:5]
    valid_paths = data_paths[5:6]

    dataset_args = {
        "node_in_dim": args.node_in_dim,
        "edge_in_dim": args.edge_in_dim,
        "fps": args.fps,
        "sample_freq": args.sample_freq,
        "window_seconds": args.window_seconds,
        "window_stride": args.window_stride,
        "self_loops": not args.no_self_loops,
        "flip_pitch": args.flip_pitch,
    }

    if args.agent_model == "gat":
        train_dataset = SoccerWindowGraphs(train_paths, verbose=is_main, **dataset_args)
        valid_dataset = SoccerWindowGraphs(valid_paths, verbose=is_main, **dataset_args)
    else:
        train_dataset = SoccerWindowTensors(train_paths, verbose=is_main, **dataset_args)
        valid_dataset = SoccerWindowTensors(valid_paths, verbose=is_main, **dataset_args)

    pin_memory = torch.cuda.is_available()
    num_workers = args.num_workers

    train_sampler = None
    valid_sampler = None
    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=args.seed, drop_last=False)
        valid_sampler = DistributedSampler(valid_dataset, shuffle=False, seed=args.seed, drop_last=False)

    collate_fn = utils.collate_window_batch if args.agent_model == "gat" else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=valid_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    # Train loop
    best_total_loss = np.inf if args.best_loss is None else args.best_loss
    best_pretrain_loss = np.inf
    best_pos_error = np.inf if args.best_pos_error is None else args.best_pos_error
    best_micro_acc = 0 if args.best_acc is None else args.best_acc
    epochs_since_best = 0
    lr = max(args.start_lr, args.min_lr)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, module.parameters()), lr=lr, weight_decay=args.weight_decay
    )

    if args.resume_file is not None:
        resume_path = os.path.join(save_path, "model", args.resume_file)
        if is_main:
            utils.printlog(f"Loading resume checkpoint: {resume_path}", save_path)
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        module.load_state_dict(ckpt)

    for e in range(args.n_epochs):
        epoch = e + 1
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if epochs_since_best >= 3 and lr > args.min_lr:
            lr = max(lr * 0.5, args.min_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            if is_main:
                utils.printlog(f"########## lr {lr} ##########", save_path)
            epochs_since_best = 0

        if is_main:
            utils.printlog(f"\nEpoch {epoch:d}", save_path)
        start_time = time.time()

        rloss_weight = args.rloss_weight if args.agent_model.startswith("set") else 0
        crf_weight = args.crf_weight if epoch > args.ce_epochs else 0.0
        train_losses = run_epoch(
            model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            macro_weight=args.macro_weight,
            rloss_weight=rloss_weight,
            crf_weight=crf_weight,
            use_emit_loss=args.emit_loss,
            use_src_loss=args.src_loss,
            use_dst_loss=args.dst_loss,
            clip=args.clip,
            train=True,
            is_main=is_main,
            print_batch=args.print_batch,
        )
        if is_main:
            utils.printlog(f"Train:\t {utils.loss_str(train_losses)}", save_path)

        valid_losses = run_epoch(
            model,
            valid_loader,
            device,
            macro_weight=args.macro_weight,
            rloss_weight=rloss_weight,
            crf_weight=crf_weight,
            use_emit_loss=args.emit_loss,
            use_src_loss=args.src_loss,
            use_dst_loss=args.dst_loss,
            train=False,
        )
        if is_main:
            utils.printlog(f"Valid:\t {utils.loss_str(valid_losses)}", save_path)
            utils.printlog(f"Time:\t {time.time() - start_time:.2f}s", save_path)

        valid_total_loss = sum([value for key, value in valid_losses.items() if key.endswith("loss")])

        # Best model on validation set
        if args.crf_weight > 0 and epoch <= args.ce_epochs:
            if best_pretrain_loss == np.inf or valid_total_loss < best_pretrain_loss:
                best_pretrain_loss = valid_total_loss
                if is_main:
                    path = f"{save_path}/model/state_dict_best_pretrain.pt"
                    torch.save(module.state_dict(), path)
                    utils.printlog("###### Best Pretrain Loss #######", save_path)
        else:
            force_best = args.crf_weight > 0 and epoch == args.ce_epochs + 1
            if force_best or best_total_loss == np.inf or valid_total_loss < best_total_loss:
                best_total_loss = valid_total_loss
                epochs_since_best = 0
                if is_main:
                    path = f"{save_path}/model/state_dict_best.pt"
                    torch.save(module.state_dict(), path)
                    utils.printlog("######## Best Total Loss ########", save_path)
            else:
                epochs_since_best += 1

            if "edge_acc" in valid_losses and valid_losses["edge_acc"] > best_micro_acc:
                best_micro_acc = valid_losses["edge_acc"]
                epochs_since_best = 0
                if is_main:
                    path = f"{save_path}/model/state_dict_best_acc.pt"
                    torch.save(module.state_dict(), path)
                    utils.printlog("######### Best Accuracy #########", save_path)

            if "pos_error" in valid_losses and (best_pos_error == np.inf or valid_losses["pos_error"] < best_pos_error):
                best_pos_error = valid_losses["pos_error"]
                epochs_since_best = 0
                if is_main:
                    path = f"{save_path}/model/state_dict_best_pe.pt"
                    torch.save(module.state_dict(), path)
                    utils.printlog("######## Best Pos Error #########", save_path)

        # Periodically save model
        if is_main and epoch % args.save_epoch == 0:
            path = f"{save_path}/model/state_dict_{epoch}.pt"
            torch.save(module.state_dict(), path)
            utils.printlog("########## Saved Model ##########", save_path)

    if is_main:
        utils.printlog(f"Best Valid Loss: {best_total_loss:.4f}", save_path)

    if distributed and dist.is_initialized():
        dist.destroy_process_group()
