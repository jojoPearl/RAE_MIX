import argparse
import logging
import math
import os
import json
from collections import defaultdict, OrderedDict
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from PIL import Image, ImageDraw
from copy import deepcopy
from glob import glob
from time import time
from pathlib import Path
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from omegaconf import OmegaConf
from torchvision.utils import save_image, make_grid


##### model imports
from stage1 import RAE
from stage2.models import Stage2ModelProtocol
from stage2.transport import create_transport, Sampler

##### general utils
from utils import wandb_utils
from utils.model_utils import instantiate_from_config
from utils.train_utils import *
from utils.optim_utils import build_optimizer, build_scheduler
from utils.resume_utils import *
from utils.wandb_utils import *
from utils.dist_utils import *

##### Eval utils
from eval import evaluate_generation_distributed

class EdgeResidualAdapter(nn.Module):
    """
    edge_rgb (B,3,256,256) -> per-layer residuals: List[(B,L,D)]
    """
    def __init__(
        self,
        num_layers: int,
        dec_dim: int,
        edge_in_ch: int = 3,
        edge_feat_ch: int = 768,
        hw: int = 16,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dec_dim = dec_dim
        self.hw = hw
        self.cnn = nn.Sequential(
            nn.Conv2d(edge_in_ch, 64, 3, 2, 1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv2d(256, edge_feat_ch, 3, 2, 1),
            nn.GroupNorm(8, edge_feat_ch),
            nn.SiLU(),
        )
        self.to_dec = nn.Linear(edge_feat_ch, dec_dim)
        self.per_layer = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dec_dim),
                nn.Linear(dec_dim, dec_dim),
            )
            for _ in range(num_layers)
        ])
        for head in self.per_layer:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, edge_rgb: torch.Tensor, Lx: int) -> list:
        feat = self.cnn(edge_rgb)
        B, C, H, W = feat.shape
        assert H == self.hw and W == self.hw, f"expect {self.hw}x{self.hw}, got {H}x{W}"
        tok = feat.flatten(2).transpose(1, 2)
        tok = self.to_dec(tok)
        if tok.shape[1] != Lx:
            tok = F.interpolate(
                tok.transpose(1, 2),
                size=Lx,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        residuals = [head(tok) for head in self.per_layer]
        return residuals


class ResidualDDTWrapper(nn.Module):
    """
    Wrap base diffusion model:
    - keep in_channels unchanged
    - inject per-layer residuals from edge adapter
    """
    def __init__(self, base_model, edge_adapter):
        super().__init__()
        self.base = base_model
        self.edge_adapter = edge_adapter

    def _get_Lx(self, x: torch.Tensor) -> int:
        if hasattr(self.base, "x_embedder") and hasattr(self.base.x_embedder, "num_patches"):
            return int(self.base.x_embedder.num_patches)
        if hasattr(self.base, "x_patch_size"):
            ps = int(self.base.x_patch_size)
            return (x.shape[2] // ps) * (x.shape[3] // ps)
        return 256

    def forward(self, x, t, y=None, edge_latent=None, **kwargs):
        control_residuals = None
        if edge_latent is not None:
            Lx = self._get_Lx(x)
            control_residuals = self.edge_adapter(edge_latent, Lx=Lx)
        return self.base(x, t, y=y, control_residuals=control_residuals, **kwargs)

    def forward_with_cfg(self, x, t, y, cfg_scale, edge_latent=None, **kwargs):
        control_residuals = None
        if edge_latent is not None:
            Lx = self._get_Lx(x)
            control_residuals = self.edge_adapter(edge_latent, Lx=Lx)
        return self.base.forward_with_cfg(
            x, t, y, cfg_scale, control_residuals=control_residuals, **kwargs
        )

def annotate(img: torch.Tensor, text: str) -> torch.Tensor:
    """
    img: (3, H, W) in [0, 1]
    """
    img = img.detach().clamp(0, 1)
    img_np = (img * 255).byte().permute(1, 2, 0).cpu().numpy()
    pil = Image.fromarray(img_np)
    draw = ImageDraw.Draw(pil)
    draw.text((5, 5), text, fill=(255, 0, 0))
    return transforms.ToTensor()(pil)


def save_checkpoint(
    path: str,
    step: int,
    epoch: int,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
) -> None:
    state = {
        "step": step,
        "epoch": epoch,
        "model": model.module.state_dict(),
        "ema": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
) -> Tuple[int, int]:
    checkpoint = torch.load(path, map_location="cpu")
    model.module.load_state_dict(checkpoint["model"])
    ema_model.load_state_dict(checkpoint["ema"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint.get("epoch", 0), checkpoint.get("step", 0)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-2 transport model on RAE latents.")
    parser.add_argument("--config", type=str, required=True, help="YAML config containing stage_1 and stage_2 sections.")
    parser.add_argument("--data-path", type=Path, required=True, help="Directory with ImageFolder structure for training.")
    parser.add_argument("--results-dir", type=str, default="ckpts", help="Directory to store training outputs.")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256, help="Input image resolution.")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="fp32", help="Compute precision for training.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--compile", action="store_true", help="Use torch compile (for rae.encode and model.forward).")
    parser.add_argument("--ckpt", type=str, default=None, help="Optional checkpoint path to resume training.")
    parser.add_argument("--global-seed", type=int, default=None, help="Override training.global_seed from the config.")
    args = parser.parse_args()
    return args
def main():
    """Trains a new SiT model using config-driven hyperparameters."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Training currently requires at least one GPU.")
    rank, world_size = 0, 1
    # limited by current machine
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    full_cfg = OmegaConf.load(args.config)
    (
        rae_config,
        model_config,
        transport_config,
        sampler_config,
        guidance_config,
        misc_config,
        training_config,
        eval_config
    ) = parse_configs(full_cfg)

    if rae_config is None or model_config is None:
        raise ValueError("Config must provide both stage_1 and stage_2 sections.")

    def to_dict(cfg_section):
        if cfg_section is None:
            return {}
        return OmegaConf.to_container(cfg_section, resolve=True)

    misc = to_dict(misc_config)
    transport_cfg = to_dict(transport_config)
    sampler_cfg = to_dict(sampler_config)
    guidance_cfg = to_dict(guidance_config)
    training_cfg = to_dict(training_config)
    ilsvrc_class_index = training_cfg.get("ilsvrc_class_index") or os.environ.get("IMAGENET_CLASS_INDEX")

    num_classes = int(misc.get("num_classes", 1000))
    null_label = int(misc.get("null_label", num_classes))
    latent_size = tuple(int(dim) for dim in misc.get("latent_size", (768, 16, 16)))
    shift_dim = misc.get("time_dist_shift_dim", math.prod(latent_size))
    shift_base = misc.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(shift_dim / shift_base)

    grad_accum_steps = int(training_cfg.get("grad_accum_steps", 1))
    clip_grad_val = training_cfg.get("clip_grad", 1.0)
    clip_grad = float(clip_grad_val) if clip_grad_val is not None else None
    if clip_grad is not None and clip_grad <= 0:
        clip_grad = None
    ema_decay = float(training_cfg.get("ema_decay", 0.9995))
    num_epochs = int(training_cfg.get("epochs", 1400))
    
    # Force logic: ignore global_batch_size, use batch_size
    batch_size = int(training_cfg.get("batch_size", 2))
    global_batch_size = batch_size * world_size * grad_accum_steps
            
    num_workers = int(training_cfg.get("num_workers", 4))
    sample_every = int(training_cfg.get("sample_every", 2500)) 
    checkpoint_interval = int(training_cfg.get("checkpoint_interval", 4)) 
    cfg_scale_override = training_cfg.get("cfg_scale", None)
    default_seed = int(training_cfg.get("global_seed", 0))
    edge_dropout = float(training_cfg.get("edge_dropout", 0.0))
    label_drop_rate = float(training_cfg.get("label_drop_rate", 0.1))
    
    do_eval = False

    global_seed = args.global_seed if args.global_seed is not None else default_seed
    seed = global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    micro_batch_size = batch_size
        
    use_fp16 = args.precision == "fp16"
    use_bf16 = args.precision == "bf16"
    autocast_dtype = torch.float16 if use_fp16 else torch.bfloat16
    autocast_enabled = use_fp16 or use_bf16
    autocast_kwargs = dict(dtype=autocast_dtype, enabled=autocast_enabled)
    scaler = GradScaler(enabled=use_fp16)

    transport_params = dict(transport_cfg.get("params", {}))
    path_type = transport_params.get("path_type", "Linear")
    prediction = transport_params.get("prediction", "velocity")
    loss_weight = transport_params.get("loss_weight")
    transport_params.pop("time_dist_shift", None)

    sampler_mode = sampler_cfg.get("mode", "ODE").upper()
    sampler_params = dict(sampler_cfg.get("params", {}))

    guidance_scale = float(guidance_cfg.get("scale", 1.0))
    if cfg_scale_override is not None:
        guidance_scale = float(cfg_scale_override)
    guidance_method = guidance_cfg.get("method", "cfg")
    t_min = 0.0
    t_max = 1.0
    
    experiment_dir, checkpoint_dir, logger = configure_experiment_dirs(args, rank)
    if rank == 0 and ilsvrc_class_index:
        logger.info(f"Using ILSVRC class index mapping: {ilsvrc_class_index}")
    idx2name = None
    if ilsvrc_class_index:
        try:
            with open(ilsvrc_class_index, "r") as f:
                class_index = json.load(f)
            idx2name = {int(k): v[1] for k, v in class_index.items()}
        except Exception as exc:
            if rank == 0:
                logger.warning(f"Failed to load class names from {ilsvrc_class_index}: {exc}")
    samples_dir = os.path.join(experiment_dir, "samples_ab")
    os.makedirs(samples_dir, exist_ok=True)
    
    #### Model init
    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval()
    base_model: Stage2ModelProtocol = instantiate_from_config(model_config).to(device) 

    # Load Pretrained Base Model
    pretrained_ckpt_path = "/home/bjia-25/workspace/papers/RAE/code/rae_project/RAE_MIX/models/DiTs/Dinov2/wReg_base/ImageNet256/DiTDH-XL/stage2_model.pt"
    if os.path.exists(pretrained_ckpt_path):
        if rank == 0:
            logger.info(f"Loading base model: {pretrained_ckpt_path}")
        ckpt = torch.load(pretrained_ckpt_path, map_location="cpu")
        sd = ckpt["ema"] if "ema" in ckpt else ckpt["model"] if "model" in ckpt else ckpt
        new_sd = OrderedDict()
        for k, v in sd.items():
            if k.startswith("module."):
                k = k[7:]
            new_sd[k] = v
        base_model.load_state_dict(new_sd, strict=False)
        if rank == 0:
            logger.info("Base model loaded.")
    
    if args.compile:
        try:
            rae.encode = torch.compile(rae.encode)
        except:
            pass

    # === Scheme A: per-layer residual adapter ===
    base_model.requires_grad_(False)
    base_model.eval()

    num_layers = len(base_model.blocks)
    dec_dim = getattr(base_model, "decoder_hidden_size", getattr(base_model, "hidden_size", 2048))

    edge_adapter = EdgeResidualAdapter(
        num_layers=num_layers,
        dec_dim=dec_dim,
        edge_in_ch=3,
        edge_feat_ch=latent_size[0],
        hw=16,
    ).to(device)

    model = ResidualDDTWrapper(base_model, edge_adapter).to(device)
    ema_model = model
    for p in model.parameters():
        p.requires_grad = False
    for p in edge_adapter.parameters():
        p.requires_grad = True

    ddp_model = model
    ddp_model.train()
    base_model.eval()
    
    def unwrap(m):
        return m.module if hasattr(m, "module") else m
        
    model_param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model Parameters: {model_param_count/1e6:.2f}M")

    #### Opt
    training_cfg["fused"] = False
    optimizer = torch.optim.AdamW(
        list(edge_adapter.parameters()),
        lr=float(training_cfg.get("optimizer", {}).get("lr", 1e-4)),
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    scheduler = None

    scaler, autocast_kwargs = get_autocast_scaler(args)
    
    ### Data
    stage2_transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
    ])
    loader, sampler = prepare_dataloader(
        args.data_path,
        micro_batch_size,
        num_workers,
        rank,
        world_size,
        transform=stage2_transform,
        image_size=args.image_size,
        return_edges=True,
        ilsvrc_class_index_path=ilsvrc_class_index,
    )

    # ============================================================
    # [NEW] Part A: Independent Visualization Loader (non-DDP)
    # ============================================================
    if rank == 0:
        vis_dataset = CannyImageFolder(
            str(args.data_path),
            image_size=args.image_size,
            random_flip=True,
            ilsvrc_class_index_path=ilsvrc_class_index,
        )
        vis_loader = DataLoader(
            vis_dataset,
            batch_size=5,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        vis_iter = iter(vis_loader)
        logger.info(f"[Init] Visualization loader ready. (Size: {len(vis_loader)})")
    else:
        vis_loader = None
        vis_iter = None
    
    loader_batches = len(loader)
    steps_per_epoch = loader_batches // grad_accum_steps
    
    transport = create_transport(**transport_params, time_dist_shift=time_dist_shift)
    transport_sampler = Sampler(transport)
    eval_sampler = transport_sampler.sample_ode(**sampler_params)
    
    # Guidance
    sample_model_kwargs = dict()
    ema_model_fn = ema_model.forward
    train_model_fn = unwrap(ddp_model)

    start_epoch = 0
    global_step = 0
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
        if ckpt_path.is_dir():
            maybe_ckpt = find_resume_checkpoint(str(ckpt_path))
            if maybe_ckpt is None:
                raise FileNotFoundError(f"No checkpoint found in {ckpt_path}")
            ckpt_path = Path(maybe_ckpt)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        loaded_epoch, global_step = load_checkpoint(
            ckpt_path,
            ddp_model,
            ema_model,
            optimizer,
            scheduler,
        )
        start_epoch = loaded_epoch + 1
        logger.info(
            f"[Rank {rank}] Resumed from {ckpt_path} (epoch={loaded_epoch}, step={global_step})."
        )

    def to_01(x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        x = x.float()
        signed = (x.min() < -0.05).item()
        if signed:
            x = (x + 1.0) * 0.5
        x_min = x.amin(dim=(1, 2, 3), keepdim=True)
        x_max = x.amax(dim=(1, 2, 3), keepdim=True)
        too_narrow = ((x_max - x_min) < 0.2).float()
        x_stretch = (x - x_min) / (x_max - x_min + 1e-6)
        x = too_narrow * x_stretch + (1 - too_narrow) * x
        return x.clamp(0, 1)
    
    # Training Loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        model.base.eval()
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        epoch_metrics: Dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(1, device=device))
        num_batches = 0
        optimizer.zero_grad(set_to_none=True)
        
        for step, (images, labels, edges) in enumerate(loader):
            # =======================================================
            # Label dropout for classifier-free guidance
            # =======================================================
            real_labels = labels.to(device)
            null_labels = torch.full_like(real_labels, null_label)
            mask = torch.rand(real_labels.shape[0], device=device) < label_drop_rate
            train_labels = torch.where(mask, null_labels, real_labels)

            images = images.to(device)
            edge_rgb = edges.to(device)
            
            with torch.no_grad():
                z = rae.encode(images)
                # === [Change 2] Feed raw edge image ===
                edge_condition = edge_rgb * 2.0 - 1.0

            is_dropped = torch.rand((), device=edge_condition.device).item() < edge_dropout
            edge_for_model = None if is_dropped else edge_condition
                
            with autocast(**autocast_kwargs):
                loss_dict = transport.training_losses(
                    train_model_fn,
                    z,
                    {"y": train_labels, "edge_latent": edge_for_model},
                    return_intermediates=False,
                )
                loss = loss_dict["loss"].mean()
            
            loss = loss.float()
            if not loss.requires_grad:
                if rank == 0:
                    logger.warning("[WARN] loss has no grad; skip backward this step")
                optimizer.zero_grad(set_to_none=True)
                continue
            if scaler:
                scaler.scale(loss / grad_accum_steps).backward()
            else:
                (loss / grad_accum_steps).backward()
            
            if (step + 1) % grad_accum_steps == 0:
                if clip_grad:
                    if scaler:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(unwrap(ddp_model).edge_adapter.parameters()),
                        clip_grad,
                    )
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                # EMA disabled for ControlNet
                global_step += 1
                if (
                    rank == 0
                    and training_cfg.get("checkpoint_every_steps")
                    and (global_step % int(training_cfg["checkpoint_every_steps"]) == 0)
                ):
                    ckpt_path = os.path.join(checkpoint_dir, f"step-{global_step:07d}.pt")
                    logger.info(f"Saving checkpoint at step {global_step}...")
                    save_checkpoint(
                        ckpt_path,
                        global_step,
                        epoch + 1,
                        ddp_model,
                        ema_model,
                        optimizer,
                        scheduler,
                    )
                if rank == 0 and global_step % 100 == 0:
                    logger.info(
                        f"[step {global_step}] "
                        f"loss={loss.item():.4f} "
                        f"diff={loss_diff.item():.4f}"
                    )
                if rank == 0 and global_step % 100 == 0:
                    with torch.no_grad():
                        x1_chk = z[:1].detach()
                        t_chk, x0_chk, _ = transport.sample(x1_chk)
                        t_chk, xt_chk, _ = transport.path_sampler.plan(t_chk, x0_chk, x1_chk)
                        y_chk = real_labels[:1]
                        e_chk = edge_condition[:1]

                        p_edge = train_model_fn(xt_chk, t_chk, y=y_chk, edge_latent=e_chk)
                        p_no = train_model_fn(xt_chk, t_chk, y=y_chk, edge_latent=None)
                        diff = (p_edge - p_no).abs().mean().item()

                        adapter = unwrap(ddp_model).edge_adapter
                        if hasattr(base_model, "x_embedder") and hasattr(base_model.x_embedder, "num_patches"):
                            Lx = int(base_model.x_embedder.num_patches)
                        elif hasattr(base_model, "x_patch_size"):
                            ps = int(base_model.x_patch_size)
                            Lx = (xt_chk.shape[2] // ps) * (xt_chk.shape[3] // ps)
                        else:
                            Lx = 256
                        residuals = adapter(e_chk, Lx=Lx)
                        r0 = residuals[0]
                        c_absmean = r0.abs().mean().item()
                        c_std = r0.std().item()
                        del residuals
                        del r0

                        logger.info(
                            f"[EDGE_SENS] step={global_step} | "
                            f"pred_diff={diff:.3e} | "
                            f"res_absmean={c_absmean:.3e} | res_std={c_std:.3e}"
                        )
            
            epoch_metrics['loss'] += loss.detach()
            
            # Sampling
            if (step + 1) % grad_accum_steps == 0:
                if (global_step > 0) and (global_step % sample_every == 0) and rank == 0:
                    model.eval()
                    logger.info(f"[{global_step}] Generating EMA samples (Decoupled Batch)...")

                    # -------------------------------------------------
                    # 1. Get decoupled visualization batch
                    # -------------------------------------------------
                    try:
                        vis_images, vis_labels, vis_edges = next(vis_iter)
                    except StopIteration:
                        vis_iter = iter(vis_loader)
                        vis_images, vis_labels, vis_edges = next(vis_iter)

                    n_vis = min(5, vis_images.shape[0])
                    vis_images = vis_images[:n_vis].to(device)
                    vis_labels = vis_labels[:n_vis].to(device)
                    vis_edges = vis_edges[:n_vis].to(device)

                    # -----------------------
                    # 0) Prepare GT + edge condition (same as training)
                    # -----------------------
                    if vis_edges.shape[1] == 1:
                        edge_vis_rgb = vis_edges.repeat(1, 3, 1, 1)
                    else:
                        edge_vis_rgb = vis_edges
                    gt_img = vis_images.detach().cpu()
                    gt_edge = edge_vis_rgb.detach().cpu().clamp(0, 1)
                    edge_vis_cond = (edge_vis_rgb * 2.0 - 1.0).to(device)

                    # -----------------------
                    # 2) Inference samples (same noise for fair compare)
                    # -----------------------
                    zs = torch.randn(n_vis, *latent_size, device=device, dtype=torch.float32)

                    with torch.no_grad(), autocast(**autocast_kwargs):
                        def fn_y_edge(x, t, **kw):
                            return ema_model_fn(x, t, y=vis_labels, edge_latent=edge_vis_cond)

                        def fn_y_noedge(x, t, **kw):
                            return ema_model_fn(x, t, y=vis_labels, edge_latent=None)

                        traj = eval_sampler(
                            zs,
                            fn_y_edge,
                            y=vis_labels,
                            **sample_model_kwargs,
                        )
                        lat_edge = traj[-1]
                        del traj

                        traj = eval_sampler(
                            zs,
                            fn_y_noedge,
                            y=vis_labels,
                            **sample_model_kwargs,
                        )
                        lat_noedge = traj[-1]
                        del traj

                    img_edge = to_01(rae.decode(lat_edge).float()).cpu()
                    img_noedge = to_01(rae.decode(lat_noedge).float()).cpu()
                    del lat_edge
                    del lat_noedge
                    torch.cuda.empty_cache()

                    # -----------------------
                    # 3) Build 5-col grid
                    # -----------------------
                    rows = []
                    for i in range(n_vis):
                        y_i = int(vis_labels[i].item())
                        if idx2name is None:
                            cls_name = f"y={y_i}"
                        else:
                            cls_name = idx2name.get(y_i, f"unknown_{y_i}")

                        col1 = annotate(gt_img[i], f"GT | {cls_name}")
                        col2 = annotate(gt_edge[i], "GT edge")
                        col3 = annotate(img_edge[i], "Infer: y + GT edge")
                        col4 = annotate(img_noedge[i], "Infer: y (no edge)")

                        row = torch.cat([col1, col2, col3, col4], dim=2)
                        rows.append(row)

                    final_grid = torch.cat(rows, dim=1)
                    save_path = os.path.join(
                        samples_dir,
                        f"step_{global_step:07d}_4col_GT_Edge_InferEdge_InferNoEdge.png",
                    )
                    save_image(final_grid, save_path)
                    logger.info(f"Saved: {save_path}")

                    model.train()
            
                if training_cfg.get("max_steps") and global_step >= int(training_cfg["max_steps"]):
                    if rank == 0:
                        ckpt_path = os.path.join(checkpoint_dir, f"step-{global_step:07d}.pt")
                        logger.info(f"Saving final checkpoint at step {global_step}...")
                        save_checkpoint(
                            ckpt_path,
                            global_step,
                            epoch,
                            ddp_model,
                            ema_model,
                            optimizer,
                            scheduler,
                        )
                    return
            num_batches += 1
        if rank == 0 and checkpoint_interval > 0 and ((epoch + 1) % checkpoint_interval == 0):
            ckpt_path = os.path.join(
                checkpoint_dir,
                f"epoch-{epoch + 1:04d}_step-{global_step:07d}.pt",
            )
            logger.info(f"Saving checkpoint at epoch {epoch + 1}...")
            save_checkpoint(
                ckpt_path,
                global_step,
                epoch + 1,
                ddp_model,
                ema_model,
                optimizer,
                scheduler,
            )

    cleanup_distributed()



if __name__ == "__main__":
    main()
