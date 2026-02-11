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

class LatentMixer(nn.Module):
    """
    Switch to a CNN-based Adapter (ControlNet-Lite style).
    Directly maps pixel-space edges (B, 3, 256, 256) to Latent-space (B, 768, 16, 16).
    
    Structure: 4 layers of Conv2d with stride 2.
    256 -> 128 -> 64 -> 32 -> 16.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 768):
        super().__init__()
        # Define a lightweight CNN
        self.net = nn.Sequential(
            # 256 -> 128
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            # 128 -> 64
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            # 64 -> 32
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            # 32 -> 16
            nn.Conv2d(256, out_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            # Final projection (Zero Init)
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )
        
        # Add a tiny perturbation to the last layer to avoid gradient dead-ends
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, edge_image: torch.Tensor) -> torch.Tensor:
        # Input: (B, 3, 256, 256)
        # Output: (B, 768, 16, 16)
        return self.net(edge_image)

def patch_dit_input_layer(model, extra_channels):
    """
    Full expansion for DDT:
      Expand both s_embedder and x_embedder to accept extra channels.
    """
    targets = ["s_embedder", "x_embedder"]
    patched_count = 0
    new_in_channels = None

    for name in targets:
        if not hasattr(model, name):
            continue
        embedder = getattr(model, name)

        # unwrap any previous wrapper
        if hasattr(embedder, "original"):
            print(f"[Patch] Unwrapping {name} before expansion...")
            embedder = embedder.original

        if hasattr(embedder, "proj"):
            old_proj = embedder.proj
            is_obj = True
        elif isinstance(embedder, nn.Conv2d):
            old_proj = embedder
            is_obj = False
        else:
            continue

        out_channels, old_in_channels, k_h, k_w = old_proj.weight.shape
        if old_in_channels > 1000:
            print(f"[Info] {name} already expanded ({old_in_channels}), skip.")
            patched_count += 1
            new_in_channels = old_in_channels
            continue

        new_in_channels = old_in_channels + extra_channels
        new_proj = nn.Conv2d(
            new_in_channels,
            out_channels,
            kernel_size=(k_h, k_w),
            stride=old_proj.stride,
            padding=old_proj.padding,
            bias=(old_proj.bias is not None),
        ).to(old_proj.weight.device)
        with torch.no_grad():
            new_proj.weight[:, :old_in_channels] = old_proj.weight
            nn.init.normal_(new_proj.weight[:, old_in_channels:], mean=0.0, std=1e-3)
            if old_proj.bias is not None:
                new_proj.bias = old_proj.bias

        if is_obj:
            embedder.proj = new_proj
            setattr(model, name, embedder)
        else:
            setattr(model, name, new_proj)

        print(f"[Patch Success] {name} expanded: {old_in_channels} -> {new_in_channels}")
        patched_count += 1

    if new_in_channels is not None:
        model.in_channels = new_in_channels
    return model


class ConcatDiTWrapper(nn.Module):
    """
    Concatenate noisy latent and edge latent along channel dimension.
    """
    def __init__(self, base_model, mixer):
        super().__init__()
        self.base = base_model
        self.mixer = mixer
        self.mixer_layer0_mean = 0.0
        self.edge_t_gate = 0.6
        self.edge_boost = 3.0

    def forward(self, x, t, y=None, edge_latent=None, **kwargs):
        if edge_latent is not None:
            c = self.mixer(edge_latent)
            tt = t
            if tt.dim() == 0:
                tt = tt.view(1, 1, 1, 1)
            elif tt.dim() == 1:
                tt = tt.view(-1, 1, 1, 1)
            elif tt.dim() == 2:
                tt = tt.view(-1, 1, 1, 1)
            tt = tt.to(device=c.device, dtype=c.dtype)
            boost = torch.full_like(tt, self.edge_boost)
            one = torch.ones_like(tt)
            w = torch.where(tt > self.edge_t_gate, boost, one)
            c = c * w
            with torch.no_grad():
                self.mixer_layer0_mean = c.mean().item()
            x_in = torch.cat([x, c], dim=1)
        else:
            b, _, h, w = x.shape
            c_cond = self.base.in_channels - x.shape[1]
            dummy_c = torch.zeros(b, c_cond, h, w, device=x.device, dtype=x.dtype)
            x_in = torch.cat([x, dummy_c], dim=1)
        return self.base(x_in, t, y=y, **kwargs)

    def forward_with_cfg(self, x, t, y, cfg_scale, edge_latent=None, **kwargs):
        b2 = x.shape[0]
        b = b2 // 2
        if edge_latent is not None:
            c = self.mixer(edge_latent)
            c_uncond = torch.zeros_like(c)
            c_combined = torch.cat([c, c_uncond], dim=0)
            x_in = torch.cat([x, c_combined], dim=1)
        else:
            c_cond = self.base.in_channels - x.shape[1]
            dummy = torch.zeros(b2, c_cond, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
            x_in = torch.cat([x, dummy], dim=1)
        return self.base.forward_with_cfg(x_in, t, y, cfg_scale, **kwargs)


class BoundaryAwareLoss(nn.Module):
    """
    Sobel-based boundary alignment loss between predicted clean latent and edge map.
    """
    def __init__(self, device: Optional[torch.device] = None):
        super().__init__()
        gx = torch.tensor(
            [[-1.0, 0.0, 1.0],
             [-2.0, 0.0, 2.0],
             [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        gy = torch.tensor(
            [[-1.0, -2.0, -1.0],
             [0.0, 0.0, 0.0],
             [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_gx", gx)
        self.register_buffer("sobel_gy", gy)
        if device is not None:
            self.to(device)

    def forward(self, x0_pred: torch.Tensor, edge_gt_rgb: torch.Tensor) -> torch.Tensor:
        # edge_gt_rgb: [B, 3, 256, 256] in [0,1]
        edge_small = F.interpolate(
            edge_gt_rgb.mean(1, keepdim=True), size=(16, 16), mode="area"
        )
        latent_map = torch.sqrt(torch.mean(x0_pred ** 2, dim=1, keepdim=True) + 1e-8)
        g_x = F.conv2d(latent_map, self.sobel_gx, padding=1)
        g_y = F.conv2d(latent_map, self.sobel_gy, padding=1)
        pred_edge = torch.sqrt(g_x ** 2 + g_y ** 2 + 1e-6)

        pred_edge = pred_edge / (pred_edge.mean(dim=[2, 3], keepdim=True) + 1e-6)
        edge_small = edge_small / (edge_small.mean(dim=[2, 3], keepdim=True) + 1e-6)

        loss_boundary = F.l1_loss(pred_edge, edge_small)
        return loss_boundary


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
        "model": (model.module.state_dict() if hasattr(model, "module") else model.state_dict()),
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
    if hasattr(model, "module"):
        model.module.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint["model"])
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

    # === Input concatenation components ===
    base_model.requires_grad_(False)
    base_model.eval()

    latent_mixer = LatentMixer(in_channels=3, out_channels=latent_size[0]).to(device)
    nn.init.normal_(latent_mixer.net[-1].weight, mean=0.0, std=1e-3)
    nn.init.zeros_(latent_mixer.net[-1].bias)

    base_model = patch_dit_input_layer(base_model, extra_channels=latent_size[0])
    model = ConcatDiTWrapper(base_model, latent_mixer).to(device)
    ema_model = model
    for p in model.parameters():
        p.requires_grad = False
    for p in latent_mixer.parameters():
        p.requires_grad = True
    if hasattr(base_model, "x_embedder"):
        for p in base_model.x_embedder.parameters():
            p.requires_grad = True
    elif hasattr(base_model, "patch_embed"):
        for p in base_model.patch_embed.parameters():
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
    trainable_params = []
    trainable_params += list(latent_mixer.parameters())
    trainable_params += [p for p in base_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
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

    decode_signed = None
    if rank == 0:
        with torch.no_grad():
            probe = torch.randn(1, *latent_size, device=device)
            probe_img = rae.decode(probe).detach()
            decode_min = probe_img.min().item()
            decode_max = probe_img.max().item()
            decode_signed = decode_min < -0.1
            logger.info(
                f"[Decode Range] min={decode_min:.3f}, max={decode_max:.3f}, signed={decode_signed}"
            )

    def to_01(x: torch.Tensor) -> torch.Tensor:
        if decode_signed:
            x = (x + 1.0) * 0.5
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
                # ---- (A) Original diffusion loss ----
                loss_dict = transport.training_losses(
                    train_model_fn,
                    z,
                    {"y": train_labels, "edge_latent": edge_for_model},
                    return_intermediates=False,
                )
                loss_diff = loss_dict["loss"].mean()

                # ---- (C) Total loss ----
                loss = loss_diff
            
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
                        list(unwrap(ddp_model).mixer.parameters())
                        + [p for p in base_model.parameters() if p.requires_grad],
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

                        c = unwrap(ddp_model).mixer(e_chk)
                        c_absmean = c.abs().mean().item()
                        c_std = c.std().item()

                        logger.info(
                            f"[EDGE_SENS] step={global_step} | "
                            f"pred_diff={diff:.3e} | "
                            f"mixer_absmean={c_absmean:.3e} | mixer_std={c_std:.3e}"
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
                    # 1) Clean latent decode (z_gt decode)
                    # -----------------------
                    with torch.no_grad(), autocast(**autocast_kwargs):
                        z_gt = rae.encode(vis_images)
                    img_clean = to_01(rae.decode(z_gt).float()).cpu()

                    # -----------------------
                    # 2) Inference samples (same noise for fair compare)
                    # -----------------------
                    zs = torch.randn(n_vis, *latent_size, device=device, dtype=torch.float32)
                    s_edge = float(training_cfg.get("edge_cfg_scale", 3.0))

                    with torch.no_grad(), autocast(**autocast_kwargs):
                        def fn_edge_cfg(x, t, **kw):
                            v_cond = ema_model_fn(x, t, y=vis_labels, edge_latent=edge_vis_cond)
                            v_un = ema_model_fn(x, t, y=vis_labels, edge_latent=None)
                            return v_un + s_edge * (v_cond - v_un)

                        def fn_y_noedge(x, t, **kw):
                            return ema_model_fn(x, t, y=vis_labels, edge_latent=None)

                        lat_edgecfg = eval_sampler(
                            zs,
                            fn_edge_cfg,
                            y=vis_labels,
                            **sample_model_kwargs,
                        )[-1]
                        lat_noedge = eval_sampler(
                            zs,
                            fn_y_noedge,
                            y=vis_labels,
                            **sample_model_kwargs,
                        )[-1]

                    img_edgecfg = to_01(rae.decode(lat_edgecfg).float()).cpu()
                    img_noedge = to_01(rae.decode(lat_noedge).float()).cpu()

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
                        col3 = annotate(img_clean[i], "Clean latent(z)")
                        col4 = annotate(img_edgecfg[i], f"Infer: y+edge CFG | s={s_edge}")
                        col5 = annotate(img_noedge[i], "Infer: y (no edge)")

                        row = torch.cat([col1, col2, col3, col4, col5], dim=2)
                        rows.append(row)

                    final_grid = torch.cat(rows, dim=1)
                    save_path = os.path.join(
                        samples_dir,
                        f"step_{global_step:07d}_5col_GT_Edge_Clean_EdgeCFG_NoEdge.png",
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
