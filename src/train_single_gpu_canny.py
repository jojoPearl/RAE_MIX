# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for SiT using PyTorch DDP.
"""
import argparse
import logging
import math
import os
from collections import defaultdict, OrderedDict
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
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
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
from pathlib import Path
import math
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

def gate_t(t: torch.Tensor, mode: str = "linear") -> torch.Tensor:
    if t.dim() == 0:
        t = t[None]
    if t.dim() > 1:
        t = t.view(-1)
    if mode == "linear":
        g = (1.0 - t)
    elif mode == "sqrt":
        g = torch.sqrt(torch.clamp(1.0 - t, min=0.0))
    elif mode in ("none", "constant", ""):
        g = torch.ones_like(t)
    else:
        raise ValueError(f"Unknown gate mode: {mode}")
    return g.view(-1, 1, 1, 1)


class LatentMixer(nn.Module):
    """
    Slightly deeper mixer: dx = F(c).
    Zero-init on last layer => initial no-op.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return self.net(c)


class XTCondWrapper(torch.nn.Module):
    """
    Inject condition into x_t at every model call:
    x_t <- x_t + alpha * g(t) * mixer(edge_latent)
    """
    def __init__(self, base_model: torch.nn.Module, mixer: torch.nn.Module, alpha: float, gate_mode: str):
        super().__init__()
        self.base = base_model
        self.mixer = mixer
        self.alpha = float(alpha)
        self.gate_mode = gate_mode

    def _inject(self, x: torch.Tensor, t: torch.Tensor, edge_latent: Optional[torch.Tensor]):
        if edge_latent is None or self.alpha == 0.0:
            return x
        if edge_latent.shape[0] != x.shape[0]:
            if x.shape[0] == 2 * edge_latent.shape[0]:
                edge_latent = torch.cat([edge_latent, edge_latent], dim=0)
            elif edge_latent.shape[0] > x.shape[0]:
                edge_latent = edge_latent[: x.shape[0]]
            else:
                raise ValueError(
                    f"edge_latent batch {edge_latent.shape[0]} does not match x batch {x.shape[0]}"
                )
        g = gate_t(t, self.gate_mode).to(dtype=x.dtype, device=x.device)
        dx = self.mixer(edge_latent.to(dtype=x.dtype))
        return x + (self.alpha * g) * dx

    def forward(self, x, t, y=None, edge_latent=None, **kw):
        x = self._inject(x, t, edge_latent)
        return self.base.forward(x, t, y=y, **kw)

    def forward_with_cfg(self, x, t, y, cfg_scale, cfg_interval=(0, 1), edge_latent=None, **kw):
        x = self._inject(x, t, edge_latent)
        return self.base.forward_with_cfg(x, t, y, cfg_scale, cfg_interval=cfg_interval, **kw)

    def forward_with_autoguidance(self, x, t, y, cfg_scale, additional_model_forward, cfg_interval=(0, 1), edge_latent=None, **kw):
        x = self._inject(x, t, edge_latent)
        return self.base.forward_with_autoguidance(
            x, t, y, cfg_scale, additional_model_forward, cfg_interval=cfg_interval, **kw
        )


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

    num_classes = int(misc.get("num_classes", 1000))
    null_label = int(misc.get("null_label", num_classes))
    latent_size = tuple(int(dim) for dim in misc.get("latent_size", (768, 16, 16)))
    shift_dim = misc.get("time_dist_shift_dim", math.prod(latent_size))
    shift_base = misc.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(shift_dim / shift_base)

    grad_accum_steps = int(training_cfg.get("grad_accum_steps", 1))
    if grad_accum_steps < 1:
        raise ValueError("Gradient accumulation steps must be >= 1.")
    clip_grad_val = training_cfg.get("clip_grad", 1.0)
    clip_grad = float(clip_grad_val) if clip_grad_val is not None else None
    if clip_grad is not None and clip_grad <= 0:
        clip_grad = None
    ema_decay = float(training_cfg.get("ema_decay", 0.9995))
    num_epochs = int(training_cfg.get("epochs", 1400))
    global_batch_size = training_cfg.get("global_batch_size", None) # optional global batch size for override
    if global_batch_size is not None:
        global_batch_size = int(global_batch_size)
        assert global_batch_size % world_size == 0, "global_batch_size must be divisible by world_size"
    else:
        batch_size = int(training_cfg.get("batch_size", 16))
        global_batch_size = batch_size * world_size * grad_accum_steps
    num_workers = int(training_cfg.get("num_workers", 4))
    log_interval = 0  # do not print per-step logs
    sample_every = int(training_cfg.get("sample_every", 2500)) 
    checkpoint_interval = int(training_cfg.get("checkpoint_interval", 4)) # ckpt interval is epoch based
    cfg_scale_override = training_cfg.get("cfg_scale", None)
    default_seed = int(training_cfg.get("global_seed", 0))
    
    if eval_config:
        """
        FID online evaluation setup
        """
        do_eval = True
        eval_interval = int(eval_config.get("eval_interval", 5000))
        eval_model = eval_config.get("eval_model", False) # by default eval ema. This decides whether to **additionally** eval the non-ema model.
        eval_data = eval_config.get("data_path", None)
        reference_npz_path = eval_config.get("reference_npz_path", None)
        assert eval_data, "eval.data_path must be specified to enable evaluation."
        assert reference_npz_path, "eval.reference_npz_path must be specified to enable evaluation."
    else:
        do_eval = False
    global_seed = args.global_seed if args.global_seed is not None else default_seed
    seed = global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    micro_batch_size = global_batch_size // (world_size * grad_accum_steps)
    use_fp16 = args.precision == "fp16"
    use_bf16 = args.precision == "bf16"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("Requested bf16 precision, but the current CUDA device does not support bfloat16.")
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

    def guidance_value(key: str, default: float) -> float:
        if key in guidance_cfg:
            return guidance_cfg[key]
        dashed_key = key.replace("_", "-")
        return guidance_cfg.get(dashed_key, default)

    t_min = float(guidance_value("t_min", 0.0))
    t_max = float(guidance_value("t_max", 1.0))
    
    experiment_dir, checkpoint_dir, logger = configure_experiment_dirs(args, rank)
    samples_dir = os.path.join(experiment_dir, "samples_ab")
    os.makedirs(samples_dir, exist_ok=True)

    def debug_shapes_and_mixer(logger, step, rank, images, edges, z, c, latent_mixer):
        if rank != 0 or step >= 10:
            return
        logger.info(f"[DBG] images: {tuple(images.shape)} | edges: {tuple(edges.shape)}")
        logger.info(f"[DBG] z: {tuple(z.shape)} | c: {tuple(c.shape)}")
        lm = latent_mixer.module if hasattr(latent_mixer, "module") else latent_mixer
        if hasattr(lm, "proj"):
            w_mean = lm.proj.weight.abs().mean().item()
        else:
            w_mean = lm.net[-1].weight.abs().mean().item()
        logger.info(f"[DBG] latent_mixer |W| mean: {w_mean:.6e}")
    
    #### Model init
    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval()
    base_model: Stage2ModelProtocol = instantiate_from_config(model_config).to(device) 

    # Load pretrained base model weights (before wrapping).
    pretrained_ckpt_path = "/home/bjia-25/workspace/papers/RAE/code/rae_project/RAE_MIX/models/DiTs/Dinov2/wReg_base/ImageNet256/DiTDH-XL/stage2_model.pt"
    if os.path.exists(pretrained_ckpt_path):
        if rank == 0:
            logger.info(f"正在加载预训练底座权重: {pretrained_ckpt_path}")
        ckpt = torch.load(pretrained_ckpt_path, map_location="cpu")
        if "ema" in ckpt:
            sd = ckpt["ema"]
            if rank == 0:
                logger.info("检测到 EMA 权重，已加载。")
        elif "model" in ckpt:
            sd = ckpt["model"]
            if rank == 0:
                logger.info("检测到 model 键，已加载普通权重。")
        else:
            sd = ckpt
            if rank == 0:
                logger.info("未检测到特定键，假设为直接的 state_dict。")
        new_sd = OrderedDict()
        for k, v in sd.items():
            if k.startswith("module."):
                k = k[7:]
            new_sd[k] = v
        missing, unexpected = base_model.load_state_dict(new_sd, strict=False)
        if rank == 0:
            if len(missing) > 0:
                logger.warning(f"缺少键 (部分可能是正常的): {missing[:5]} ...")
            if len(unexpected) > 0:
                logger.warning(f"多余键: {unexpected[:5]} ...")
            logger.info("预训练底座模型加载完成！")
    else:
        raise FileNotFoundError(f"找不到预训练权重文件: {pretrained_ckpt_path}")

    if args.compile:
        try:
            rae.encode = torch.compile(rae.encode)
        except:
            print('RAE ENCODE compile meets error, falling back to no compile')
        try:
            base_model.forward = torch.compile(base_model.forward)
        except:
            print('MODEL FORWARD compile meets error, falling back to no compile')
    else:
        pass
    latent_mixer = LatentMixer(channels=latent_size[0]).to(device)
    latent_mixer_alpha = float(training_cfg.get("latent_mixer_alpha", 0.05))
    latent_mixer_t_gate = str(training_cfg.get("latent_mixer_t_gate", "none")).lower()
    model = XTCondWrapper(base_model, latent_mixer, alpha=latent_mixer_alpha, gate_mode=latent_mixer_t_gate).to(device)
    ema_model = deepcopy(model).to(device)
    ema_model.requires_grad_(False)
    ema_model.eval()
    model.requires_grad_(True) # train stage2 model
    ddp_model = model
    # ddp_model = torch.compile(ddp_model) # fix shape compile, see if it works
    ddp_model.train()
    def unwrap(m):
        return m.module if hasattr(m, "module") else m
    # freeze all params
    for p in ddp_model.parameters():
        p.requires_grad = False
    # unfreeze latent mixer only
    for p in unwrap(ddp_model).mixer.parameters():
        p.requires_grad = True
    # no need to put RAE into DDP since it's frozen
    model_param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model Parameters: {model_param_count/1e6:.2f}M")

    # Edge latent null-centering to ensure "no-edge" is a true no-op condition.
    with torch.no_grad():
        edge_null_rgb = torch.zeros(1, 3, args.image_size, args.image_size, device=device)
        edge_null_latent = rae.encode(edge_null_rgb).detach()
    
    #### Opt, Schedl init
    training_cfg["fused"] = False
    lr = float(training_cfg.get("optimizer", {}).get("lr", 1e-3))
    optimizer = torch.optim.AdamW(
        unwrap(ddp_model).mixer.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    optim_msg = "Optimizer: AdamW (x_t mixer only) lr=2e-4 betas=(0.9,0.95) weight_decay=0.0"

    ### AMP init
    scaler, autocast_kwargs = get_autocast_scaler(args)
    
    
    ### Data init
    stage2_transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    loader, sampler = prepare_dataloader(
        args.data_path,
        micro_batch_size,
        num_workers,
        rank,
        world_size,
        transform=stage2_transform,
        return_edges=True,
    )
    if do_eval:
        eval_dataset = ImageFolder(
            str(eval_data),
            transform=transforms.Compose([
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
                transforms.ToTensor(),
            ])
        )
        logger.info(f"Evaluation dataset loaded from {eval_data}, containing {len(eval_dataset)} images.")
        
    loader_batches = len(loader)
    # if loader_batches % grad_accum_steps != 0:
    #     raise ValueError("Number of loader batches must be divisible by grad_accum_steps when drop_last=True.")
    steps_per_epoch = loader_batches // grad_accum_steps
    if steps_per_epoch <= 0:
        raise ValueError("Gradient accumulation configuration results in zero optimizer steps per epoch.")
    
    scheduler = None
    sched_msg = None
    if training_cfg.get("scheduler"):
        scheduler, sched_msg = build_scheduler(optimizer, steps_per_epoch, training_cfg)
    
    #### Transport init
    transport = create_transport(
        **transport_params,
        time_dist_shift=time_dist_shift,
    )
    transport_sampler = Sampler(transport)

    if sampler_mode == "ODE":
        eval_sampler = transport_sampler.sample_ode(**sampler_params)
    elif sampler_mode == "SDE":
        eval_sampler = transport_sampler.sample_sde(**sampler_params)
    else:
        raise NotImplementedError(f"Invalid sampling mode {sampler_mode}.")
    
    
    ### Guidance Init
    guid_model_forward = None
    if guidance_scale > 1.0 and guidance_method == "autoguidance":
        guidance_model_cfg = guidance_cfg.get("guidance_model")
        if guidance_model_cfg is None:
            raise ValueError("Please provide a guidance model config when using autoguidance.")
        guid_model: Stage2ModelProtocol = instantiate_from_config(guidance_model_cfg).to(device)
        guid_model.eval()
        guid_model_forward = guid_model.forward
            
    log_steps = 0
    start_time = time()
    use_guidance = guidance_scale > 1.0
    zs = torch.randn(micro_batch_size, *latent_size, device=device, dtype=torch.float32) # always use float for noise sampling
    n = micro_batch_size
    if use_guidance:
        zs = torch.cat([zs, zs], dim=0)
        y_null = torch.full((n,), null_label, device=device)
        ys = torch.cat([ys, y_null], dim=0)
        sample_model_kwargs = dict(
            cfg_scale=guidance_scale,
            cfg_interval=(t_min, t_max),
        )
        if guidance_method == "autoguidance":
            if guid_model_forward is None:
                raise RuntimeError("Guidance model forward is not initialized.")
            sample_model_kwargs["additional_model_forward"] = guid_model_forward
            ema_model_fn = ema_model.forward_with_autoguidance
            model_fn = model.forward_with_autoguidance
        else:
            ema_model_fn = ema_model.forward_with_cfg
            model_fn = model.forward_with_cfg
    else:
        sample_model_kwargs = dict()
        ema_model_fn = ema_model.forward
        model_fn = model.forward
    train_model_fn = unwrap(ddp_model)

    ### Resuming and checkpointing
    start_epoch = 0
    global_step = 0
    if args.ckpt is not None:
        logger.info(f"Loading pretrained Stage2 ckpt: {args.ckpt}")
        state = torch.load(args.ckpt, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            sd = state["model"]
        else:
            sd = state
        msg = unwrap(ddp_model).load_state_dict(sd, strict=False)
        logger.info(f"[CKPT] missing_keys={len(msg.missing_keys)} unexpected_keys={len(msg.unexpected_keys)}")
        logger.info(f"[CKPT] example missing: {msg.missing_keys[:20]}")
        logger.info(f"[CKPT] example unexpected: {msg.unexpected_keys[:20]}")
        if isinstance(state, dict) and "ema" in state:
            ema_model.load_state_dict(state["ema"], strict=False)
        else:
            ema_model.load_state_dict(sd, strict=False)
        logger.info("Loaded.")
    maybe_resume_ckpt_path = find_resume_checkpoint(experiment_dir)
    if maybe_resume_ckpt_path is not None:
        logger.info(f"Experiment resume checkpoint found at {maybe_resume_ckpt_path}, automatically resuming...")
        ckpt_path = Path(maybe_resume_ckpt_path)
        if ckpt_path.is_file():
            start_epoch, global_step = load_checkpoint(
                ckpt_path,
                ddp_model,
                ema_model,
                optimizer,
                scheduler,
            )
            logger.info(f"[Rank {rank}] Resumed from {ckpt_path} (epoch={start_epoch}, step={global_step}).")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    else:
        # starting from fresh, save worktree and configs
        if rank == 0:
            save_worktree(experiment_dir, full_cfg)
            logger.info(f"Saved training worktree and config to {experiment_dir}.")
    ### Logging experiment details
    if rank == 0:
        num_params = sum(p.numel() for p in rae.parameters())
        logger.info(f"Stage-1 RAE parameters: {num_params/1e6:.2f}M")
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Stage-2 Model parameters: {num_params/1e6:.2f}M")
        if clip_grad is not None:
            logger.info(f"Clipping gradients to max norm {clip_grad}.")
        else:
            logger.info("Not clipping gradients.")
        # print optim and schel
        logger.info(optim_msg)
        print(sched_msg if sched_msg else "No LR scheduler.")
        logger.info(f"Training for {num_epochs} epochs, batch size {micro_batch_size} per GPU.")
        logger.info(f"Dataset contains {len(loader.dataset)} samples, {steps_per_epoch} steps per epoch.")
        logger.info(f"Running with world size {world_size}, starting from epoch {start_epoch} to {num_epochs}.")

    # dist.barrier() 
    for epoch in range(start_epoch, num_epochs):
        model.train()
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        epoch_metrics: Dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(1, device=device))
        num_batches = 0
        optimizer.zero_grad(set_to_none=True)
        if checkpoint_interval > 0 and epoch % checkpoint_interval == 0  and rank == 0:
            logger.info(f"Saving checkpoint at epoch {epoch}...")
            ckpt_path = f"{checkpoint_dir}/ep-{epoch:07d}.pt" 
            save_checkpoint(
                ckpt_path,
                global_step,
                epoch,
                ddp_model,
                ema_model,
                optimizer,
                scheduler,
            )
        for step, (images, labels, edges) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            edge_rgb = edges.to(device)
            with torch.no_grad(): # TODO: wrap this in autocast?
                z = rae.encode(images)
                edge_latent = rae.encode(edge_rgb)
                edge_latent = edge_latent - edge_null_latent
            with autocast(**autocast_kwargs):
                loss = transport.training_losses(
                    train_model_fn,
                    z,
                    {"y": labels, "edge_latent": edge_latent},
                )["loss"].mean()
            loss.float()
            if scaler:
                scaler.scale(loss / grad_accum_steps).backward()
            else:
                (loss / grad_accum_steps).backward()
            if (step + 1) % grad_accum_steps == 0:
                if clip_grad:
                    if scaler:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(unwrap(ddp_model).mixer.parameters(), clip_grad)
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                update_ema(ema_model, unwrap(ddp_model), decay=ema_decay)
                global_step += 1
            epoch_metrics['loss'] += loss.detach()
            if (step + 1) % grad_accum_steps == 0:
                if (global_step > 0) and (global_step % sample_every == 0) and rank == 0:
                    model.eval()
                    logger.info("Generating EMA samples (A/B)...")

                    with torch.no_grad():
                        # ===== Fixed noise =====
                        zs_samples = zs[:8]  # (8, C, H, W)
                        edge_rgb_A = edge_rgb[:8]  # correct canny
                        edge_rgb_B = torch.zeros_like(edge_rgb_A)

                        labels_vis = labels[:8]

                        with autocast(**autocast_kwargs):
                            c_A = rae.encode(edge_rgb_A) - edge_null_latent
                            c_B = torch.zeros_like(c_A)
                            delta_c = (c_A - c_B).abs().mean().item()
                            delta_m = (latent_mixer(c_A) - latent_mixer(c_B)).abs().mean().item()
                            if zs_samples.shape[0] != c_A.shape[0]:
                                if zs_samples.shape[0] == 2 * c_A.shape[0]:
                                    c_A = torch.cat([c_A, c_A], dim=0)
                                    c_B = torch.cat([c_B, c_B], dim=0)
                                else:
                                    raise ValueError(
                                        f"edge_latent batch {c_A.shape[0]} does not match z batch {zs_samples.shape[0]}"
                                    )
                            t_dbg = torch.full((c_A.shape[0],), 0.5, device=c_A.device)
                            scale_dbg = latent_mixer_alpha * gate_t(t_dbg, latent_mixer_t_gate).to(dtype=zs_samples.dtype)
                            dx_A = latent_mixer(c_A.to(dtype=zs_samples.dtype))
                            dx_B = latent_mixer(c_B.to(dtype=zs_samples.dtype))
                            delta_xt = ((scale_dbg * dx_A) - (scale_dbg * dx_B)).abs().mean().item()
                            logger.info(
                                f"[DBG] |cA-cB|={delta_c:.6e} |m(cA)-m(cB)|={delta_m:.6e} |xtA-xtB|={delta_xt:.6e}"
                            )
                            def bind_edge(fn, edge_latent):
                                def _f(x, t, **kw):
                                    return fn(x, t, edge_latent=edge_latent, **kw)
                                return _f

                            fn_A = bind_edge(ema_model_fn, c_A)
                            fn_B = bind_edge(ema_model_fn, c_B)
                            t_test = torch.tensor([0.5], device=device)
                            outA = ema_model_fn(zs_samples[:1], t_test, y=labels_vis[:1], edge_latent=c_A[:1])
                            outB = ema_model_fn(zs_samples[:1], t_test, y=labels_vis[:1], edge_latent=c_B[:1])
                            logger.info(f"[DBG] direct call Δ = {(outA - outB).abs().mean().item():.6e}")

                            samples_A = eval_sampler(zs_samples, fn_A, y=labels_vis, **sample_model_kwargs)[-1]
                            samples_B = eval_sampler(zs_samples, fn_B, y=labels_vis, **sample_model_kwargs)[-1]

                        # Decode
                        samples_A = rae.decode(samples_A).cpu().float()
                        samples_B = rae.decode(samples_B).cpu().float()

                        # ImageNet de-normalization (no (x+1)/2).
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                        samples_A = samples_A * std + mean
                        samples_B = samples_B * std + mean
                        samples_A = samples_A.clamp(0, 1)
                        samples_B = samples_B.clamp(0, 1)

                        # Debug: reconstruct edge latent to verify RAE can encode edges.
                        rec_edge = rae.decode(c_A + edge_null_latent).cpu().float()
                        rec_edge = rec_edge * std + mean
                        rec_edge = rec_edge.clamp(0, 1)
                        save_image(
                            rec_edge,
                            os.path.join(samples_dir, f"step_{global_step:07d}_debug_edge_recon.png"),
                        )
                        logger.info("[DBG] Edge recon saved. Check if it still looks like edges.")

                        # ===== Grid: top A, bottom B =====
                        grid_A = make_grid(samples_A, nrow=4)
                        grid_B = make_grid(samples_B, nrow=4)
                        grid = torch.cat([grid_A, grid_B], dim=1)

                        save_path = os.path.join(samples_dir, f"step_{global_step:07d}_AB.png")
                        save_image(grid, save_path)

                        # Save canny edges too
                        save_image(
                            make_grid(edge_rgb_A.cpu(), nrow=4),
                            os.path.join(samples_dir, f"step_{global_step:07d}_edges.png"),
                        )

                        logger.info(f"[A/B] Saved to {save_path}")

                    model.train()
                if do_eval and (eval_interval > 0 and global_step % eval_interval == 0):
                    logger.info("Starting evaluation...")
                    model.eval()
                    eval_models = [(ema_model_fn, "ema")]
                    if eval_model:
                        eval_models.append((model_fn, "model"))
                    for fn, mod_name in eval_models:
                        eval_stats = evaluate_generation_distributed(
                            fn,
                            eval_sampler,
                            latent_size,
                            sample_model_kwargs,
                            use_guidance,
                            rae,
                            eval_dataset,
                            len(eval_dataset),
                            rank = rank,
                            world_size = world_size,
                            device = device,
                            batch_size = micro_batch_size,
                            experiment_dir = experiment_dir,
                            global_step = global_step,
                            autocast_kwargs = autocast_kwargs,
                            reference_npz_path = reference_npz_path
                        )
                        # log with prefix
                        eval_stats = {f"eval_{mod_name}/{k}": v for k, v in eval_stats.items()} if eval_stats is not None else {}
                        if args.wandb:
                            wandb_utils.log(eval_stats, step=global_step)
                        model.train()
                    logger.info("Evaluation done.")
                if training_cfg.get("max_steps") is not None and global_step >= int(training_cfg["max_steps"]):
                    logger.info(f"Reached max_steps={training_cfg['max_steps']}, stopping.")
                    return
            num_batches += 1
        if rank == 0 and num_batches > 0:
            avg_loss = epoch_metrics['loss'].item() / num_batches 
            epoch_stats = {
                "epoch/loss": avg_loss,
            }
            logger.info(
                f"[Epoch {epoch}] "
                + ", ".join(f"{k}: {v:.4f}" for k, v in epoch_stats.items())
            )
            if args.wandb:
                wandb_utils.log(epoch_stats, step=global_step)
    # save the final ckpt
    if rank == 0:
        logger.info(f"Saving final checkpoint at epoch {num_epochs}...")
        ckpt_path = f"{checkpoint_dir}/ep-last.pt" 
        save_checkpoint(
            ckpt_path,
            global_step,
            num_epochs,
            ddp_model,
            ema_model,
            optimizer,
            scheduler,
        )
    # dist.barrier()
    logger.info("Done!")
    cleanup_distributed()



if __name__ == "__main__":
    main()
