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
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import make_grid, save_image
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

def unwrap_model(m: torch.nn.Module) -> torch.nn.Module:
    return m.module if hasattr(m, "module") else m

def save_checkpoint(
    path: str,
    step: int,
    epoch: int,
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
) -> None:
    state = {
        "step": step,
        "epoch": epoch,
        "model": unwrap_model(model).state_dict(),
        "ema": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
) -> Tuple[int, int]:
    checkpoint = torch.load(path, map_location="cpu")
    unwrap_model(model).load_state_dict(checkpoint["model"])
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
    rank = 0
    world_size = 1
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
    log_interval = int(training_cfg.get("log_interval", 100))
    sample_every = int(training_cfg.get("sample_every", 2500)) 
    checkpoint_interval = int(training_cfg.get("checkpoint_interval", 4)) # ckpt interval is epoch based
    cfg_scale_override = training_cfg.get("cfg_scale", None)
    default_seed = int(training_cfg.get("global_seed", 0))
    max_steps = training_cfg.get("max_steps", None)
    max_steps = int(max_steps) if max_steps is not None else None
    
    # if eval_config:
    #     """
    #     FID online evaluation setup
    #     """
    #     do_eval = True
    #     eval_interval = int(eval_config.get("eval_interval", 5000))
    #     eval_model = eval_config.get("eval_model", False) # by default eval ema. This decides whether to **additionally** eval the non-ema model.
    #     eval_data = eval_config.get("data_path", None)
    #     reference_npz_path = eval_config.get("reference_npz_path", None)
    #     assert eval_data, "eval.data_path must be specified to enable evaluation."
    #     assert reference_npz_path, "eval.reference_npz_path must be specified to enable evaluation."
    # else:
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
    
    #### Model init
    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval()
    model: Stage2ModelProtocol = instantiate_from_config(model_config).to(device) 
    if args.compile:
        try:
            rae.encode = torch.compile(rae.encode)
        except:
            print('RAE ENCODE compile meets error, falling back to no compile')
        try:
            model.forward = torch.compile(model.forward)
        except:
            print('MODEL FORWARD compile meets error, falling back to no compile')
    ema_model = deepcopy(model).to(device)
    ema_model.requires_grad_(False)
    ema_model.eval()
    model.requires_grad_(True) # train stage2 model
    ddp_model = model
    ddp_model.train()
    # no need to put RAE into DDP since it's frozen
    model_param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model Parameters: {model_param_count/1e6:.2f}M")
    
    #### Opt, Schedl init
    optimizer, optim_msg = build_optimizer([p for p in model.parameters() if p.requires_grad], training_cfg)

    ### AMP init
    scaler, autocast_kwargs = get_autocast_scaler(args)
    
    
    ### Data init
    stage2_transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    dataset = ImageFolder(
        str(args.data_path),
        transform=stage2_transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    sampler = None
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
    if loader_batches % grad_accum_steps != 0:
        raise ValueError("Number of loader batches must be divisible by grad_accum_steps when drop_last=True.")
    steps_per_epoch = loader_batches // grad_accum_steps
    if steps_per_epoch <= 0:
        raise ValueError("Gradient accumulation configuration results in zero optimizer steps per epoch.")
    
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
    running_loss = 0.0
    start_time = time()
    use_guidance = guidance_scale > 1.0
    zs = torch.randn(micro_batch_size, *latent_size, device=device, dtype=torch.float32) # always use float for noise sampling
    if use_guidance:
        zs = torch.cat([zs, zs], dim=0)
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

    ### Resuming and checkpointing
    start_epoch = 0
    global_step = 0
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

    eval_skip_warned = False
    stop_training = False
    for epoch in range(start_epoch, num_epochs):
        model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch_metrics: Dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(1, device=device))
        num_batches = 0
        optimizer.zero_grad()
        accum_counter = 0
        step_loss_accum = 0.0
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
        for step, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad(): # TODO: wrap this in autocast?
                z = rae.encode(images)
            optimizer.zero_grad(set_to_none=True)
            model_kwargs = dict(y=labels)
            with autocast(**autocast_kwargs):
                loss = transport.training_losses(ddp_model, z, model_kwargs)["loss"].mean()
            loss.float()
            if scaler:
                scaler.scale(loss / grad_accum_steps).backward()
            else:
                (loss / grad_accum_steps).backward()
            if clip_grad:
                if scaler:
                    scaler.unscale_(optimizer) 
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), clip_grad)
            if global_step % grad_accum_steps == 0:
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                update_ema(ema_model, unwrap_model(ddp_model), decay=ema_decay)
            running_loss += loss.item()
            epoch_metrics['loss'] += loss.detach()
            
            if log_interval > 0 and global_step % log_interval == 0 and rank == 0:
                avg_loss = running_loss / log_interval # flow loss often has large variance so we record avg loss
                steps = torch.tensor(log_interval, device=device)
                stats = {
                    "train/loss": avg_loss,
                    "train/lr": optimizer.param_groups[0]["lr"],
                }
                logger.info(
                    f"[Epoch {epoch} | Step {global_step}] "
                    + ", ".join(f"{k}: {v:.4f}" for k, v in stats.items())
                )
                if args.wandb:
                    wandb_utils.log(
                        stats,
                        step=global_step,
                    )
                running_loss = 0.0
            if global_step % sample_every == 0 and rank == 0:
                model.eval()
                logger.info("Generating EMA samples...")
                with torch.no_grad():
                    n_vis = min(8, images.shape[0])
                    x_vis = images[:n_vis].detach().cpu()

                    zs_samples = zs[:n_vis]
                    visual_sample_model_kwargs = deepcopy(sample_model_kwargs)
                    visual_sample_model_kwargs["y"] = labels[:n_vis]

                    with autocast(**autocast_kwargs):
                        latent_samples = eval_sampler(zs_samples, ema_model_fn, **visual_sample_model_kwargs)[-1]
                    latent_samples = latent_samples.float()
                    if use_guidance:
                        latent_samples, _ = latent_samples.chunk(2, dim=0)

                    x_gen = rae.decode(latent_samples).detach().cpu().float()

                    top = make_grid(x_vis, nrow=n_vis, normalize=True, value_range=(0, 1))
                    bot = make_grid(x_gen, nrow=n_vis, normalize=True, value_range=(0, 1))
                    grid = torch.cat([top, bot], dim=1)

                    os.makedirs(os.path.join(experiment_dir, "samples"), exist_ok=True)
                    out_path = os.path.join(experiment_dir, "samples", f"step_{global_step:08d}.png")
                    save_image(grid, out_path)

                    y_list = labels[:n_vis].detach().cpu().tolist()
                    with open(os.path.join(experiment_dir, "samples", f"step_{global_step:08d}_y.txt"), "w") as f:
                        f.write("y: " + " ".join(map(str, y_list)) + "\n")

                    if args.wandb:
                        wandb_utils.log_image(torch.cat([x_vis, x_gen], dim=0), global_step)

                logger.info(f"Saved sample grid to {out_path}.")
                model.train()
            if do_eval and world_size > 1 and (eval_interval > 0 and global_step % eval_interval == 0):
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
            elif do_eval and world_size == 1 and (eval_interval > 0 and global_step % eval_interval == 0):
                if not eval_skip_warned and rank == 0:
                    logger.warning("Skipping distributed eval on single GPU (world_size=1).")
                    eval_skip_warned = True
            global_step += 1
            num_batches += 1
            if max_steps is not None and global_step >= max_steps:
                if rank == 0:
                    logger.info(f"Reached max_steps={max_steps}, stopping training.")
                stop_training = True
                break
        if stop_training:
            break
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
        if stop_training:
            break
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
    logger.info("Done!")



if __name__ == "__main__":
    main()
