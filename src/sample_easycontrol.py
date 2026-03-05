#!/usr/bin/env python3
"""
Standalone EasyControl sampling from a single canny image + class label.

This script does not modify training code or require dataset loading.
"""

import argparse
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.cuda.amp import autocast
import torchvision.transforms.functional as TF
from torchvision.utils import save_image

from stage1 import RAE
from stage2.models import Stage2ModelProtocol
from stage2.transport import Sampler, create_transport
from train_easycontrol import (
    EasyControlAdapterFullInjection,
    EasyControlDiTWrapper,
    load_stage2_weights,
)
from utils.model_utils import instantiate_from_config
from utils.train_utils import center_crop_arr, parse_configs


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def to_dict(cfg_section):
    if cfg_section is None:
        return {}
    return OmegaConf.to_container(cfg_section, resolve=True)


def cfg_get(d: Dict, key: str, default=None):
    if d is None:
        return default
    if key in d:
        return d[key]
    dashed = key.replace("_", "-")
    return d.get(dashed, default)


def get_autocast_kwargs(precision: str) -> Dict[str, Any]:
    if precision == "fp16":
        return {"enabled": True, "dtype": torch.float16}
    if precision == "bf16":
        return {"enabled": True, "dtype": torch.bfloat16}
    return {"enabled": False}


def load_adapter_weights(adapter: torch.nn.Module, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "ema_adapter" in ckpt:
            state_dict = ckpt["ema_adapter"]
        elif "adapter" in ckpt:
            state_dict = ckpt["adapter"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt
    adapter.load_state_dict(state_dict, strict=False)


def load_canny_rgb(path: Path, image_size: int) -> torch.Tensor:
    if path.is_dir():
        exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        image_files = [p for p in sorted(path.iterdir()) if p.is_file() and p.suffix.lower() in exts]
        if not image_files:
            raise FileNotFoundError(f"No image files found under directory: {path}")
        canny_candidates = [p for p in image_files if "canny" in p.name.lower()]
        path = canny_candidates[0] if canny_candidates else image_files[0]
        print(f"[INFO] --canny-path is directory, using: {path}")

    with Image.open(path) as img:
        img = img.convert("RGB")
        img = center_crop_arr(img, image_size)
    x = TF.to_tensor(img).float().clamp(0.0, 1.0)  # [3,H,W]
    x = x.mean(dim=0, keepdim=True).repeat(3, 1, 1)  # [3,H,W]
    return x


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("EasyControl sampling from canny + class")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--canny-path", type=Path, required=True)
    p.add_argument("--class-label", type=int, required=True)
    p.add_argument("--adapter-ckpt", type=str, default=None)
    p.add_argument("--outdir", type=Path, default=Path("easycontrol_canny_sample"))
    p.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    p.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="bf16")
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cfg-scale", type=float, default=None)
    p.add_argument("--control-scale", type=float, default=None)
    p.add_argument("--canny-noise-std", type=float, default=0.0)
    p.add_argument("--save-individual", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    autocast_kwargs = get_autocast_kwargs(args.precision)

    cfg = OmegaConf.load(args.config)
    rae_cfg, model_cfg, transport_cfg, sampler_cfg, guidance_cfg, misc_cfg, training_cfg, _ = parse_configs(cfg)
    misc = to_dict(misc_cfg)
    training = to_dict(training_cfg)
    transport_cfg_dict = to_dict(transport_cfg)
    sampler_cfg_dict = to_dict(sampler_cfg)
    guidance_cfg_dict = to_dict(guidance_cfg)

    dit_checkpoint = cfg_get(training, "dit_checkpoint", None)
    if dit_checkpoint is None:
        raise ValueError("training.dit_checkpoint is required in config")
    adapter_ckpt = (
        args.adapter_ckpt
        or cfg_get(training, "easycontrol_adapter_ckpt", None)
        or cfg_get(training, "adapter_ckpt", None)
    )
    if adapter_ckpt is None:
        raise ValueError("Provide --adapter-ckpt or set training.easycontrol_adapter_ckpt in config")

    num_classes = int(misc.get("num_classes", 1000))
    null_label = int(misc.get("null_label", num_classes))
    class_label = int(args.class_label)
    if class_label < 0 or class_label >= num_classes:
        raise ValueError(f"class-label={class_label} out of range [0, {num_classes - 1}]")

    latent_size = tuple(int(d) for d in misc.get("latent_size", (768, 16, 16)))
    shift_dim = misc.get("time_dist_shift_dim", math.prod(latent_size))
    shift_base = misc.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(shift_dim / shift_base)

    cfg_scale = (
        float(args.cfg_scale)
        if args.cfg_scale is not None
        else float(cfg_get(training, "cfg_scale", guidance_cfg_dict.get("scale", 1.5)))
    )
    cfg_interval: Tuple[float, float] = (
        float(cfg_get(training, "cfg_t_min", cfg_get(training, "t_min", guidance_cfg_dict.get("t_min", 0.0)))),
        float(cfg_get(training, "cfg_t_max", cfg_get(training, "t_max", guidance_cfg_dict.get("t_max", 1.0)))),
    )
    control_scale = (
        float(args.control_scale)
        if args.control_scale is not None
        else float(cfg_get(training, "control_scale_sample", 3.0))
    )

    print("[INFO] Loading RAE...")
    rae: RAE = instantiate_from_config(rae_cfg).to(device)
    rae.eval()
    for p in rae.parameters():
        p.requires_grad = False

    print("[INFO] Loading Stage-2 base model...")
    base_model: Stage2ModelProtocol = instantiate_from_config(model_cfg).to(device)
    load_stage2_weights(base_model, dit_checkpoint)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    print("[INFO] Building adapter...")
    adapter = EasyControlAdapterFullInjection(
        num_encoder_blocks=int(base_model.num_encoder_blocks),
        num_decoder_blocks=int(base_model.num_decoder_blocks),
        encoder_hidden_dim=int(base_model.encoder_hidden_size),
        decoder_hidden_dim=int(base_model.decoder_hidden_size),
        condition_channels=3,
        condition_latent_dim=int(cfg_get(training, "condition_latent_dim", 768)),
        condition_patch_size=int(cfg_get(training, "condition_patch_size", 4)),
        lora_rank=int(cfg_get(training, "lora_rank", 16)),
        lora_alpha=float(cfg_get(training, "lora_alpha", 16.0)),
        qkv_scale=float(cfg_get(training, "qkv_scale", 0.05)),
        control_clamp=float(cfg_get(training, "control_clamp", 5.0)),
        final_residual_scale=float(cfg_get(training, "final_residual_scale", 1.0)),
        encoder_layer_decay=float(cfg_get(training, "encoder_layer_decay", 0.98)),
        decoder_layer_decay=float(cfg_get(training, "decoder_layer_decay", 0.95)),
    ).to(device)
    load_adapter_weights(adapter, adapter_ckpt)
    adapter.eval()
    for p in adapter.parameters():
        p.requires_grad = False

    model_ctrl = EasyControlDiTWrapper(
        base_model,
        adapter,
        encoder_warmup_steps=int(cfg_get(training, "encoder_warmup_steps", 0)),
    ).to(device)
    model_ctrl.eval()

    transport_params = dict(transport_cfg_dict.get("params", {}))
    transport_params.pop("time_dist_shift", None)
    transport = create_transport(**transport_params, time_dist_shift=time_dist_shift)
    transport_sampler = Sampler(transport)

    sampler_mode = str(sampler_cfg_dict.get("mode", "ODE")).upper()
    sampler_params = dict(sampler_cfg_dict.get("params", {}))
    if sampler_mode == "ODE":
        eval_sampler = transport_sampler.sample_ode(**sampler_params)
    elif sampler_mode == "SDE":
        eval_sampler = transport_sampler.sample_sde(**sampler_params)
    else:
        raise ValueError(f"Invalid sampler mode: {sampler_mode}")

    canny_single = load_canny_rgb(args.canny_path, args.image_size)  # [3,H,W]
    n = max(1, int(args.num_samples))
    canny_batch = canny_single.unsqueeze(0).repeat(n, 1, 1, 1).to(device)
    if float(args.canny_noise_std) > 0.0:
        canny_batch = torch.clamp(
            canny_batch + torch.randn_like(canny_batch) * float(args.canny_noise_std),
            0.0,
            1.0,
        )

    y_cond = torch.full((n,), class_label, device=device, dtype=torch.long)
    y_null = torch.full((n,), null_label, device=device, dtype=torch.long)
    y_cfg = torch.cat([y_cond, y_null], dim=0)

    edge_cond = canny_batch * 2.0 - 1.0
    edge_cfg = torch.cat([edge_cond, torch.zeros_like(edge_cond)], dim=0)

    z = torch.randn(n, *latent_size, device=device, dtype=torch.float32)
    z_cfg = torch.cat([z, z], dim=0)

    print(
        f"[INFO] Sampling with class={class_label}, n={n}, "
        f"cfg_scale={cfg_scale:.3f}, control_scale={control_scale:.3f}, "
        f"canny_noise_std={float(args.canny_noise_std):.4f}"
    )
    with torch.no_grad(), autocast(**autocast_kwargs):
        lat = eval_sampler(
            z_cfg,
            model_ctrl.forward_with_cfg,
            y=y_cfg,
            cfg_scale=cfg_scale,
            cfg_interval=cfg_interval,
            condition_image=edge_cfg,
            control_scale=control_scale,
            global_step=0,
        )[-1]
        lat, _ = lat.chunk(2, dim=0)
        img = rae.decode(lat).detach().cpu().clamp(0.0, 1.0)

    canny_vis = canny_batch.detach().cpu().clamp(0.0, 1.0)
    rows = [torch.cat([canny_vis[i], img[i]], dim=2) for i in range(n)]
    grid = torch.cat(rows, dim=1)

    args.outdir.mkdir(parents=True, exist_ok=True)
    grid_path = args.outdir / "canny_class_grid.png"
    save_image(grid, grid_path)
    print(f"[INFO] Saved grid: {grid_path}")

    if args.save_individual:
        for i in range(n):
            save_image(canny_vis[i], args.outdir / f"{i:03d}_canny.png")
            save_image(img[i], args.outdir / f"{i:03d}_sample.png")
        print(f"[INFO] Saved individual images under {args.outdir}")


if __name__ == "__main__":
    main()
