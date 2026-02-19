import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import kornia
import torch
from omegaconf import OmegaConf
from torch.cuda.amp import autocast
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from tqdm import tqdm

from stage1 import RAE
from stage2.models import Stage2ModelProtocol
from stage2.transport import Sampler, create_transport
from train_easycontrol import EasyControlAdapterFullInjection, EasyControlDiTWrapper
from utils.model_utils import instantiate_from_config
from utils.train_utils import center_crop_arr, get_autocast_scaler, parse_configs


def load_stage2_weights(model: torch.nn.Module, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "ema" in ckpt:
            state_dict = ckpt["ema"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    clean_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        clean_state_dict[key] = value
    model.load_state_dict(clean_state_dict, strict=False)


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


def load_synset_to_ilsvrc(json_path: str) -> Dict[str, int]:
    with open(json_path, "r") as handle:
        obj = json.load(handle)
    mapping = {}
    for key, value in obj.items():
        if isinstance(value, (list, tuple)) and len(value) >= 1:
            mapping[value[0]] = int(key)
    return mapping


class MappedImageFolder(ImageFolder):
    def __init__(self, root: str, transform, synset_to_ilsvrc: Optional[Dict[str, int]] = None):
        super().__init__(root=root, transform=transform)
        self.synset_to_ilsvrc = synset_to_ilsvrc

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        if self.synset_to_ilsvrc is not None:
            synset = self.classes[target]
            if synset not in self.synset_to_ilsvrc:
                raise KeyError(f"Synset '{synset}' missing in class-index mapping.")
            target = self.synset_to_ilsvrc[synset]
        return image, target


def build_eval_dataloader(
    data_path: Path,
    batch_size: int,
    num_workers: int,
    transform,
    ilsvrc_class_index_path: Optional[str] = None,
) -> DataLoader:
    mapping = None
    if ilsvrc_class_index_path:
        if not os.path.exists(ilsvrc_class_index_path):
            raise FileNotFoundError(f"Class-index mapping file not found: {ilsvrc_class_index_path}")
        mapping = load_synset_to_ilsvrc(ilsvrc_class_index_path)
    dataset = MappedImageFolder(str(data_path), transform=transform, synset_to_ilsvrc=mapping)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader


def compute_canny01(img01: torch.Tensor, low: float = 0.1, high: float = 0.2) -> torch.Tensor:
    x = img01.clamp(0, 1)
    if x.shape[1] == 3:
        gray = kornia.color.rgb_to_grayscale(x)
    else:
        gray = x
    edges, _ = kornia.filters.canny(gray, low_threshold=low, high_threshold=high)
    return edges.clamp(0, 1)


@torch.no_grad()
def f1_from_edges(pred_edge01: torch.Tensor, gt_edge01: torch.Tensor, bin_thresh: float = 0.5, eps: float = 1e-8) -> torch.Tensor:
    pred = (pred_edge01 > bin_thresh).float()
    gt = (gt_edge01 > bin_thresh).float()
    tp = (pred * gt).sum(dim=(1, 2, 3))
    fp = (pred * (1.0 - gt)).sum(dim=(1, 2, 3))
    fn = ((1.0 - pred) * gt).sum(dim=(1, 2, 3))
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return 2.0 * precision * recall / (precision + recall + eps)


def sample_labels(num: int, num_classes: int, mode: str, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    if mode == "equal":
        base = num // num_classes
        labels = torch.arange(num_classes).repeat_interleave(base)
        if labels.numel() < num:
            extra = torch.randint(0, num_classes, (num - labels.numel(),), generator=generator)
            labels = torch.cat([labels, extra], dim=0)
        return labels[:num]

    if mode == "random":
        return torch.randint(0, num_classes, (num,), generator=generator)

    raise ValueError(f"Invalid label sampling mode: {mode}")


def build_sampler_from_config(transport_cfg_dict: Dict, sampler_cfg_dict: Dict, time_dist_shift: float):
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
    return eval_sampler


def make_cfg_batch(z: torch.Tensor, y: torch.Tensor, null_label: int) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = y.shape[0]
    z_cfg = torch.cat([z, z], dim=0)
    y_null = torch.full((batch_size,), null_label, device=y.device, dtype=y.dtype)
    y_cfg = torch.cat([y, y_null], dim=0)
    return z_cfg, y_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Eval EasyControl vs No-Control (FID + Canny-F1)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="bf16")

    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--label-sampling", type=str, choices=["dataset", "equal", "random"], default="dataset")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--empty-cache-every", type=int, default=20)

    parser.add_argument("--adapter-ckpt", type=str, default=None)
    parser.add_argument("--ilsvrc-class-index", type=str, default=None)
    parser.add_argument("--control-scale", type=float, default=2.0)

    parser.add_argument("--canny-low", type=float, default=0.1)
    parser.add_argument("--canny-high", type=float, default=0.2)
    parser.add_argument("--f1-bin-thresh", type=float, default=0.5)

    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--outdir", type=str, default="eval_out")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation script")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    os.makedirs(args.outdir, exist_ok=True)
    if args.save_images:
        os.makedirs(os.path.join(args.outdir, "baseline_png"), exist_ok=True)
        os.makedirs(os.path.join(args.outdir, "control_png"), exist_ok=True)

    full_cfg = OmegaConf.load(args.config)
    rae_cfg, model_cfg, transport_cfg, sampler_cfg, guidance_cfg, misc_cfg, training_cfg, _ = parse_configs(full_cfg)

    misc = to_dict(misc_cfg)
    training = to_dict(training_cfg)
    transport_cfg_dict = to_dict(transport_cfg)
    sampler_cfg_dict = to_dict(sampler_cfg)
    guidance_cfg_dict = to_dict(guidance_cfg)

    dit_checkpoint = cfg_get(training, "dit_checkpoint", None)
    if dit_checkpoint is None:
        raise ValueError("training.dit_checkpoint must be set in config")

    adapter_ckpt = args.adapter_ckpt or cfg_get(training, "easycontrol_adapter_ckpt", None)
    if adapter_ckpt is None:
        raise ValueError("Provide adapter checkpoint via --adapter-ckpt or training.easycontrol_adapter_ckpt")

    latent_size = tuple(int(d) for d in misc.get("latent_size", (768, 16, 16)))
    shift_dim = misc.get("time_dist_shift_dim", math.prod(latent_size))
    shift_base = misc.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(shift_dim / shift_base)

    num_classes = int(misc.get("num_classes", 1000))
    null_label = int(misc.get("null_label", num_classes))
    ilsvrc_class_index = (
        args.ilsvrc_class_index
        or cfg_get(training, "ilsvrc_class_index", None)
        or cfg_get(training, "ilsvrc_class_index_path", None)
    )
    if args.label_sampling == "dataset" and not ilsvrc_class_index:
        raise ValueError(
            "label-sampling=dataset requires class-index mapping. "
            "Set --ilsvrc-class-index or training.ilsvrc_class_index."
        )

    _, autocast_kwargs = get_autocast_scaler(args)

    stage2_transform = transforms.Compose(
        [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
            transforms.ToTensor(),
        ]
    )
    loader = build_eval_dataloader(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transform=stage2_transform,
        ilsvrc_class_index_path=ilsvrc_class_index,
    )
    dataset_num_classes = len(getattr(loader.dataset, "classes", []))
    print(
        f"Data loader ready: classes={dataset_num_classes}, "
        f"model_num_classes={num_classes}, class_mapping={'on' if ilsvrc_class_index else 'off'}"
    )

    rae: RAE = instantiate_from_config(rae_cfg).to(device)
    rae.eval()
    for param in rae.parameters():
        param.requires_grad = False

    base_model: Stage2ModelProtocol = instantiate_from_config(model_cfg).to(device)
    load_stage2_weights(base_model, dit_checkpoint)
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad = False

    eval_sampler = build_sampler_from_config(transport_cfg_dict, sampler_cfg_dict, time_dist_shift=time_dist_shift)

    cfg_scale = float(cfg_get(training, "cfg_scale", guidance_cfg_dict.get("scale", 1.0)))
    cfg_interval = (
        float(cfg_get(training, "cfg_t_min", guidance_cfg_dict.get("t_min", 0.0))),
        float(cfg_get(training, "cfg_t_max", guidance_cfg_dict.get("t_max", 1.0))),
    )

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
        encoder_layer_decay=float(cfg_get(training, "encoder_layer_decay", 0.98)),
        decoder_layer_decay=float(cfg_get(training, "decoder_layer_decay", 0.95)),
    ).to(device)
    load_adapter_weights(adapter, adapter_ckpt)
    adapter.eval()
    for param in adapter.parameters():
        param.requires_grad = False

    model_ctrl = EasyControlDiTWrapper(
        base_model,
        adapter,
        encoder_warmup_steps=int(cfg_get(training, "encoder_warmup_steps", 0)),
    ).to(device)
    model_ctrl.eval()

    fid_baseline = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    fid_control = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    f1_sum_baseline = torch.zeros((), device=device)
    f1_sum_control = torch.zeros((), device=device)
    f1_count = torch.zeros((), device=device)

    labels_local = None
    if args.label_sampling != "dataset":
        labels_all = sample_labels(args.num_samples, num_classes=num_classes, mode=args.label_sampling, seed=args.seed)
        labels_local = labels_all.to(device)

    gt_iter = iter(loader)

    def next_gt_batch(n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        nonlocal gt_iter
        image_chunks = []
        label_chunks = []
        remaining = n
        while remaining > 0:
            try:
                batch = next(gt_iter)
            except StopIteration:
                gt_iter = iter(loader)
                batch = next(gt_iter)

            if isinstance(batch, (list, tuple)):
                img = batch[0]
                lbl = batch[1]
            else:
                img = batch
                lbl = torch.zeros((img.shape[0],), dtype=torch.long)

            take = min(remaining, int(img.shape[0]))
            image_chunks.append(img[:take])
            label_chunks.append(lbl[:take])
            remaining -= take

        images = torch.cat(image_chunks, dim=0).to(device)
        labels = torch.cat(label_chunks, dim=0).to(device=device, dtype=torch.long)
        return images, labels

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    with torch.no_grad():
        done = 0
        step_idx = 0
        pbar = tqdm(total=args.num_samples, desc="Sampling", dynamic_ncols=True, smoothing=0.1)
        while done < args.num_samples:
            bsz = min(args.batch_size, args.num_samples - done)
            gt_img, gt_labels = next_gt_batch(bsz)
            gt_img = gt_img.clamp(0, 1)
            if args.label_sampling == "dataset":
                labels = gt_labels
            else:
                labels = labels_local[done:done + bsz]
            if int(labels.min().item()) < 0 or int(labels.max().item()) >= num_classes:
                raise ValueError(
                    f"Label out of range: min={int(labels.min().item())}, max={int(labels.max().item())}, "
                    f"expected [0, {num_classes - 1}]"
                )
            canny_gt = compute_canny01(gt_img, low=args.canny_low, high=args.canny_high)
            canny_rgb = canny_gt.repeat(1, 3, 1, 1)
            condition = canny_rgb * 2.0 - 1.0

            z = torch.randn(bsz, *latent_size, device=device, dtype=torch.float32)
            z_cfg, y_cfg = make_cfg_batch(z, labels, null_label=null_label)

            with autocast(**autocast_kwargs):
                lat_b = eval_sampler(
                    z_cfg,
                    base_model.forward_with_cfg,
                    y=y_cfg,
                    cfg_scale=cfg_scale,
                    cfg_interval=cfg_interval,
                    control_residuals=None,
                )[-1]
                lat_b, _ = lat_b.chunk(2, dim=0)
                img_b = rae.decode(lat_b).clamp(0, 1)

            condition_cfg = torch.cat([condition, condition], dim=0)
            with autocast(**autocast_kwargs):
                lat_c = eval_sampler(
                    z_cfg,
                    model_ctrl.forward_with_cfg,
                    y=y_cfg,
                    cfg_scale=cfg_scale,
                    cfg_interval=cfg_interval,
                    condition_image=condition_cfg,
                    control_scale=float(args.control_scale),
                    global_step=0,
                )[-1]
                lat_c, _ = lat_c.chunk(2, dim=0)
                img_c = rae.decode(lat_c).clamp(0, 1)

            fid_baseline.update(gt_img, real=True)
            fid_control.update(gt_img, real=True)
            fid_baseline.update(img_b, real=False)
            fid_control.update(img_c, real=False)

            canny_b = compute_canny01(img_b, low=args.canny_low, high=args.canny_high)
            canny_c = compute_canny01(img_c, low=args.canny_low, high=args.canny_high)

            f1_sum_baseline += f1_from_edges(canny_b, canny_gt, bin_thresh=args.f1_bin_thresh).sum()
            f1_sum_control += f1_from_edges(canny_c, canny_gt, bin_thresh=args.f1_bin_thresh).sum()
            f1_count += torch.tensor(float(bsz), device=device)

            if args.save_images:
                for i in range(bsz):
                    idx = done + i
                    save_image(img_b[i].cpu(), os.path.join(args.outdir, "baseline_png", f"{idx:07d}.png"))
                    save_image(img_c[i].cpu(), os.path.join(args.outdir, "control_png", f"{idx:07d}.png"))
                    save_image(canny_rgb[i].cpu(), os.path.join(args.outdir, "control_png", f"{idx:07d}_canny.png"))

            done += bsz
            step_idx += 1
            pbar.update(bsz)

            del (
                lat_b,
                lat_c,
                img_b,
                img_c,
                canny_b,
                canny_c,
                canny_gt,
                canny_rgb,
                condition,
                condition_cfg,
                gt_img,
                gt_labels,
                z,
                z_cfg,
                y_cfg,
            )
            if args.empty_cache_every > 0 and (step_idx % args.empty_cache_every == 0):
                torch.cuda.empty_cache()
        pbar.close()

    fid_b = float(fid_baseline.compute().item())
    fid_c = float(fid_control.compute().item())

    f1_b_mean = float((f1_sum_baseline / (f1_count + 1e-8)).item())
    f1_c_mean = float((f1_sum_control / (f1_count + 1e-8)).item())

    print("=== Eval Summary ===")
    print(f"num_samples={args.num_samples} image_size={args.image_size} cfg_scale={cfg_scale} cfg_interval={cfg_interval}")
    print(f"label_sampling={args.label_sampling} class_mapping={'on' if ilsvrc_class_index else 'off'}")
    print(f"control_scale={args.control_scale:.2f}")
    print(f"canny_thresholds=({args.canny_low},{args.canny_high}) f1_bin_thresh={args.f1_bin_thresh}")
    print("--- Baseline ---")
    print(f"FID={fid_b:.4f}")
    print(f"CannyF1={f1_b_mean:.4f}")
    print("--- Control ---")
    print(f"FID={fid_c:.4f}")
    print(f"CannyF1={f1_c_mean:.4f}")


if __name__ == "__main__":
    main()
