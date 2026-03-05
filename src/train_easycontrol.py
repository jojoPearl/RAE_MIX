"""
Single-file EasyControl training script (full encoder+decoder injection).

This file intentionally keeps adapter/model wrappers and training loop together
for easier iteration/debugging in one place.
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.utils import save_image

from stage1 import RAE
from stage2.models import Stage2ModelProtocol
from stage2.transport import Sampler, create_transport
from utils.model_utils import instantiate_from_config
from utils.resume_utils import configure_experiment_dirs, save_worktree
from utils.train_utils import center_crop_arr, get_autocast_scaler, parse_configs, prepare_dataloader


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# -----------------------------------------------------------------------------
# Generic Helpers
# -----------------------------------------------------------------------------

def load_stage2_weights(model: nn.Module, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "ema" in ckpt:
            sd = ckpt["ema"]
        elif "model" in ckpt:
            sd = ckpt["model"]
        else:
            sd = ckpt
    else:
        sd = ckpt

    clean_sd = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[7:]
        clean_sd[k] = v
    model.load_state_dict(clean_sd, strict=False)


def annotate_tensor(img01: torch.Tensor, text: str) -> torch.Tensor:
    img01 = img01.detach().clamp(0, 1)
    img_np = (img01 * 255).byte().permute(1, 2, 0).cpu().numpy()
    pil = Image.fromarray(img_np)
    draw = ImageDraw.Draw(pil)
    draw.rectangle([0, 0, pil.size[0], 18], fill=(255, 255, 255))
    draw.text((5, 2), text, fill=(0, 0, 0))
    return TF.to_tensor(pil)


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


def ensure_three_channels(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    return x


def save_adapter_checkpoint(
    path: str,
    step: int,
    epoch: int,
    adapter: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "adapter": adapter.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def revive_dead_qkv_paths(adapter: nn.Module, std: float = 1e-4) -> List[str]:
    """
    Some old checkpoints may contain fully-zero qkv control paths.
    Re-initialize tiny non-zero weights so gradients can flow again.
    """
    revived: List[str] = []
    if std <= 0.0:
        return revived
    with torch.no_grad():
        for i, blk in enumerate(getattr(adapter, "decoder_blocks", [])):
            if int(torch.count_nonzero(blk.qkv_out.weight).item()) == 0:
                nn.init.normal_(blk.qkv_out.weight, mean=0.0, std=std)
                revived.append(f"decoder_blocks.{i}.qkv_out.weight")
            if int(torch.count_nonzero(blk.qkv_lora.lora_B).item()) == 0:
                nn.init.normal_(blk.qkv_lora.lora_B, mean=0.0, std=std)
                revived.append(f"decoder_blocks.{i}.qkv_lora.lora_B")
    return revived


# -----------------------------------------------------------------------------
# EasyControl Adapter
# -----------------------------------------------------------------------------

class ConditionEncoder(nn.Module):
    """
    Hybrid condition encoder:
    - multi-layer CNN stem for local edge features
    - patch-grid projection for token alignment with DiT grid
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 768,
        stem_channels: int = 256,
        patch_size: int = 4,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, stem_channels, 3, stride=1, padding=1),
            nn.GroupNorm(16, stem_channels),
            nn.SiLU(),
            nn.Conv2d(stem_channels, stem_channels, 3, stride=1, padding=1),
            nn.GroupNorm(16, stem_channels),
            nn.SiLU(),
        )
        self.patch_proj = nn.Conv2d(
            stem_channels,
            out_channels,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.min().item() < -0.1:
            x = (x + 1.0) * 0.5
        x = x.clamp(0, 1)
        feat = self.stem(x)
        feat = self.patch_proj(feat)
        return feat.flatten(2).transpose(1, 2)


class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 16.0):
        super().__init__()
        self.scaling = alpha / float(rank)
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.lora_A @ self.lora_B) * self.scaling


class EncoderControlBlock(nn.Module):
    def __init__(self, hidden_dim: int, lora_rank: int = 16, lora_alpha: float = 16.0):
        super().__init__()
        self.lora = LoRALinear(hidden_dim, hidden_dim, lora_rank, lora_alpha)
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-6),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.lora(x))


class DecoderControlBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        qkv_scale: float = 0.05,
    ):
        super().__init__()
        self.qkv_scale = float(qkv_scale)

        self.pre_lora = LoRALinear(hidden_dim, hidden_dim, lora_rank, lora_alpha)
        self.pre_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-6),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.zeros_(self.pre_mlp[-1].weight)
        nn.init.zeros_(self.pre_mlp[-1].bias)

        self.qkv_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.qkv_lora = LoRALinear(hidden_dim, 3 * hidden_dim, lora_rank, lora_alpha)
        self.qkv_out = nn.Linear(3 * hidden_dim, 3 * hidden_dim)
        # Keep this branch near-zero but not exactly zero, otherwise gradients can die.
        nn.init.normal_(self.qkv_out.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.qkv_out.bias)
        nn.init.normal_(self.qkv_lora.lora_B, mean=0.0, std=1e-4)

        self.post_lora = LoRALinear(hidden_dim, hidden_dim, lora_rank, lora_alpha)
        self.post_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-6),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.zeros_(self.post_mlp[-1].weight)
        nn.init.zeros_(self.post_mlp[-1].bias)

        self.mlp_lora = LoRALinear(hidden_dim, hidden_dim, lora_rank, lora_alpha)
        self.mlp_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-6),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.zeros_(self.mlp_mlp[-1].weight)
        nn.init.zeros_(self.mlp_mlp[-1].bias)

    def forward(self, x: torch.Tensor):
        pre = self.pre_mlp(self.pre_lora(x))
        qkv = self.qkv_out(self.qkv_lora(self.qkv_norm(x))) * self.qkv_scale
        post = self.post_mlp(self.post_lora(x))
        mlp = self.mlp_mlp(self.mlp_lora(x))
        return pre, qkv, post, mlp


class EasyControlAdapterFullInjection(nn.Module):
    """
    Full-injection adapter:
    - Encoder post-residuals for all encoder blocks
    - Decoder pre/qkv/post/mlp residuals for all decoder blocks
    - Final residual before final layer
    """

    def __init__(
        self,
        num_encoder_blocks: int,
        num_decoder_blocks: int,
        encoder_hidden_dim: int,
        decoder_hidden_dim: int,
        condition_channels: int = 3,
        condition_latent_dim: int = 768,
        condition_patch_size: int = 4,
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        qkv_scale: float = 0.05,
        control_clamp: float = 5.0,
        final_residual_scale: float = 1.0,
        encoder_layer_decay: float = 0.98,
        decoder_layer_decay: float = 0.95,
        encoder_residual_norm: str = "layernorm",
        encoder_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.num_encoder_blocks = int(num_encoder_blocks)
        self.num_decoder_blocks = int(num_decoder_blocks)
        self.encoder_hidden_dim = int(encoder_hidden_dim)
        self.decoder_hidden_dim = int(decoder_hidden_dim)
        self.control_clamp = float(control_clamp)
        self.final_residual_scale = float(final_residual_scale)
        self.encoder_residual_norm = str(encoder_residual_norm).strip().lower()
        self.encoder_norm_eps = float(encoder_norm_eps)
        if self.encoder_residual_norm not in {"none", "layernorm", "rmsnorm"}:
            raise ValueError(
                f"Unsupported encoder_residual_norm={encoder_residual_norm}. "
                "Choose from: none, layernorm, rmsnorm."
            )

        self.condition_encoder = ConditionEncoder(
            in_channels=condition_channels,
            out_channels=condition_latent_dim,
            patch_size=condition_patch_size,
        )

        self.encoder_input_proj = nn.Linear(condition_latent_dim, encoder_hidden_dim)
        self.decoder_input_proj = nn.Linear(condition_latent_dim, decoder_hidden_dim)

        self.encoder_blocks = nn.ModuleList(
            [EncoderControlBlock(encoder_hidden_dim, lora_rank, lora_alpha) for _ in range(self.num_encoder_blocks)]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderControlBlock(
                    decoder_hidden_dim,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    qkv_scale=qkv_scale,
                )
                for _ in range(self.num_decoder_blocks)
            ]
        )

        self.final_lora = LoRALinear(decoder_hidden_dim, decoder_hidden_dim, lora_rank, lora_alpha)
        self.final_mlp = nn.Sequential(
            nn.LayerNorm(decoder_hidden_dim, eps=1e-6),
            nn.Linear(decoder_hidden_dim, decoder_hidden_dim),
            nn.SiLU(),
            nn.Linear(decoder_hidden_dim, decoder_hidden_dim),
        )
        nn.init.zeros_(self.final_mlp[-1].weight)
        nn.init.zeros_(self.final_mlp[-1].bias)

        enc_init = torch.pow(
            torch.tensor(float(encoder_layer_decay), dtype=torch.float32),
            torch.arange(self.num_encoder_blocks, dtype=torch.float32),
        ).view(-1, 1, 1)
        dec_init = torch.pow(
            torch.tensor(float(decoder_layer_decay), dtype=torch.float32),
            torch.arange(self.num_decoder_blocks, dtype=torch.float32),
        ).view(-1, 1, 1)
        self.encoder_layer_scales = nn.Parameter(enc_init)
        self.decoder_layer_scales = nn.Parameter(dec_init)

        self.num_control_tensors = self.num_encoder_blocks + 4 * self.num_decoder_blocks + 1

    def _clamp_residual(self, x: torch.Tensor) -> torch.Tensor:
        if self.control_clamp <= 0.0:
            return x
        return torch.clamp(x, -self.control_clamp, self.control_clamp)

    def _normalize_encoder_residual(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoder_residual_norm == "none":
            return x
        if self.encoder_residual_norm == "layernorm":
            return nn.functional.layer_norm(x, (x.shape[-1],), eps=self.encoder_norm_eps)
        # rmsnorm without affine params to keep adapter checkpoints backward-compatible.
        denom = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.encoder_norm_eps)
        return x * denom

    def encode_condition(self, condition_image: torch.Tensor) -> torch.Tensor:
        return self.condition_encoder(condition_image)

    @staticmethod
    def position_aware_interpolation(cond_tokens: torch.Tensor, target_length: int) -> torch.Tensor:
        b, l_cond, d = cond_tokens.shape
        if l_cond == target_length:
            return cond_tokens

        side_cond = int(math.sqrt(l_cond))
        side_target = int(math.sqrt(target_length))
        cond_2d = cond_tokens.transpose(1, 2).reshape(b, d, side_cond, side_cond)
        cond_2d = nn.functional.interpolate(
            cond_2d,
            size=(side_target, side_target),
            mode="bilinear",
            align_corners=False,
        )
        return cond_2d.flatten(2).transpose(1, 2)

    def generate_control_residuals(
        self,
        cond_tokens: torch.Tensor,
        control_scale: float = 1.0,
        encoder_target_len: Optional[int] = None,
        decoder_target_len: Optional[int] = None,
    ) -> Dict[str, object]:
        if encoder_target_len is None:
            encoder_target_len = cond_tokens.shape[1]
        if decoder_target_len is None:
            decoder_target_len = cond_tokens.shape[1]

        cond_enc = self.position_aware_interpolation(cond_tokens, int(encoder_target_len))
        cond_dec = self.position_aware_interpolation(cond_tokens, int(decoder_target_len))

        enc_feat = self.encoder_input_proj(cond_enc)
        dec_feat = self.decoder_input_proj(cond_dec)

        encoder_post: List[torch.Tensor] = []
        for i, blk in enumerate(self.encoder_blocks):
            scale = self.encoder_layer_scales[i] * control_scale
            enc_res = self._normalize_encoder_residual(blk(enc_feat))
            encoder_post.append(self._clamp_residual(enc_res * scale))

        decoder_pre: List[torch.Tensor] = []
        decoder_qkv: List[torch.Tensor] = []
        decoder_post: List[torch.Tensor] = []
        decoder_mlp: List[torch.Tensor] = []
        for i, blk in enumerate(self.decoder_blocks):
            scale = self.decoder_layer_scales[i] * control_scale
            pre, qkv, post, mlp = blk(dec_feat)
            decoder_pre.append(self._clamp_residual(pre * scale))
            decoder_qkv.append(self._clamp_residual(qkv * scale))
            decoder_post.append(self._clamp_residual(post * scale))
            decoder_mlp.append(self._clamp_residual(mlp * scale))

        final = self._clamp_residual(
            self.final_mlp(self.final_lora(dec_feat)) * control_scale * self.final_residual_scale
        )

        return {
            "encoder_post": encoder_post,
            "decoder_pre": decoder_pre,
            "decoder_qkv": decoder_qkv,
            "decoder_post": decoder_post,
            "decoder_mlp": decoder_mlp,
            "final": final,
        }


# -----------------------------------------------------------------------------
# Model Wrappers / Samplers
# -----------------------------------------------------------------------------

class EasyControlDiTWrapper(nn.Module):
    """
    Wrapper for full-injection adapter.
    Produces control payload for both encoder and decoder paths.
    """

    def __init__(
        self,
        base_model: Stage2ModelProtocol,
        adapter: EasyControlAdapterFullInjection,
        encoder_warmup_steps: int = 0,
    ):
        super().__init__()
        self.base = base_model
        self.adapter = adapter
        self.encoder_warmup_steps = max(0, int(encoder_warmup_steps))

        for p in self.base.parameters():
            p.requires_grad = False

    def _get_encoder_seq_length(self) -> int:
        if hasattr(self.base, "s_embedder") and hasattr(self.base.s_embedder, "num_patches"):
            return int(self.base.s_embedder.num_patches)
        return 256

    def _get_decoder_seq_length(self) -> int:
        if hasattr(self.base, "x_embedder") and hasattr(self.base.x_embedder, "num_patches"):
            return int(self.base.x_embedder.num_patches)
        return 256

    def _flatten_control_payload(self, payload: Dict[str, object]) -> List[torch.Tensor]:
        enc_post: List[torch.Tensor] = payload["encoder_post"]
        dec_pre: List[torch.Tensor] = payload["decoder_pre"]
        dec_qkv: List[torch.Tensor] = payload["decoder_qkv"]
        dec_post: List[torch.Tensor] = payload["decoder_post"]
        dec_mlp: List[torch.Tensor] = payload["decoder_mlp"]
        final: torch.Tensor = payload["final"]

        if len(enc_post) == 0:
            raise ValueError("encoder_post must contain at least one tensor")
        if len(dec_pre) == 0:
            raise ValueError("decoder_pre must contain at least one tensor")

        b, ls, d_enc = enc_post[0].shape
        zeros_enc = torch.zeros((b, ls, d_enc), device=enc_post[0].device, dtype=enc_post[0].dtype)
        zeros_enc_qkv = torch.zeros((b, ls, 3 * d_enc), device=enc_post[0].device, dtype=enc_post[0].dtype)

        b2, lx, d_dec = dec_pre[0].shape
        zeros_dec = torch.zeros((b2, lx, d_dec), device=dec_pre[0].device, dtype=dec_pre[0].dtype)
        zeros_dec_qkv = torch.zeros((b2, lx, 3 * d_dec), device=dec_pre[0].device, dtype=dec_pre[0].dtype)

        out: List[torch.Tensor] = []

        # Encoder blocks: pre/qkv/mlp slots are zero; post carries control.
        for i in range(int(self.base.num_encoder_blocks)):
            out.append(zeros_enc)
            out.append(zeros_enc_qkv)
            out.append(zeros_enc)
            out.append(enc_post[i])

        # Decoder blocks: all four slots are populated.
        for j in range(int(self.base.num_decoder_blocks)):
            out.append(dec_pre[j])
            out.append(dec_qkv[j])
            out.append(dec_mlp[j])
            out.append(dec_post[j])

        out.append(final)
        return out

    def _build_control_payload(
        self,
        condition_image: torch.Tensor,
        control_scale: float,
        global_step: Optional[int] = None,
    ):
        cond_tokens = self.adapter.encode_condition(condition_image)
        payload = self.adapter.generate_control_residuals(
            cond_tokens,
            control_scale=control_scale,
            encoder_target_len=self._get_encoder_seq_length(),
            decoder_target_len=self._get_decoder_seq_length(),
        )

        # Encoder warm-up: ramp encoder control from 0 -> 1 for first N steps.
        # Only apply during training when an integer global_step is provided.
        if self.encoder_warmup_steps > 0 and isinstance(global_step, int):
            encoder_multiplier = max(0.0, min(1.0, float(global_step) / float(self.encoder_warmup_steps)))
            if encoder_multiplier < 1.0:
                payload["encoder_post"] = [r * encoder_multiplier for r in payload["encoder_post"]]

        return self._flatten_control_payload(payload)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        condition_image: Optional[torch.Tensor] = None,
        control_scale: float = 1.0,
        **kwargs,
    ):
        global_step = kwargs.pop("global_step", None)
        if condition_image is None or float(control_scale) == 0.0:
            return self.base(x, t, y=y, control_residuals=None, **kwargs)

        control_payload = self._build_control_payload(
            condition_image,
            float(control_scale),
            global_step=global_step,
        )
        return self.base(x, t, y=y, control_residuals=control_payload, **kwargs)

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        cfg_scale: float,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        condition_image: Optional[torch.Tensor] = None,
        control_scale: float = 1.0,
        **kwargs,
    ):
        global_step = kwargs.pop("global_step", None)
        if condition_image is None or float(control_scale) == 0.0:
            return self.base.forward_with_cfg(
                x,
                t,
                y,
                cfg_scale,
                cfg_interval=cfg_interval,
                control_residuals=None,
                **kwargs,
            )

        batch = int(x.shape[0])
        if batch % 2 != 0:
            raise ValueError(f"CFG expects even batch size, got {batch}")
        half = batch // 2

        # Use only conditional images to build control, then force uncond control to zero.
        if int(condition_image.shape[0]) == batch:
            cond_image = condition_image[:half]
        elif int(condition_image.shape[0]) == half:
            cond_image = condition_image
        else:
            raise ValueError(
                f"condition_image batch {int(condition_image.shape[0])} must be either {half} (cond-only) or {batch} (cfg-paired)"
            )

        cond_payload = self._build_control_payload(
            cond_image,
            float(control_scale),
            global_step=global_step,
        )
        control_payload: List[torch.Tensor] = []
        for i, res in enumerate(cond_payload):
            if int(res.shape[0]) != half:
                raise ValueError(
                    f"control_payload[{i}] batch {int(res.shape[0])} must match conditional half-batch {half}"
                )
            control_payload.append(torch.cat([res, torch.zeros_like(res)], dim=0))

        return self.base.forward_with_cfg(
            x,
            t,
            y,
            cfg_scale,
            cfg_interval=cfg_interval,
            control_residuals=control_payload,
            **kwargs,
        )


class BaselineSampler:
    """
    Baseline sampler aligned with sample.py behavior.
    """

    def __init__(
        self,
        base_model: Stage2ModelProtocol,
        rae: RAE,
        eval_sampler,
        autocast_kwargs: Dict,
        misc: Dict,
        guidance_cfg: Dict,
        device: torch.device,
    ):
        self.base_model = base_model
        self.rae = rae
        self.eval_sampler = eval_sampler
        self.autocast_kwargs = autocast_kwargs
        self.misc = misc
        self.guidance_cfg = guidance_cfg
        self.device = device
        self.guid_model_cache = None

    def _guidance(self):
        scale = float(self.guidance_cfg.get("scale", 1.0))
        method = str(self.guidance_cfg.get("method", "cfg"))
        t_min = float(self.guidance_cfg.get("t_min", 0.0))
        t_max = float(self.guidance_cfg.get("t_max", 1.0))
        return scale, method, t_min, t_max

    def _get_guidance_model_forward(self):
        guid_model_cfg = self.guidance_cfg.get("guidance_model", None)
        if guid_model_cfg is None:
            raise ValueError("guidance_model must be provided for autoguidance.")
        if self.guid_model_cache is None:
            self.guid_model_cache = instantiate_from_config(guid_model_cfg).to(self.device)
            self.guid_model_cache.eval()
        return self.guid_model_cache.forward

    def sample(self, z: torch.Tensor, y_cond: torch.Tensor) -> torch.Tensor:
        bsz = int(y_cond.shape[0])
        z_base = torch.cat([z, z], dim=0)

        num_classes_cfg = int(self.misc.get("num_classes", 1000))
        y_null = torch.full((bsz,), num_classes_cfg, device=y_cond.device, dtype=y_cond.dtype)
        y_base = torch.cat([y_cond, y_null], dim=0)

        guidance_scale, guidance_method, t_min, t_max = self._guidance()

        if guidance_scale > 1.0:
            model_kwargs = {
                "y": y_base,
                "cfg_scale": guidance_scale,
                "cfg_interval": (t_min, t_max),
            }
            if guidance_method == "autoguidance":
                model_kwargs["additional_model_forward"] = self._get_guidance_model_forward()
                model_fwd = self.base_model.forward_with_autoguidance
            else:
                model_fwd = self.base_model.forward_with_cfg
        else:
            model_kwargs = {"y": y_base}
            model_fwd = self.base_model.forward

        with torch.no_grad(), autocast(**self.autocast_kwargs):
            lat = self.eval_sampler(z_base, model_fwd, **model_kwargs)[-1]
            lat, _ = lat.chunk(2, dim=0)
            return self.rae.decode(lat).detach().cpu()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("EasyControl training for DiTwDDTHead")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--data-path", type=Path, required=True)
    p.add_argument("--results-dir", type=str, default="ckpts/easycontrol")
    p.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    p.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="bf16")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--global-seed", type=int, default=None)
    p.add_argument("--print-args", action="store_true")
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.print_args:
        args_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
        print(json.dumps(args_dict, indent=2, sort_keys=True))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    full_cfg = OmegaConf.load(args.config)
    rae_cfg, model_cfg, transport_cfg, sampler_cfg, guidance_cfg, misc_cfg, training_cfg, _ = parse_configs(full_cfg)

    misc = to_dict(misc_cfg)
    training = to_dict(training_cfg)
    transport_cfg_dict = to_dict(transport_cfg)
    sampler_cfg_dict = to_dict(sampler_cfg)
    guidance_cfg_dict = to_dict(guidance_cfg)

    dit_checkpoint = cfg_get(training, "dit_checkpoint", None)
    if dit_checkpoint is None:
        raise ValueError("training.dit_checkpoint is required in config")

    max_steps = int(cfg_get(training, "max_steps", 50000))
    log_every = int(cfg_get(training, "log_every", 100))
    sample_every = int(cfg_get(training, "sample_every", 1000))
    save_every = int(cfg_get(training, "save_every", 5000))
    batch_size = int(cfg_get(training, "batch_size", 4))
    accum_steps = max(1, int(cfg_get(training, "accum_steps", 1)))
    num_workers = int(cfg_get(training, "num_workers", 8))
    label_drop_rate = float(cfg_get(training, "label_drop_rate", 0.1))
    edge_dropout = float(cfg_get(training, "edge_dropout", 0.1))

    # Full injection is stronger; prefer broad training exposure and moderate sampling scale.
    control_scale_train_max = float(
        cfg_get(training, "control_scale_train_max", cfg_get(training, "control_scale_train", 2.0))
    )
    control_scale_train_min = float(cfg_get(training, "control_scale_train_min", 0.0))
    if control_scale_train_min < 0.0:
        control_scale_train_min = 0.0
    if control_scale_train_min > control_scale_train_max:
        control_scale_train_min = control_scale_train_max
    randomize_control_scale_train = bool(cfg_get(training, "randomize_control_scale_train", True))
    control_scale_sample_raw = float(cfg_get(training, "control_scale_sample", control_scale_train_max))
    align_control_scale_sample_to_train_max = bool(cfg_get(training, "align_control_scale_sample_to_train_max", True))
    control_scale_sample = (
        control_scale_train_max if align_control_scale_sample_to_train_max else control_scale_sample_raw
    )
    control_scale_warmup_steps = int(cfg_get(training, "control_scale_warmup_steps", 5000))
    control_scale_warmup_factor = float(cfg_get(training, "control_scale_warmup_factor", 0.5))
    warmup_mode = str(cfg_get(training, "warmup_mode", "scale_only")).strip().lower()
    if warmup_mode not in {"scale_only", "encoder_only", "both", "none"}:
        raise ValueError(
            f"Invalid training.warmup_mode={warmup_mode}. "
            "Expected one of: scale_only, encoder_only, both, none."
        )
    encoder_warmup_steps_cfg = int(cfg_get(training, "encoder_warmup_steps", 2000))
    enable_control_scale_warmup = warmup_mode in {"scale_only", "both"}
    encoder_warmup_steps = encoder_warmup_steps_cfg if warmup_mode in {"encoder_only", "both"} else 0
    revive_dead_qkv_on_resume = bool(cfg_get(training, "revive_dead_qkv_on_resume", True))
    revive_dead_qkv_std = float(cfg_get(training, "revive_dead_qkv_std", 1e-4))

    cfg_scale = float(cfg_get(training, "cfg_scale", 1.5))
    cfg_interval = (
        float(cfg_get(training, "cfg_t_min", cfg_get(training, "t_min", 0.0))),
        float(cfg_get(training, "cfg_t_max", cfg_get(training, "t_max", 1.0))),
    )

    # Keep baseline sampling and easycontrol sampling on the same guidance setup.
    guidance_cfg_dict["scale"] = cfg_scale
    guidance_cfg_dict["method"] = str(guidance_cfg_dict.get("method", "cfg"))
    guidance_cfg_dict["t_min"] = cfg_interval[0]
    guidance_cfg_dict["t_max"] = cfg_interval[1]

    ilsvrc_class_index = cfg_get(training, "ilsvrc_class_index", None)

    seed = args.global_seed if args.global_seed is not None else int(cfg_get(training, "global_seed", 42))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    latent_size = tuple(int(d) for d in misc.get("latent_size", (768, 16, 16)))
    shift_dim = misc.get("time_dist_shift_dim", math.prod(latent_size))
    shift_base = misc.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(shift_dim / shift_base)

    enable_log_txt = os.environ.get("ENABLE_LOG_TXT", "0") == "1"
    save_worktree_flag = os.environ.get("SAVE_WORKTREE", "0") == "1"

    experiment_dir, checkpoint_dir, logger = configure_experiment_dirs(
        args, rank=0, enable_file_log=enable_log_txt
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    samples_dir = os.path.join(experiment_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    if ilsvrc_class_index:
        logger.info(f"Using class index mapping: {ilsvrc_class_index}")
    if align_control_scale_sample_to_train_max and abs(control_scale_sample_raw - control_scale_train_max) > 1e-6:
        logger.warning(
            "Aligning control_scale_sample to control_scale_train_max: "
            f"{control_scale_sample_raw:.3f} -> {control_scale_train_max:.3f}"
        )
    logger.info(
        f"Warmup mode={warmup_mode}, control_scale_warmup={'on' if enable_control_scale_warmup else 'off'}, "
        f"encoder_warmup_steps={encoder_warmup_steps}"
    )

    idx2name = None
    if ilsvrc_class_index and os.path.exists(ilsvrc_class_index):
        try:
            with open(ilsvrc_class_index, "r") as f:
                class_index = json.load(f)
            idx2name = {int(k): v[1] for k, v in class_index.items()}
        except Exception as exc:
            logger.warning(f"Failed to load class names: {exc}")

    if save_worktree_flag and os.path.isdir(experiment_dir):
        save_worktree(experiment_dir, full_cfg)

    logger.info("Loading RAE...")
    rae: RAE = instantiate_from_config(rae_cfg).to(device)
    rae.eval()
    for p in rae.parameters():
        p.requires_grad = False

    logger.info("Loading base model...")
    base_model: Stage2ModelProtocol = instantiate_from_config(model_cfg).to(device)
    load_stage2_weights(base_model, dit_checkpoint)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    logger.info("Building full-injection adapter...")
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
        encoder_residual_norm=str(cfg_get(training, "encoder_residual_norm", "layernorm")),
        encoder_norm_eps=float(cfg_get(training, "encoder_norm_eps", 1e-6)),
    ).to(device)

    adapter_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    logger.info(f"Adapter params: {adapter_params / 1e6:.2f}M")
    logger.info(
        f"Encoder residual norm={adapter.encoder_residual_norm}, encoder_norm_eps={adapter.encoder_norm_eps:g}"
    )

    model = EasyControlDiTWrapper(
        base_model,
        adapter,
        encoder_warmup_steps=encoder_warmup_steps,
    ).to(device)
    model.train()
    model.base.eval()

    if args.compile:
        try:
            torch._dynamo.config.suppress_errors = True
            model.forward = torch.compile(model.forward)
            logger.info("torch.compile enabled for model.forward")
        except Exception as exc:
            logger.warning(f"torch.compile failed: {exc}")

    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=float(cfg_get(training, "lr", 1e-4)),
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )

    scaler, autocast_kwargs = get_autocast_scaler(args)

    stage2_transform = transforms.Compose(
        [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
            transforms.ToTensor(),
        ]
    )

    loader, _ = prepare_dataloader(
        args.data_path,
        batch_size,
        num_workers,
        rank=0,
        world_size=1,
        transform=stage2_transform,
        image_size=args.image_size,
        return_edges=True,
        ilsvrc_class_index_path=ilsvrc_class_index,
    )
    dataset_num_classes = len(getattr(loader.dataset, "classes", []))
    expected_num_classes = int(misc.get("num_classes", 1000))
    if dataset_num_classes > 0:
        logger.info(
            f"Dataset classes={dataset_num_classes}, model classes={expected_num_classes}, "
            f"class mapping={'on' if ilsvrc_class_index else 'off'}"
        )
    if (not ilsvrc_class_index) and dataset_num_classes > 0 and dataset_num_classes != expected_num_classes:
        logger.warning(
            "ilsvrc_class_index is not set while dataset class count differs from model num_classes. "
            "ImageFolder labels will be re-indexed and class conditioning will mismatch."
        )

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

    null_label = int(misc.get("null_label", int(misc.get("num_classes", 1000))))

    baseline_sampler = BaselineSampler(
        base_model=base_model,
        rae=rae,
        eval_sampler=eval_sampler,
        autocast_kwargs=autocast_kwargs,
        misc=misc,
        guidance_cfg=guidance_cfg_dict,
        device=device,
    )

    def print_control_stats(
        condition_image: torch.Tensor,
        control_scale: float,
        prefix: str,
        global_step: Optional[int] = None,
    ):
        with torch.no_grad():
            residuals = model._build_control_payload(
                condition_image,
                control_scale,
                global_step=global_step,
            )
            n_enc = int(model.base.num_encoder_blocks)
            n_dec = int(model.base.num_decoder_blocks)
            enc_post_idx = [4 * i + 3 for i in range(n_enc)]
            dec_qkv_idx = [4 * (n_enc + i) + 1 for i in range(n_dec)]
            enc_post = [residuals[i] for i in enc_post_idx]
            dec_qkv = [residuals[i] for i in dec_qkv_idx]
            stats = {
                "enc_post_mean": float(torch.stack([r.abs().mean() for r in enc_post]).mean().item()) if enc_post else 0.0,
                "dec_qkv_mean": float(torch.stack([r.abs().mean() for r in dec_qkv]).mean().item()) if dec_qkv else 0.0,
            }
        logger.info(
            f"{prefix} control_abs_mean: enc_post={stats['enc_post_mean']:.3e}, dec_qkv={stats['dec_qkv_mean']:.3e}"
        )

    def sample_and_save(step: int, images: torch.Tensor, labels: torch.Tensor, edges: torch.Tensor) -> None:
        model.eval()
        base_model.eval()
        rae.eval()

        n_vis = min(5, images.shape[0])
        gt = images[:n_vis].detach().cpu()

        edge_rgb = ensure_three_channels(edges[:n_vis])
        edge_rgb_cpu = edge_rgb.detach().cpu().clamp(0, 1)
        edge_cond = edge_rgb.to(device) * 2.0 - 1.0

        y_cond = labels[:n_vis].to(device)
        z = torch.randn(n_vis, *latent_size, device=device, dtype=torch.float32)

        img_base = baseline_sampler.sample(z, y_cond)

        with torch.no_grad(), autocast(**autocast_kwargs):
            z_cfg = torch.cat([z, z], dim=0)
            y_null = torch.full((n_vis,), null_label, device=device, dtype=y_cond.dtype)
            y_cfg = torch.cat([y_cond, y_null], dim=0)
            edge_null = torch.zeros_like(edge_cond)
            edge_cfg = torch.cat([edge_cond, edge_null], dim=0)

            lat_ctrl = eval_sampler(
                z_cfg,
                model.forward_with_cfg,
                y=y_cfg,
                cfg_scale=cfg_scale,
                cfg_interval=cfg_interval,
                condition_image=edge_cfg,
                control_scale=control_scale_sample,
                global_step=step,
            )[-1]
            lat_ctrl, _ = lat_ctrl.chunk(2, dim=0)
            img_ctrl = rae.decode(lat_ctrl).detach().cpu()

        rows = []
        for i in range(n_vis):
            # Skip text drawing for faster visualization.
            c1 = gt[i].clamp(0, 1)
            c2 = edge_rgb_cpu[i]
            c3 = img_ctrl[i].clamp(0, 1)
            c4 = img_base[i].clamp(0, 1)
            rows.append(torch.cat([c1, c2, c3, c4], dim=2))

        grid = torch.cat(rows, dim=1)
        out_path = os.path.join(samples_dir, f"step_{step:07d}.png")
        save_image(grid, out_path)
        logger.info(f"Saved samples: {out_path}")

        model.train()
        model.base.eval()

    def sample_two_canny_same_zy(step: int, labels: torch.Tensor, edges: torch.Tensor) -> None:
        model.eval()
        base_model.eval()
        rae.eval()

        y0 = labels[0:1].to(device)
        y_null = torch.full((1,), null_label, device=device, dtype=y0.dtype)
        y_cfg = torch.cat([y0, y_null], dim=0)
        z0 = torch.randn(1, *latent_size, device=device, dtype=torch.float32)
        z_cfg = torch.cat([z0, z0], dim=0)

        idx_a = 0
        idx_b = 1 if edges.shape[0] > 1 else 0

        edge_a = ensure_three_channels(edges[idx_a : idx_a + 1].to(device))
        edge_b = ensure_three_channels(edges[idx_b : idx_b + 1].to(device))

        edge_a_cond = edge_a * 2.0 - 1.0
        edge_b_cond = edge_b * 2.0 - 1.0

        ablation_control_scale = float(cfg_get(training, "ablation_control_scale", control_scale_sample))
        print_control_stats(edge_a_cond, ablation_control_scale, prefix="[two-canny A]", global_step=step)
        print_control_stats(edge_b_cond, ablation_control_scale, prefix="[two-canny B]", global_step=step)

        edge_a_vis = edge_a.detach().cpu().clamp(0, 1)
        edge_b_vis = edge_b.detach().cpu().clamp(0, 1)
        edge_a_cfg = torch.cat([edge_a_cond, torch.zeros_like(edge_a_cond)], dim=0)
        edge_b_cfg = torch.cat([edge_b_cond, torch.zeros_like(edge_b_cond)], dim=0)

        with torch.no_grad(), autocast(**autocast_kwargs):
            lat_a = eval_sampler(
                z_cfg,
                model.forward_with_cfg,
                y=y_cfg,
                cfg_scale=cfg_scale,
                cfg_interval=cfg_interval,
                condition_image=edge_a_cfg,
                control_scale=ablation_control_scale,
                global_step=step,
            )[-1]
            lat_a, _ = lat_a.chunk(2, dim=0)
            img_a = rae.decode(lat_a).detach().cpu().clamp(0, 1)

        with torch.no_grad(), autocast(**autocast_kwargs):
            lat_b = eval_sampler(
                z_cfg,
                model.forward_with_cfg,
                y=y_cfg,
                cfg_scale=cfg_scale,
                cfg_interval=cfg_interval,
                condition_image=edge_b_cfg,
                control_scale=ablation_control_scale,
                global_step=step,
            )[-1]
            lat_b, _ = lat_b.chunk(2, dim=0)
            img_b = rae.decode(lat_b).detach().cpu().clamp(0, 1)

        with torch.no_grad(), autocast(**autocast_kwargs):
            lat_base = eval_sampler(
                z_cfg,
                base_model.forward_with_cfg,
                y=y_cfg,
                cfg_scale=cfg_scale,
                cfg_interval=cfg_interval,
                control_residuals=None,
            )[-1]
            lat_base, _ = lat_base.chunk(2, dim=0)
            img_base = rae.decode(lat_base).detach().cpu().clamp(0, 1)

        row = torch.cat(
            [
                edge_a_vis[0],
                img_a[0],
                edge_b_vis[0],
                img_b[0],
                img_base[0],
            ],
            dim=2,
        )

        # out_path = os.path.join(samples_dir, f"ablation_two_canny_step_{step:07d}.png")
        # save_image(row, out_path)
        # logger.info(f"Saved two-canny test: {out_path}")

        model.train()
        model.base.eval()

    start_step = 0
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu")
        adapter.load_state_dict(ckpt["adapter"])
        if revive_dead_qkv_on_resume:
            revived_keys = revive_dead_qkv_paths(adapter, std=revive_dead_qkv_std)
            if revived_keys:
                logger.warning(
                    f"Revived {len(revived_keys)} dead qkv tensors after resume "
                    f"(std={revive_dead_qkv_std:g}); this checkpoint needs continued training."
                )
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0))
        start_epoch = int(ckpt.get("epoch", 0))

    logger.info("Starting EasyControl training")
    logger.info(
        f"max_steps={max_steps}, batch_size={batch_size}, accum_steps={accum_steps}, "
        f"effective_batch={batch_size * accum_steps}, control_scale_train=[{control_scale_train_min}, {control_scale_train_max}], "
        f"control_scale_sample={control_scale_sample}, cfg_scale={cfg_scale}, cfg_interval={cfg_interval}, "
        f"warmup_mode={warmup_mode}"
    )

    global_step = start_step
    running_loss = 0.0
    accum_counter = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, 10000):
        for images, labels, edges in loader:
            if global_step >= max_steps:
                break

            images = images.to(device)
            labels = labels.to(device)
            edges = edges.to(device)

            real_labels = labels
            null_labels = torch.full_like(real_labels, null_label)
            drop_mask = torch.rand(real_labels.shape[0], device=device) < label_drop_rate
            train_labels = torch.where(drop_mask, null_labels, real_labels)

            edge_rgb = ensure_three_channels(edges)
            edge_cond = edge_rgb * 2.0 - 1.0

            if edge_dropout > 0.0:
                keep_mask = (torch.rand(edge_cond.shape[0], device=device) >= edge_dropout).float().view(-1, 1, 1, 1)
                edge_cond_in = edge_cond * keep_mask
            else:
                edge_cond_in = edge_cond

            with torch.no_grad():
                z = rae.encode(images)

            if randomize_control_scale_train:
                control_scale_train = float(
                    torch.empty(1, device=device).uniform_(control_scale_train_min, control_scale_train_max).item()
                )
            else:
                control_scale_train = control_scale_train_max
            if enable_control_scale_warmup and control_scale_warmup_steps > 0 and global_step < control_scale_warmup_steps:
                control_scale_train = control_scale_train * control_scale_warmup_factor

            with autocast(**autocast_kwargs):
                loss = transport.training_losses(
                    model,
                    z,
                    {
                        "y": train_labels,
                        "condition_image": edge_cond_in,
                        "control_scale": control_scale_train,
                        "global_step": global_step,
                    },
                )["loss"].mean()

            loss_f = loss.float()
            loss_bwd = loss_f / float(accum_steps)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss_bwd).backward()
            else:
                loss_bwd.backward()

            accum_counter += 1
            if accum_counter >= accum_steps:
                if scaler is not None and scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accum_counter = 0

            running_loss += float(loss_f.item())
            global_step += 1

            if global_step % log_every == 0:
                avg_loss = running_loss / float(log_every)
                logger.info(f"[step {global_step}/{max_steps}] loss={avg_loss:.4f} control_scale_train={control_scale_train:.3f}")
                running_loss = 0.0

            if global_step % sample_every == 0:
                logger.info(f"[step {global_step}] sampling...")
                sample_and_save(global_step, images, labels, edges)
                # sample_two_canny_same_zy(global_step, labels, edges)
                pass

            if global_step % save_every == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"adapter_step-{global_step:07d}.pt")
                try:
                    save_adapter_checkpoint(ckpt_path, global_step, epoch, adapter, optimizer)
                    logger.info(f"Saved checkpoint: {ckpt_path}")
                except Exception as exc:
                    logger.warning(f"Checkpoint save failed at step {global_step}: {exc}")

        if global_step >= max_steps and accum_counter > 0:
            if scaler is not None and scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accum_counter = 0

        if global_step >= max_steps:
            break

    final_path = os.path.join(checkpoint_dir, f"adapter_final_step-{global_step:07d}.pt")
    try:
        save_adapter_checkpoint(final_path, global_step, epoch, adapter, optimizer)
        logger.info(f"Training complete. Final checkpoint: {final_path}")
    except Exception as exc:
        logger.warning(f"Training complete, but final checkpoint save failed: {exc}")


if __name__ == "__main__":
    main()
