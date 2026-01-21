import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

import os
import sys
import json
import datetime
import argparse
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torchvision.utils import save_image
from transformers import CLIPModel, CLIPProcessor

# -------------------- Path setup --------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

# -------------------- Project imports --------------------
from src.semantic.segmentation import get_text_guided_coords, extract_semantic_mask_with_clipseg
from src.semantic.utils import cleanup_memory, load_and_transform, _calculate_dynamic_coords
from src.semantic.resize import apply_m1_scaling
from src.semantic.modelManager import ModelManager
from src.semantic.config import DEVICE, DTYPE_MODERN

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = DEVICE


# =========================================================================
# CLIP Scorer (optional rerank)
# =========================================================================
class CLIPScorerTF:
    def __init__(self, device: torch.device, model_id: str = "openai/clip-vit-base-patch32"):
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()
        self._text_cache: Dict[str, torch.Tensor] = {}

        self._mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def _encode_text_cached(self, texts: List[str]) -> torch.Tensor:
        out: List[Optional[torch.Tensor]] = []
        missing: List[str] = []
        missing_idx: List[int] = []

        for i, t in enumerate(texts):
            if t in self._text_cache:
                out.append(self._text_cache[t])
            else:
                out.append(None)
                missing.append(t)
                missing_idx.append(i)

        if missing:
            tok = self.processor(text=missing, images=None, return_tensors="pt", padding=True, truncation=True)
            tok = {k: v.to(self.device) for k, v in tok.items() if isinstance(v, torch.Tensor)}
            txt_feat = self.model.get_text_features(**tok)
            txt_feat = F.normalize(txt_feat, dim=-1)
            for j, t in enumerate(missing):
                emb = txt_feat[j:j + 1]
                self._text_cache[t] = emb
                out[missing_idx[j]] = emb

        return torch.cat([x for x in out if x is not None], dim=0)

    @torch.no_grad()
    def score(self, images_01: torch.Tensor, texts: List[str]) -> torch.Tensor:
        assert images_01.ndim == 4 and images_01.shape[1] == 3
        n = images_01.shape[0]
        if len(texts) == 1 and n > 1:
            texts = texts * n
        assert len(texts) == n

        img = images_01.to(self.device, dtype=torch.float32).clamp(0, 1)
        img = F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)
        img = (img - self._mean) / self._std

        img_feat = self.model.get_image_features(pixel_values=img)
        img_feat = F.normalize(img_feat, dim=-1)
        txt_feat = self._encode_text_cached(texts)
        return (img_feat * txt_feat).sum(dim=-1)


# =========================================================================
# Utils
# =========================================================================
def feather_mask(mask: torch.Tensor, iters: int = 2, k: int = 3) -> torch.Tensor:
    m = mask
    pad = k // 2
    for _ in range(iters):
        m = F.avg_pool2d(m, kernel_size=k, stride=1, padding=pad)
    return m.clamp(0, 1)


def bbox_from_mask(mask: torch.Tensor, thr: float = 0.2) -> Optional[Tuple[int, int, int, int]]:
    m = (mask[0, 0] > thr).detach().cpu().numpy()
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def clamp_coords(ts_h: int, te_h: int, ts_w: int, te_w: int, H: int, W: int) -> Tuple[int, int, int, int]:
    ts_h = max(0, min(int(ts_h), H))
    te_h = max(0, min(int(te_h), H))
    ts_w = max(0, min(int(ts_w), W))
    te_w = max(0, min(int(te_w), W))
    if te_h <= ts_h:
        te_h = min(H, ts_h + 1)
    if te_w <= ts_w:
        te_w = min(W, ts_w + 1)
    return ts_h, te_h, ts_w, te_w


def safe_patch_slice(t: torch.Tensor, ts_h: int, te_h: int, ts_w: int, te_w: int) -> torch.Tensor:
    return t[:, :, ts_h:te_h, ts_w:te_w]


def generate_candidate_coords(
    H: int, W: int, ph: int, pw: int,
    base_coords: Tuple[int, int, int, int],
    num: int = 12, jitter: float = 0.12
) -> List[Tuple[int, int, int, int]]:
    ts_h, _, ts_w, _ = base_coords
    coords: List[Tuple[int, int, int, int]] = []
    for _ in range(num):
        dh = int(np.random.uniform(-1.0, 1.0) * jitter * H)
        dw = int(np.random.uniform(-1.0, 1.0) * jitter * W)
        s_h = ts_h + dh
        s_w = ts_w + dw
        e_h = s_h + ph
        e_w = s_w + pw
        s_h, e_h, s_w, e_w = clamp_coords(s_h, e_h, s_w, e_w, H, W)
        coords.append((s_h, e_h, s_w, e_w))
    return coords


# =========================================================================
# Object extraction (pixel space -> tensor + soft mask)
# =========================================================================
def get_cropped_object_tensor(
    raw_image: Image.Image,
    target_text: str,
    target_size_for_encoder: int,
    background_mode: str = "mean",
    clipseg_threshold: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    W_orig, H_orig = raw_image.size

    mask_soft = extract_semantic_mask_with_clipseg(
        image=raw_image,
        target_text=target_text,
        feature_size=(H_orig, W_orig),
        threshold=clipseg_threshold
    ).cpu().squeeze().clamp(0, 1).numpy()

    binary = (mask_soft > clipseg_threshold).astype(np.uint8)
    rows, cols = np.any(binary, axis=1), np.any(binary, axis=0)

    if not np.any(rows) or not np.any(cols):
        obj_to_scale = raw_image
        mask_to_scale = Image.fromarray((np.ones((H_orig, W_orig)) * 255).astype(np.uint8), mode="L")
    else:
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        obj_to_scale = Image.fromarray(np.array(raw_image)[rmin:rmax + 1, cmin:cmax + 1])
        mask_to_scale = Image.fromarray((mask_soft[rmin:rmax + 1, cmin:cmax + 1] * 255.0).astype(np.uint8), mode="L")

    # encoder 输入：始终用 M1=1.0（高质量）
    final_obj_pil = apply_m1_scaling(obj_to_scale.convert("RGB"), 1.0, target_size_for_encoder, background_mode)
    final_mask_pil = apply_m1_scaling(mask_to_scale.convert("RGB"), 1.0, target_size_for_encoder, background_mode="black").convert("L")

    obj_t = T.ToTensor()(final_obj_pil).unsqueeze(0).to(device=device, dtype=DTYPE_MODERN)
    mask_t = T.ToTensor()(final_mask_pil).unsqueeze(0).to(device=device, dtype=torch.float32)
    return obj_t, mask_t[:, 0:1].clamp(0, 1)


# =========================================================================
# Fusion mode A: v2 (scale in latent canvas via affine_grid/grid_sample)
# =========================================================================
def scale_in_latent_canvas(
    feat: torch.Tensor,
    mask: torch.Tensor,
    scale_factor: float,
    padding_mode: str = "zeros",
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, C, H, W = feat.shape
    a = 1.0 / max(float(scale_factor), 1e-6)

    mask = mask.to(device=feat.device, dtype=feat.dtype)

    theta = torch.tensor(
        [[[a, 0.0, 0.0],
          [0.0, a, 0.0]]],
        device=feat.device,
        dtype=feat.dtype
    ).repeat(B, 1, 1)

    grid = F.affine_grid(theta, size=feat.size(), align_corners=False)
    feat_s = F.grid_sample(feat, grid, mode="bilinear", padding_mode=padding_mode, align_corners=False)

    grid_m = F.affine_grid(theta, size=mask.size(), align_corners=False)
    grid_m = grid_m.to(dtype=feat.dtype)
    mask_s = F.grid_sample(mask, grid_m, mode="bilinear", padding_mode=padding_mode, align_corners=False)
    return feat_s, mask_s.clamp(0, 1)


@torch.no_grad()
def semantic_fusion_v2(
    canvas_latent: torch.Tensor,
    object_latent: torch.Tensor,
    object_mask_img: torch.Tensor,
    raw_base_image: Image.Image,
    location_prompt: Optional[str],
    target_area: str,
    scale_factor: float,
    use_smart_placement: bool = True,
    mask_thr_bbox: float = 0.2,
    feather_iters: int = 2,
    feather_k: int = 3,
    overlap_mode: str = "allow",
    occupied_mask_latent: Optional[torch.Tensor] = None,
    forced_coords: Optional[Tuple[int, int, int, int]] = None,
    alpha: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, C, H, W = canvas_latent.shape

    mask_l = F.interpolate(object_mask_img, size=(H, W), mode="bilinear", align_corners=False).clamp(0, 1)
    obj_s, mask_s = scale_in_latent_canvas(object_latent, mask_l, scale_factor=float(scale_factor))
    mask_s = feather_mask(mask_s, iters=feather_iters, k=feather_k)

    bb = bbox_from_mask(mask_s, thr=mask_thr_bbox)
    if bb is None:
        return canvas_latent, torch.zeros((B, 1, H, W), device=canvas_latent.device, dtype=canvas_latent.dtype)

    y0, y1, x0, x1 = bb
    obj_p = obj_s[:, :, y0:y1 + 1, x0:x1 + 1]
    m_p = mask_s[:, :, y0:y1 + 1, x0:x1 + 1]
    ph, pw = obj_p.shape[-2:]

    if forced_coords is not None:
        ts_h, te_h, ts_w, te_w = forced_coords
    else:
        coords = None
        if use_smart_placement and location_prompt and raw_base_image is not None:
            avoid = occupied_mask_latent if occupied_mask_latent is not None else None
            coords = get_text_guided_coords(
                raw_base_image, location_prompt, ph, pw, (H, W),
                avoid_mask=avoid, avoid_strength=float(10.0),
                avoid_thr=float(0.05),
            )
        if coords is None:
            ts_h, te_h, ts_w, te_w = _calculate_dynamic_coords(H, W, ph, pw, target_area)
        else:
            ts_h, te_h, ts_w, te_w = coords

    ts_h, te_h, ts_w, te_w = clamp_coords(ts_h, te_h, ts_w, te_w, H, W)

    base_p = safe_patch_slice(canvas_latent, ts_h, te_h, ts_w, te_w)
    if base_p.shape[-2:] != obj_p.shape[-2:]:
        obj_p = F.interpolate(obj_p, size=base_p.shape[-2:], mode="bilinear", align_corners=False)
        m_p = F.interpolate(m_p, size=base_p.shape[-2:], mode="bilinear", align_corners=False)

    m_eff = (m_p * float(alpha)).clamp(0, 1)

    if overlap_mode != "allow" and occupied_mask_latent is not None:
        occ_p = safe_patch_slice(occupied_mask_latent, ts_h, te_h, ts_w, te_w).clamp(0, 1)
        if overlap_mode == "no_overwrite":
            m_eff = m_eff * (1.0 - (occ_p > 0.2).float())
        elif overlap_mode == "alpha":
            m_eff = m_eff * (1.0 - 0.5 * occ_p)

    new_canvas = canvas_latent.clone()
    new_canvas[:, :, ts_h:te_h, ts_w:te_w] = m_eff * obj_p + (1.0 - m_eff) * base_p

    union_mask = torch.zeros((B, 1, H, W), device=new_canvas.device, dtype=new_canvas.dtype)
    union_mask[:, :, ts_h:te_h, ts_w:te_w] = m_eff.clamp(0, 1)

    return new_canvas, union_mask


# =========================================================================
# Fusion mode B: m2 (latent crop -> resize -> paste)
# =========================================================================
@torch.no_grad()
def semantic_fusion_m2(
    canvas_latent: torch.Tensor,
    object_latent: torch.Tensor,
    object_mask_img: torch.Tensor,
    raw_base_image: Image.Image,
    location_prompt: Optional[str],
    target_area: str,
    scale_factor: float,
    use_smart_placement: bool = True,
    mask_thr_bbox: float = 0.2,
    overlap_mode: str = "allow",
    occupied_mask_latent: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
    base_unit_ratio: float = 0.6,
    min_size: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, C, H, W = canvas_latent.shape

    # mask to latent
    mask_lat = F.interpolate(object_mask_img, size=(H, W), mode="bilinear", align_corners=False).clamp(0, 1)
    bb = bbox_from_mask(mask_lat, thr=mask_thr_bbox)
    if bb is None:
        return canvas_latent, torch.zeros((B, 1, H, W), device=canvas_latent.device, dtype=canvas_latent.dtype)

    y0, y1, x0, x1 = bb
    obj_crop = object_latent[:, :, y0:y1 + 1, x0:x1 + 1]
    mask_crop = mask_lat[:, :, y0:y1 + 1, x0:x1 + 1]

    h_old, w_old = obj_crop.shape[-2:]
    base_unit = min(H, W) * float(base_unit_ratio)
    target_long_edge = int(base_unit * float(scale_factor))
    target_long_edge = max(min_size, min(target_long_edge, min(H, W) - 2))

    ratio = target_long_edge / max(h_old, w_old)
    h_new = max(1, int(h_old * ratio))
    w_new = max(1, int(w_old * ratio))

    obj_rs = F.interpolate(obj_crop, size=(h_new, w_new), mode="bicubic", align_corners=False)
    mask_rs = F.interpolate(mask_crop, size=(h_new, w_new), mode="bilinear", align_corners=False)

    # coords
    effective_smart = False
    if use_smart_placement and location_prompt and raw_base_image is not None:
        coords = get_text_guided_coords(raw_base_image, location_prompt, h_new, w_new, (H, W))
        if coords:
            ts_h, te_h, ts_w, te_w = coords
            effective_smart = True

    if not effective_smart:
        ts_h, te_h, ts_w, te_w = _calculate_dynamic_coords(H, W, h_new, w_new, target_area)

    ts_h, te_h, ts_w, te_w = clamp_coords(ts_h, te_h, ts_w, te_w, H, W)

    base_p = safe_patch_slice(canvas_latent, ts_h, te_h, ts_w, te_w)
    if base_p.shape[-2:] != obj_rs.shape[-2:]:
        obj_rs = F.interpolate(obj_rs, size=base_p.shape[-2:], mode="bicubic", align_corners=False)
        mask_rs = F.interpolate(mask_rs, size=base_p.shape[-2:], mode="bilinear", align_corners=False)

    m_eff = (mask_rs * float(alpha)).clamp(0, 1)

    if overlap_mode != "allow" and occupied_mask_latent is not None:
        occ_p = safe_patch_slice(occupied_mask_latent, ts_h, te_h, ts_w, te_w).clamp(0, 1)
        if overlap_mode == "no_overwrite":
            m_eff = m_eff * (1.0 - (occ_p > 0.2).float())
        elif overlap_mode == "alpha":
            m_eff = m_eff * (1.0 - 0.5 * occ_p)

    new_canvas = canvas_latent.clone()
    new_canvas[:, :, ts_h:te_h, ts_w:te_w] = m_eff * obj_rs + (1.0 - m_eff) * base_p

    union_mask = torch.zeros((B, 1, H, W), device=new_canvas.device, dtype=new_canvas.dtype)
    union_mask[:, :, ts_h:te_h, ts_w:te_w] = m_eff.clamp(0, 1)

    return new_canvas, union_mask


# =========================================================================
# Main pipeline
# =========================================================================
def stage1_run(
    base_image_path: str,
    objects_list: List[Dict[str, Any]],
    output_dir: str,
    fused_prefix: str,
    t_size: int = 448,
    global_scales: List[float] = [1.0],

    fusion_mode: str = "v2",            # v2 | m2
    rerank_clip: bool = True,           # only meaningful for v2
    overlap_mode: str = "no_overwrite", # allow | no_overwrite | alpha

    clip_model_id: str = "openai/clip-vit-base-patch32",
    default_lambda_overlap: float = 0.3,
):
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    # Load base
    base_img_tensor = load_and_transform(base_image_path, t_size).to(device=device)
    base_pil = Image.open(base_image_path).convert("RGB")

    # sort objects
    objects_sorted = sorted(objects_list, key=lambda d: d.get("z_order", 0))

    # models
    manager = ModelManager(device=device)
    rae = manager.load_rae()

    clip_scorer = None
    if rerank_clip:
        clip_scorer = CLIPScorerTF(device=device, model_id=clip_model_id)

    cleanup_memory()

    # Encode base once
    with torch.amp.autocast("cuda", dtype=DTYPE_MODERN):
        base_latent_init = rae.encode(base_img_tensor)

    # Pre-encode objects once
    encoded_objects = []
    for conf in objects_sorted:
        path = conf["path"]
        text = conf["text"]
        if not os.path.exists(path):
            print(f"[Warn] missing object image: {path}")
            continue

        r_pil = Image.open(path).convert("RGB")
        obj_t, obj_m = get_cropped_object_tensor(
            raw_image=r_pil,
            target_text=text,
            target_size_for_encoder=t_size,
            background_mode=str(conf.get("background_mode", "mean")),
            clipseg_threshold=float(conf.get("clipseg_threshold", 0.3)),
        )
        with torch.amp.autocast("cuda", dtype=DTYPE_MODERN):
            obj_l = rae.encode(obj_t)

        encoded_objects.append({"latent": obj_l, "mask_img": obj_m, "config": conf})

    def make_scale_pt(prefix: str, g: float) -> str:
        root, ext = os.path.splitext(prefix)
        if ext == "":
            ext = ".pt"
        return f"{root}_scale_{g}{ext}"

    # Run for each global scale
    for gscale in global_scales:
        canvas_latent = base_latent_init.clone()
        B, C, H, W = canvas_latent.shape

        occupied = torch.zeros((B, 1, H, W), device=device, dtype=canvas_latent.dtype)
        union_total = torch.zeros((B, 1, H, W), device=device, dtype=canvas_latent.dtype)

        for item in encoded_objects:
            conf = item["config"]

            base_scale = float(conf.get("base_scale", 1.0))
            final_sc = float(gscale * base_scale)

            location_prompt = conf.get("location_prompt", None)
            target_area = conf.get("target_area", "center")
            use_smart = bool(conf.get("use_smart", True))
            alpha = float(conf.get("alpha", 1.0))

            # ---- mode: v2 with optional rerank ----
            if fusion_mode == "v2":
                if rerank_clip and clip_scorer is not None:
                    # build score prompt
                    scene_prompt = conf.get("scene_prompt", "")
                    control_prompt = conf.get("control_prompt", conf["text"])
                    text_for_score = (scene_prompt + " " + control_prompt).strip()

                    num_cand = int(conf.get("num_candidates", 12))
                    jitter = float(conf.get("jitter", 0.12))
                    lambda_overlap = float(conf.get("lambda_overlap", default_lambda_overlap))

                    candidate_scales = conf.get("candidate_scales", [final_sc])
                    if not isinstance(candidate_scales, (list, tuple)):
                        candidate_scales = [float(candidate_scales)]
                    candidate_scales = [float(x) for x in candidate_scales]

                    best_total = None
                    best_canvas = None
                    best_union = None

                    for sc in candidate_scales:
                        # estimate object patch size (from scaled mask bbox)
                        mask_l = F.interpolate(item["mask_img"], size=(H, W), mode="bilinear", align_corners=False).clamp(0, 1)
                        _, m_sc = scale_in_latent_canvas(item["latent"], mask_l, scale_factor=sc)
                        m_sc = feather_mask(m_sc, iters=2, k=3)

                        bb = bbox_from_mask(m_sc, thr=float(conf.get("mask_thr_bbox", 0.2)))
                        if bb is None:
                            continue
                        ph = int(bb[1] - bb[0] + 1)
                        pw = int(bb[3] - bb[2] + 1)

                        # base coords
                        base_coords = None
                        if use_smart and location_prompt:
                            base_coords = get_text_guided_coords(
                                base_pil_image=base_pil,
                                prompt=location_prompt,
                                patch_h=ph,
                                patch_w=pw,
                                feat_hw=(H, W),
                                avoid_mask=occupied,
                                avoid_strength=float(10.0),
                                avoid_thr=float(0.05),
                            )

                        if base_coords is None:
                            base_coords = _calculate_dynamic_coords(H, W, ph, pw, target_area)
                        base_coords = tuple(int(x) for x in base_coords)

                        cand_coords = generate_candidate_coords(H, W, ph, pw, base_coords, num=num_cand, jitter=jitter)

                        cand_canvases = []
                        cand_unions = []

                        for coords in cand_coords:
                            c_can, c_uni = semantic_fusion_v2(
                                canvas_latent=canvas_latent,
                                object_latent=item["latent"],
                                object_mask_img=item["mask_img"],
                                raw_base_image=base_pil,
                                location_prompt=location_prompt,
                                target_area=target_area,
                                scale_factor=sc,
                                use_smart_placement=False,
                                overlap_mode=overlap_mode,
                                occupied_mask_latent=occupied,
                                forced_coords=coords,
                                alpha=alpha,
                                mask_thr_bbox=float(conf.get("mask_thr_bbox", 0.2)),
                                feather_iters=int(conf.get("feather_iters", 2)),
                                feather_k=int(conf.get("feather_k", 3)),
                            )
                            cand_canvases.append(c_can)
                            cand_unions.append(c_uni)

                        if not cand_canvases:
                            continue

                        cand_batch = torch.cat(cand_canvases, dim=0)
                        with torch.no_grad():
                            decoded = rae.decode(cand_batch).clamp(0, 1)

                        scores_clip = clip_scorer.score(decoded, [text_for_score] * decoded.shape[0])

                        overlaps = torch.stack([(u * occupied).mean() for u in cand_unions]).to(scores_clip.device)
                        scores_total = scores_clip - float(lambda_overlap) * overlaps

                        idx = int(torch.argmax(scores_total).item())
                        total_best = float(scores_total[idx].item())

                        if best_total is None or total_best > best_total:
                            best_total = total_best
                            best_canvas = cand_canvases[idx]
                            best_union = cand_unions[idx]

                    if best_canvas is None:
                        best_canvas, best_union = semantic_fusion_v2(
                            canvas_latent=canvas_latent,
                            object_latent=item["latent"],
                            object_mask_img=item["mask_img"],
                            raw_base_image=base_pil,
                            location_prompt=location_prompt,
                            target_area=target_area,
                            scale_factor=final_sc,
                            use_smart_placement=use_smart,
                            overlap_mode=overlap_mode,
                            occupied_mask_latent=occupied,
                            alpha=alpha,
                            mask_thr_bbox=float(conf.get("mask_thr_bbox", 0.2)),
                            feather_iters=int(conf.get("feather_iters", 2)),
                            feather_k=int(conf.get("feather_k", 3)),
                        )

                    canvas_latent, union_mask = best_canvas, best_union

                else:
                    canvas_latent, union_mask = semantic_fusion_v2(
                        canvas_latent=canvas_latent,
                        object_latent=item["latent"],
                        object_mask_img=item["mask_img"],
                        raw_base_image=base_pil,
                        location_prompt=location_prompt,
                        target_area=target_area,
                        scale_factor=final_sc,
                        use_smart_placement=use_smart,
                        overlap_mode=overlap_mode,
                        occupied_mask_latent=occupied,
                        alpha=alpha,
                        mask_thr_bbox=float(conf.get("mask_thr_bbox", 0.2)),
                        feather_iters=int(conf.get("feather_iters", 2)),
                        feather_k=int(conf.get("feather_k", 3)),
                    )

            # ---- mode: m2 ----
            elif fusion_mode == "m2":
                canvas_latent, union_mask = semantic_fusion_m2(
                    canvas_latent=canvas_latent,
                    object_latent=item["latent"],
                    object_mask_img=item["mask_img"],
                    raw_base_image=base_pil,
                    location_prompt=location_prompt,
                    target_area=target_area,
                    scale_factor=final_sc,
                    use_smart_placement=use_smart,
                    overlap_mode=overlap_mode,
                    occupied_mask_latent=occupied,
                    alpha=alpha,
                    mask_thr_bbox=float(conf.get("mask_thr_bbox", 0.2)),
                    base_unit_ratio=float(conf.get("m2_base_unit_ratio", 0.6)),
                    min_size=int(conf.get("m2_min_size", 8)),
                )
            else:
                raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

            union_total = (union_total + union_mask).clamp(0, 1)
            occupied = (occupied + union_mask).clamp(0, 1)

        # decode preview
        with torch.no_grad():
            preview = rae.decode(canvas_latent).clamp(0, 1)

        out_png = os.path.join(output_dir, f"check_{fusion_mode}_g{gscale}_{ts}.png")
        save_image(preview.float(), out_png)

        out_pt = make_scale_pt(fused_prefix, gscale)
        torch.save(
            {
                "fused_features": canvas_latent.detach().cpu(),
                "union_mask": union_total.detach().cpu(),
                "occupied_mask": occupied.detach().cpu(),
                "global_scale": gscale,
                "base_image": base_image_path,
                "objects_info": objects_sorted,
                "timestamp": ts,
                "fusion_mode": fusion_mode,
                "overlap_mode": overlap_mode,
                "rerank": None if not rerank_clip else {
                    "clip_model": clip_model_id,
                    "image_size": 224,
                    "lambda_overlap_default": default_lambda_overlap,
                },
            },
            out_pt,
        )

        print(f"[OK] Saved pt: {out_pt}")
        print(f"[OK] Preview: {out_png}")

    cleanup_memory()


# =========================================================================
# CLI
# =========================================================================
def parse_args():
    p = argparse.ArgumentParser("Stage1 Runner (merged options)")
    p.add_argument("--base", type=str, required=True, help="base image path")
    p.add_argument("--objects", type=str, required=True, help="objects json path (list of dicts)")
    p.add_argument("--outdir", type=str, required=True, help="output dir for previews")
    p.add_argument("--fused_prefix", type=str, required=True, help="prefix path for saving pt (e.g. out/fused.pt)")
    p.add_argument("--t_size", type=int, default=448)

    p.add_argument("--global_scales", type=str, default="1.0", help="comma list, e.g. 0.3,0.35,0.4")
    p.add_argument("--fusion_mode", type=str, default="v2", choices=["v2", "m2"])
    p.add_argument("--overlap_mode", type=str, default="no_overwrite", choices=["allow", "no_overwrite", "alpha"])

    p.add_argument("--rerank_clip", type=int, default=1, help="1 enable CLIP rerank (only v2), 0 disable")
    p.add_argument("--clip_model_id", type=str, default="openai/clip-vit-base-patch32")
    p.add_argument("--lambda_overlap_default", type=float, default=0.3)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.objects, "r", encoding="utf-8") as f:
        objects_list = json.load(f)
    assert isinstance(objects_list, list), "objects json must be a list[dict]"

    global_scales = [float(x) for x in args.global_scales.split(",") if x.strip()]

    stage1_run(
        base_image_path=args.base,
        objects_list=objects_list,
        output_dir=args.outdir,
        fused_prefix=args.fused_prefix,
        t_size=args.t_size,
        global_scales=global_scales,
        fusion_mode=args.fusion_mode,
        rerank_clip=bool(args.rerank_clip),
        overlap_mode=args.overlap_mode,
        clip_model_id=args.clip_model_id,
        default_lambda_overlap=args.lambda_overlap_default,
    )


if __name__ == "__main__":
    main()
