import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")
warnings.filterwarnings("ignore", category=UserWarning)

import os
import sys
import glob
import json
import argparse
import datetime
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torchvision.utils import save_image

# -------------------- Path setup --------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

# -------------------- Imports --------------------
from src.semantic.segmentation import get_text_guided_coords, extract_semantic_mask_with_clipseg
from src.semantic.utils import cleanup_memory, load_and_transform, _calculate_dynamic_coords
from src.semantic.resize import apply_m1_scaling
from src.semantic.modelManager import ModelManager
from src.semantic.config import DEVICE, DTYPE_MODERN

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = DEVICE
t_size = 448


# -------------------------------------------------------------------------
# CLIP Scorer (optional)
# -------------------------------------------------------------------------
class CLIPScorerTF:
    def __init__(self, device: torch.device, model_id: str = "openai/clip-vit-base-patch32"):
        from transformers import CLIPModel, CLIPProcessor
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()

        self._text_cache: Dict[str, torch.Tensor] = {}

        self._mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        self._std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

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
            tok = self.processor(
                text=missing,
                images=None,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
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


# -------------------------------------------------------------------------
# Utils
# -------------------------------------------------------------------------
def timestamp_str():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


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
    H: int,
    W: int,
    ph: int,
    pw: int,
    base_coords: Tuple[int, int, int, int],
    num: int = 12,
    jitter: float = 0.12
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


# -------------------------------------------------------------------------
# Geometry: scale in latent canvas
# -------------------------------------------------------------------------
def scale_in_latent_canvas(
    feat: torch.Tensor,
    mask: torch.Tensor,
    scale_factor: float,
    padding_mode: str = "zeros"
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

    grid_m = F.affine_grid(theta, size=mask.size(), align_corners=False).to(dtype=feat.dtype)
    mask_s = F.grid_sample(mask, grid_m, mode="bilinear", padding_mode=padding_mode, align_corners=False)
    return feat_s, mask_s.clamp(0, 1)


# -------------------------------------------------------------------------
# Crop object + mask in pixel space for encoder
# -------------------------------------------------------------------------
def get_cropped_object_tensor(
    raw_image: Image.Image,
    target_text: str,
    scale_factor: float = 1.0,
    background_mode: str = "mean",
    target_size_for_encoder: int = 448,
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
        mask_to_scale = Image.fromarray(
            (mask_soft[rmin:rmax + 1, cmin:cmax + 1] * 255.0).astype(np.uint8),
            mode="L"
        )

    final_obj_pil = apply_m1_scaling(obj_to_scale.convert("RGB"), scale_factor, target_size_for_encoder, background_mode)
    final_mask_pil = apply_m1_scaling(mask_to_scale.convert("RGB"), scale_factor, target_size_for_encoder, background_mode="black").convert("L")

    obj_t = T.ToTensor()(final_obj_pil).unsqueeze(0).to(device=device, dtype=DTYPE_MODERN)
    mask_t = T.ToTensor()(final_mask_pil).unsqueeze(0).to(device=device, dtype=torch.float32)
    return obj_t, mask_t[:, 0:1].clamp(0, 1)


# -------------------------------------------------------------------------
# Adaptive Scale (optional)
# -------------------------------------------------------------------------
def calculate_adaptive_scale(
    base_pil_image: Image.Image,
    location_prompt: str,
    H: int,
    W: int,
    gamma: float = 0.5,
    eta: float = 0.9,
    min_scale: float = 0.2,
    max_scale: float = 0.8
) -> float:
    if not location_prompt:
        return 0.5

    heatmap = extract_semantic_mask_with_clipseg(
        image=base_pil_image,
        target_text=location_prompt,
        feature_size=(H, W),
        threshold=0.1
    )
    mask_binary = (heatmap > 0.1).float()
    area_pixels = mask_binary.sum().item()
    total_pixels = H * W
    r = area_pixels / max(total_pixels, 1)

    if r < 1e-5:
        print(f"  [Adaptive] No active area for '{location_prompt}', default 0.5")
        return 0.5

    rho = r ** gamma
    suggested = rho * eta
    final = max(min_scale, min(suggested, max_scale))
    print(f"  [Adaptive] r={r:.4f} rho={rho:.4f} suggested={suggested:.4f} final={final:.4f}")
    return final


# -------------------------------------------------------------------------
# Fusion
# -------------------------------------------------------------------------
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
                raw_base_image,
                location_prompt,
                ph,
                pw,
                (H, W),
                avoid_mask=avoid,
                avoid_strength=float(10.0),
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
        m_p   = F.interpolate(m_p,   size=base_p.shape[-2:], mode="bilinear", align_corners=False)

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


# -------------------------------------------------------------------------
# Stage1 unified
# -------------------------------------------------------------------------
def stage1_composition(
    base_image_path: str,
    objects_list: List[Dict],
    output_path: str,
    fused_path_prefix: str,
    global_scale_factors: List[float],
    overlap_mode: str = "allow",
    rerank_clip: bool = False,
    clip_model_id: str = "openai/clip-vit-base-patch32",
):
    os.makedirs(output_path, exist_ok=True)
    ts = timestamp_str()

    base_img_tensor = load_and_transform(base_image_path, t_size).to(device=device)
    base_pil_image = Image.open(base_image_path).convert("RGB")

    objects_sorted = sorted(objects_list, key=lambda d: d.get("z_order", 0))

    manager = ModelManager(device=device)
    rae = manager.load_rae()
    clip_scorer = CLIPScorerTF(device=device, model_id=clip_model_id) if rerank_clip else None
    cleanup_memory()

    with torch.amp.autocast("cuda", dtype=DTYPE_MODERN):
        base_latent_init = rae.encode(base_img_tensor)

    # Pre-encode all objects once
    encoded_objects = []
    for conf in objects_sorted:
        r_pil = Image.open(conf["path"]).convert("RGB")
        obj_t, obj_m = get_cropped_object_tensor(
            raw_image=r_pil,
            target_text=conf["text"],
            scale_factor=1.0,  # keep encode quality same as your code
            background_mode="mean",
            target_size_for_encoder=t_size,
        )
        with torch.amp.autocast("cuda", dtype=DTYPE_MODERN):
            obj_l = rae.encode(obj_t)
        encoded_objects.append({"latent": obj_l, "mask_img": obj_m, "config": conf})

    def make_scale_path(prefix: str, g: float) -> str:
        root, ext = os.path.splitext(prefix)
        if ext == "":
            ext = ".pt"
        return f"{root}_scale_{g}{ext}"

    for gscale in global_scale_factors:
        print(f"\n--- [Stage1] Global scale = {gscale} ---")
        canvas_latent = base_latent_init.clone()
        B, C, H, W = canvas_latent.shape

        occupied = torch.zeros((B, 1, H, W), device=device, dtype=canvas_latent.dtype)
        union_total = torch.zeros((B, 1, H, W), device=device, dtype=canvas_latent.dtype)

        for item in encoded_objects:
            conf = item["config"]
            location_prompt = conf.get("location_prompt", None)
            target_area = conf.get("target_area", "center")
            use_smart = bool(conf.get("use_smart", True))
            alpha = float(conf.get("alpha", 1.0))
            base_scale = float(conf.get("base_scale", 1.0))

            use_adaptive = bool(conf.get("use_adaptive_scale", False))

            # base scale (default)
            final_sc = float(gscale * base_scale)

            # candidate scales
            if use_adaptive and location_prompt:
                print(f"Object: {conf['text']} | adaptive scale based on '{location_prompt}'")
                adaptive_sc = calculate_adaptive_scale(
                    base_pil_image=base_pil_image,
                    location_prompt=location_prompt,
                    H=H, W=W,
                    gamma=float(conf.get("gamma", 0.5)),
                    eta=float(conf.get("eta", 0.9)),
                    min_scale=float(conf.get("min_scale", 0.2)),
                    max_scale=float(conf.get("max_scale", 0.8)),
                )
                candidate_scales = [adaptive_sc * 0.9, adaptive_sc * 1.0, adaptive_sc * 1.1]
            else:
                candidate_scales = conf.get("candidate_scales", [final_sc])
                if not isinstance(candidate_scales, (list, tuple)):
                    candidate_scales = [float(candidate_scales)]
                candidate_scales = [float(x) for x in candidate_scales]

            # If no rerank, just do single placement once (same as your v2 basic)
            if not rerank_clip:
                canvas_latent, union_mask = semantic_fusion_v2(
                    canvas_latent=canvas_latent,
                    object_latent=item["latent"],
                    object_mask_img=item["mask_img"],
                    raw_base_image=base_pil_image,
                    location_prompt=location_prompt,
                    target_area=target_area,
                    scale_factor=candidate_scales[0],
                    use_smart_placement=use_smart,
                    overlap_mode=overlap_mode,
                    occupied_mask_latent=occupied,
                    alpha=alpha,
                )
                union_total = (union_total + union_mask).clamp(0, 1)
                occupied = (occupied + union_mask).clamp(0, 1)
                print(f"  + Fuse '{conf['text']}' scale={candidate_scales[0]:.3f} (no rerank)")
                continue

            # Rerank path (CLIP + overlap penalty)
            scene_prompt = conf.get("scene_prompt", "")
            control_prompt = conf.get("control_prompt", conf["text"])
            text_for_score = (scene_prompt + " " + control_prompt).strip()

            num_cand = int(conf.get("num_candidates", 12))
            jitter = float(conf.get("jitter", 0.12))
            lambda_overlap = float(conf.get("lambda_overlap", 0.3))

            best_total = None
            best_canvas = None
            best_union = None
            best_clip = None
            best_ov = None
            best_sc = None

            for sc in candidate_scales:
                # estimate patch size from scaled mask bbox
                mask_l = F.interpolate(item["mask_img"], size=(H, W), mode="bilinear", align_corners=False).clamp(0, 1)
                _, m_sc = scale_in_latent_canvas(item["latent"], mask_l, scale_factor=sc)
                m_sc = feather_mask(m_sc, iters=2, k=3)
                bb = bbox_from_mask(m_sc, thr=0.2)
                if bb is None:
                    continue
                ph = int(bb[1] - bb[0] + 1)
                pw = int(bb[3] - bb[2] + 1)

                base_coords = None
                if use_smart and location_prompt:
                    base_coords = get_text_guided_coords(
                        base_pil_image,
                        location_prompt,
                        ph,
                        pw,
                        (H, W),
                        avoid_mask=occupied,
                        avoid_strength=float(10.0),
                        avoid_thr=float(0.05),
                    )
                if base_coords is None:
                    base_coords = _calculate_dynamic_coords(H, W, ph, pw, target_area)

                base_coords = tuple(int(x) for x in base_coords)
                cand_coords = generate_candidate_coords(H, W, ph, pw, base_coords, num=num_cand, jitter=jitter)

                cand_canvases: List[torch.Tensor] = []
                cand_unions: List[torch.Tensor] = []
                for coords in cand_coords:
                    c_can, c_uni = semantic_fusion_v2(
                        canvas_latent=canvas_latent,
                        object_latent=item["latent"],
                        object_mask_img=item["mask_img"],
                        raw_base_image=base_pil_image,
                        location_prompt=location_prompt,
                        target_area=target_area,
                        scale_factor=sc,
                        use_smart_placement=False,
                        overlap_mode=overlap_mode,
                        occupied_mask_latent=occupied,
                        forced_coords=coords,
                        alpha=alpha,
                    )
                    cand_canvases.append(c_can)
                    cand_unions.append(c_uni)

                if not cand_canvases:
                    continue

                cand_batch = torch.cat(cand_canvases, dim=0)
                decoded = rae.decode(cand_batch).clamp(0, 1)

                scores_clip = clip_scorer.score(decoded, [text_for_score] * decoded.shape[0])  # [N]

                overlaps = []
                for u in cand_unions:
                    overlaps.append((u * occupied).mean())
                overlaps = torch.stack(overlaps).to(scores_clip.device)

                scores_total = scores_clip - lambda_overlap * overlaps

                idx = int(torch.argmax(scores_total).item())
                clip_best = float(scores_clip[idx].item())
                ov_best = float(overlaps[idx].item())
                total_best = float(scores_total[idx].item())

                if best_total is None or total_best > best_total:
                    best_total = total_best
                    best_canvas = cand_canvases[idx]
                    best_union = cand_unions[idx]
                    best_clip = clip_best
                    best_ov = ov_best
                    best_sc = sc

            if best_canvas is None:
                # fallback
                best_canvas, best_union = semantic_fusion_v2(
                    canvas_latent=canvas_latent,
                    object_latent=item["latent"],
                    object_mask_img=item["mask_img"],
                    raw_base_image=base_pil_image,
                    location_prompt=location_prompt,
                    target_area=target_area,
                    scale_factor=final_sc,
                    use_smart_placement=use_smart,
                    overlap_mode=overlap_mode,
                    occupied_mask_latent=occupied,
                    alpha=alpha,
                )
                print(f"Object: {conf['text']} | fallback (no candidate)")
            else:
                print(
                    f"Object: {conf['text']} | scale={best_sc:.3f} | "
                    f"Total={best_total:.4f} CLIP={best_clip:.4f} Overlap={best_ov:.4f}"
                )

            canvas_latent = best_canvas
            union_mask = best_union
            union_total = (union_total + union_mask).clamp(0, 1)
            occupied = (occupied + union_mask).clamp(0, 1)

        # preview
        preview = rae.decode(canvas_latent).clamp(0, 1)
        out_png = os.path.join(output_path, f"check_{gscale}_{ts}.png")
        save_image(preview.float(), out_png)

        # save pt
        out_pt = make_scale_path(fused_path_prefix, gscale)
        torch.save(
            {
                "fused_features": canvas_latent.detach().cpu(),
                "union_mask": union_total.detach().cpu(),
                "occupied_mask": occupied.detach().cpu(),
                "global_scale": gscale,
                "base_image": base_image_path,
                "objects_info": objects_sorted,
                "timestamp": ts,
                "overlap_mode": overlap_mode,
                "rerank_clip": bool(rerank_clip),
                "clip_model": clip_model_id if rerank_clip else None,
            },
            out_pt,
        )

        print(f"Saved: {out_pt}")
        print(f"Preview: {out_png}")

    del rae
    cleanup_memory()
    print("\n--- [Stage1] Completed ---")


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
def parse_list_floats(vals: List[str]) -> List[float]:
    # accept: "0.3,0.35,0.4" OR "0.3 0.35 0.4"
    if len(vals) == 1 and "," in vals[0]:
        return [float(x.strip()) for x in vals[0].split(",") if x.strip()]
    return [float(x) for x in vals]


def load_objects_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list), "objects json must be a list of dict"
    return data


def build_argparser():
    p = argparse.ArgumentParser("Stage1 unified (basic / clip rerank / adaptive scale)")
    p.add_argument("--base", "-b", required=True, help="base image path")
    p.add_argument("--objects", "-j", required=True, help="objects json path (list of dict)")
    p.add_argument("--out", "-o", required=True, help="output dir for previews and pt")
    p.add_argument("--fused_prefix", "-p", default=None, help="pt prefix path, default: <out>/fused_results.pt")
    p.add_argument("--gscales", nargs="+", default=["0.3", "0.35", "0.4", "0.45", "0.5"],
                   help="global scales, e.g. --gscales 0.3 0.35 0.4 or --gscales 0.3,0.35,0.4")
    p.add_argument("--overlap_mode", choices=["allow", "no_overwrite", "alpha"], default="allow")

    # clip rerank
    p.add_argument("--rerank_clip", action="store_true", help="enable CLIP rerank")
    p.add_argument("--clip_model", default="openai/clip-vit-base-patch32")

    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    os.makedirs(args.out, exist_ok=True)

    fused_prefix = args.fused_prefix or os.path.join(args.out, "fused_results.pt")
    gscales = parse_list_floats(args.gscales)
    objects_list = load_objects_json(args.objects)

    stage1_composition(
        base_image_path=args.base,
        objects_list=objects_list,
        output_path=args.out,
        fused_path_prefix=fused_prefix,
        global_scale_factors=gscales,
        overlap_mode=args.overlap_mode,
        rerank_clip=bool(args.rerank_clip),
        clip_model_id=args.clip_model,
    )
