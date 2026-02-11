from omegaconf import OmegaConf, DictConfig
from typing import List, Tuple, Union, Optional
from PIL import Image
import json
import numpy as np
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.datasets import ImageFolder
from pathlib import Path
from copy import deepcopy
from .dist_utils import setup_distributed
import cv2


def pil_to_np_rgb(pil: Image.Image) -> np.ndarray:
    return np.array(pil.convert("RGB"))


def make_safe_canny(img_uint8_rgb: np.ndarray) -> np.ndarray:
    """
    Robust Auto-Canny: adaptive thresholds + light dilation.
    """
    if img_uint8_rgb.shape[2] == 3:
        gray = cv2.cvtColor(img_uint8_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_uint8_rgb

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    v = np.median(blur)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    edges = cv2.Canny(blur, lower, upper)
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    return np.stack([edges, edges, edges], axis=-1)


def shift_edge_left(edge_rgb_uint8: np.ndarray, ratio: float = 0.25) -> np.ndarray:
    """
    edge_rgb_uint8: (H, W, 3) uint8
    ratio: shift amount as fraction of width
    """
    h, w, _ = edge_rgb_uint8.shape
    s = int(w * ratio)
    out = np.zeros_like(edge_rgb_uint8)
    if s < w:
        out[:, : w - s] = edge_rgb_uint8[:, s:]
    return out


def canny_from_pil(
    pil_img: Image.Image,
    *,
    out_size: Optional[int] = None,
    grabcut_iters: int = 4,
    rect_margin: float = 0.08,
    close_ks: int = 7,
    close_iter: int = 1,
    thin_erode: int = 0,
    final_dilate_ks: int = 3,
    final_dilate_iter: int = 1,
    add_inner_canny: bool = True,
    inner_low: int = 60,
    inner_high: int = 180,
    inner_weight: float = 0.35,
    **_unused,
) -> Image.Image:
    """
    Tight mask-guided outline edge using GrabCut.
    Returns a PIL image (RGB) so downstream crop/flip stays aligned with GT.
    """
    img = pil_img
    if out_size is not None:
        img = img.resize((out_size, out_size), Image.BICUBIC)
    rgb = np.array(img)
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]

    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    m = int(min(h, w) * rect_margin)
    rect = (m, m, w - 2 * m, h - 2 * m)
    cv2.grabCut(bgr, mask, rect, bgd_model, fgd_model, grabcut_iters, cv2.GC_INIT_WITH_RECT)

    fg = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        1,
        0,
    ).astype(np.uint8)

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k3, iterations=2)

    fg_erode = cv2.erode(fg, k3, iterations=1)
    outline = (fg - fg_erode).clip(0, 1).astype(np.uint8) * 255

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ks, close_ks))
    outline = cv2.morphologyEx(outline, cv2.MORPH_CLOSE, k_close, iterations=close_iter)

    if thin_erode > 0:
        outline = cv2.erode(outline, k3, iterations=thin_erode)

    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (final_dilate_ks, final_dilate_ks))
    outline = cv2.dilate(outline, k_dilate, iterations=final_dilate_iter)

    edge = outline.astype(np.float32) / 255.0
    edge = edge * fg.astype(np.float32)

    if add_inner_canny:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 1.2)
        inner = cv2.Canny(gray, inner_low, inner_high).astype(np.float32) / 255.0
        inner = inner * fg.astype(np.float32)
        edge = np.clip(edge + inner_weight * inner, 0.0, 1.0)

    edge_rgb = np.stack([edge, edge, edge], axis=-1)
    edge_rgb = (edge_rgb * 255.0).astype(np.uint8)
    return Image.fromarray(edge_rgb)


class CannyImageFolder(ImageFolder):
    """
    ImageFolder that returns image and canny edge with synchronized transforms.
    """
    def __init__(
        self,
        root,
        image_size: int = 256,
        random_flip: bool = True,
        *,
        ilsvrc_class_index_path: Optional[str] = None,
        canny_low: Optional[int] = 60,
        canny_high: Optional[int] = 160,
        canny_blur: int = 3,
        canny_auto: bool = True,
        canny_sigma: float = 0.33,
        canny_dilate: bool = True,
        canny_dilate_ksize: int = 3,
        canny_dilate_iter: int = 1,
        edge_blur_sigma: float = 1.6,
        edge_keep_topk_cc: int = 2,
        edge_min_cc_area: int = 80,
        edge_resize: bool = False,
        edge_grabcut_iters: int = 4,
        edge_rect_margin: float = 0.08,
        edge_close_ks: int = 7,
        edge_close_iter: int = 1,
        edge_thin_erode: int = 0,
        edge_final_dilate_ks: int = 3,
        edge_final_dilate_iter: int = 1,
        edge_add_inner_canny: bool = True,
        edge_inner_low: int = 60,
        edge_inner_high: int = 180,
        edge_inner_weight: float = 0.35,
    ):
        super().__init__(root, transform=None)
        self.image_size = image_size
        self.random_flip = random_flip
        self.synset_to_ilsvrc = None
        if ilsvrc_class_index_path:
            self.synset_to_ilsvrc = load_synset_to_ilsvrc(ilsvrc_class_index_path)
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.canny_blur = canny_blur
        self.canny_auto = canny_auto
        self.canny_sigma = canny_sigma
        self.canny_dilate = canny_dilate
        self.canny_dilate_ksize = canny_dilate_ksize
        self.canny_dilate_iter = canny_dilate_iter
        self.edge_blur_sigma = edge_blur_sigma
        self.edge_keep_topk_cc = edge_keep_topk_cc
        self.edge_min_cc_area = edge_min_cc_area
        self.edge_resize = edge_resize
        self.edge_grabcut_iters = edge_grabcut_iters
        self.edge_rect_margin = edge_rect_margin
        self.edge_close_ks = edge_close_ks
        self.edge_close_iter = edge_close_iter
        self.edge_thin_erode = edge_thin_erode
        self.edge_final_dilate_ks = edge_final_dilate_ks
        self.edge_final_dilate_iter = edge_final_dilate_iter
        self.edge_add_inner_canny = edge_add_inner_canny
        self.edge_inner_low = edge_inner_low
        self.edge_inner_high = edge_inner_high
        self.edge_inner_weight = edge_inner_weight

    def __getitem__(self, index):
        path, target = self.samples[index]
        if self.synset_to_ilsvrc is not None:
            synset = self.classes[target]
            if synset not in self.synset_to_ilsvrc:
                raise KeyError(f"Synset '{synset}' missing in ILSVRC mapping.")
            target = self.synset_to_ilsvrc[synset]
        sample = self.loader(path)
        edge = canny_from_pil(
            sample,
            out_size=self.image_size if self.edge_resize else None,
            grabcut_iters=self.edge_grabcut_iters,
            rect_margin=self.edge_rect_margin,
            close_ks=self.edge_close_ks,
            close_iter=self.edge_close_iter,
            thin_erode=self.edge_thin_erode,
            final_dilate_ks=self.edge_final_dilate_ks,
            final_dilate_iter=self.edge_final_dilate_iter,
            add_inner_canny=self.edge_add_inner_canny,
            inner_low=self.edge_inner_low,
            inner_high=self.edge_inner_high,
            inner_weight=self.edge_inner_weight,
        )
        sample = center_crop_arr(sample, self.image_size)
        edge = center_crop_arr(edge, self.image_size)
        if self.random_flip and torch.rand(1).item() < 0.5:
            sample = TF.hflip(sample)
            edge = TF.hflip(edge)
        sample = TF.to_tensor(sample)
        edge = TF.to_tensor(edge)
        return sample, target, edge


def load_synset_to_ilsvrc(json_path: str) -> dict:
    """
    Load ImageNet class index mapping.
    Expected format: {"0": ["n01440764", "tench"], ...}
    Returns: {synset: int(ilsvrc_idx)}
    """
    with open(json_path, "r") as f:
        obj = json.load(f)
    mapping = {}
    for k, v in obj.items():
        if isinstance(v, (list, tuple)) and len(v) >= 1:
            synset = v[0]
        else:
            continue
        mapping[synset] = int(k)
    return mapping


def parse_configs(config: Union[DictConfig, str]) -> Tuple[DictConfig, DictConfig, DictConfig, DictConfig, DictConfig, DictConfig, DictConfig]:
    """Load a config file and return component sections as DictConfigs."""
    if isinstance(config, str):
        config = OmegaConf.load(config)
    rae_config = config.get("stage_1", None)
    stage2_config = config.get("stage_2", None)
    transport_config = config.get("transport", None)
    sampler_config = config.get("sampler", None)
    guidance_config = config.get("guidance", None)
    misc = config.get("misc", None)
    training_config = config.get("training", None)
    eval_config = config.get("eval", None)
    return rae_config, stage2_config, transport_config, sampler_config, guidance_config, misc, training_config, eval_config

def none_or_str(value):
    if value == 'None':
        return None
    return value

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def prepare_dataloader(
    data_path: Path,
    batch_size: int,
    workers: int,
    rank: int,
    world_size: int,
    transform: List= None,
    image_size: int = 256,
    return_edges: bool = False,
    ilsvrc_class_index_path: Optional[str] = None,
) -> Tuple[DataLoader, DistributedSampler]:
    if return_edges:
        dataset = CannyImageFolder(
            str(data_path),
            image_size=image_size,
            random_flip=True,
            ilsvrc_class_index_path=ilsvrc_class_index_path,
        )
    else:
        dataset = ImageFolder(str(data_path), transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader, sampler

def get_autocast_scaler(args) -> Tuple[dict, torch.cuda.amp.GradScaler | None]:
    if args.precision == "fp16":
        scaler = GradScaler()
        autocast_kwargs = dict(enabled=True, dtype=torch.float16)
    elif args.precision == "bf16":
        scaler = None
        autocast_kwargs = dict(enabled=True, dtype=torch.bfloat16)
    else:
        scaler = None
        autocast_kwargs = dict(enabled=False)
    
    return scaler, autocast_kwargs
