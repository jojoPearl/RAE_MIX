#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, json, random, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

# ---- project imports ----
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.semantic.modelManager import ModelManager
from src.semantic.config import DEVICE, DTYPE, DTYPE_MODERN

device = DEVICE
dtype  = DTYPE

OUT_ROOT = "assets/out_latent_canny"

class CannyExtractor(nn.Module):
    def __init__(self, low=0.1, high=0.2):
        super().__init__()
        import kornia, kornia.filters
        self.kornia = kornia
        self.filters = kornia.filters
        self.low, self.high = low, high

    @torch.no_grad()
    def forward(self, x):  # x: [B,3,H,W] in [0,1]
        g = self.kornia.color.rgb_to_grayscale(x)
        e, _ = self.filters.canny(g, low_threshold=self.low, high_threshold=self.high)
        return e.clamp(0,1).repeat(1,3,1,1)

class ManifestDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_jsonl: str):
        self.items = []
        with open(manifest_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                obj = json.loads(line)
                p = obj.get("subset_path", obj["src_path"])
                self.items.append((p, int(obj["class_id"])))
        if not self.items:
            raise RuntimeError("Empty manifest")

        self.to_tensor = T.ToTensor()

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        img = Image.open(p).convert("RGB").resize((448,448), Image.LANCZOS)
        x = self.to_tensor(img)  # [3,448,448] in [0,1]
        return x, torch.tensor(y, dtype=torch.long)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--canny_low", type=float, default=0.1)
    ap.add_argument("--canny_high", type=float, default=0.2)
    ap.add_argument("--p_drop_y", type=float, default=0.1)   # CFG for class only
    ap.add_argument("--save_every", type=int, default=200)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    out_dir = Path(OUT_ROOT); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "latest.pt"

    manager = ModelManager(device=device)
    rae = manager.load_rae().eval()
    dit = manager.load_dit()
    if dit is None:
        raise RuntimeError("ModelManager.load_dit() returned None.")
    dit = dit.to(device=device, dtype=dtype).train()

    opt = torch.optim.AdamW(dit.parameters(), lr=args.lr, betas=(0.9,0.999), weight_decay=0.01)

    canny = CannyExtractor(args.canny_low, args.canny_high).to(device=device).eval()

    ds = ManifestDataset(args.manifest)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                     num_workers=2, pin_memory=True, drop_last=True)
    it = iter(dl)
    ema = None

    for step in range(1, args.steps+1):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(dl); x, y = next(it)

        x = x.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=torch.long)
        B = x.shape[0]

        # class CFG dropout (可选)
        y_used = y.clone()
        dropy = (torch.rand(B, device=device) < args.p_drop_y)
        y_used[dropy] = 1000  # null class id（按你工程约定）

        # 1) canny as "data", encode to latent z0_edge
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=DTYPE_MODERN):
            e = canny(x.float())      # [B,3,448,448]
            z0 = rae.encode(e)        # [B,768,32,32]  (edge latent)

        # 2) rectified flow mix with pure noise
        t = torch.rand(B, device=device, dtype=dtype).clamp(1e-4, 1.0)  # [B]
        eps = torch.randn_like(z0)
        t4  = t.view(B,1,1,1)
        zt  = (1.0 - t4) * z0 + t4 * eps
        target_v = (eps - z0)

        # 3) DiT predicts velocity
        pred_v = dit(zt, t, y_used, s=None, mask=None)
        loss = F.mse_loss(pred_v, target_v)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dit.parameters(), 1.0)
        opt.step()

        lv = float(loss.detach().cpu())
        ema = lv if ema is None else (0.95*ema + 0.05*lv)
        if step % 50 == 0:
            print(f"step {step:05d}/{args.steps}  loss {lv:.6f}  ema {ema:.6f}")

        if step % args.save_every == 0 or step == args.steps:
            torch.save({
                "format": "dit_latent_edge_v1",
                "manifest": args.manifest,
                "canny_thresholds": [float(args.canny_low), float(args.canny_high)],
                "step": int(step),
                "seed": int(args.seed),
                "dit": dit.state_dict(),
            }, ckpt_path)
            print(f"[Saved] {ckpt_path}")

if __name__ == "__main__":
    main()
