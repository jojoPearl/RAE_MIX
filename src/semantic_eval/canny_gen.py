#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random
import argparse
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG", ".JPG"}

def list_images(folder: Path):
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix in IMG_EXTS]

def load_imagenet_class_index():
    """
    Returns wnid -> (class_id, class_name)
    Uses torchvision's imagenet_class_index.json if available.
    """
    try:
        import torchvision
        root = Path(torchvision.__file__).parent
        candidates = list(root.rglob("imagenet_class_index.json"))
        if not candidates:
            raise FileNotFoundError("imagenet_class_index.json not found in torchvision.")
        with open(candidates[0], "r", encoding="utf-8") as f:
            idx = json.load(f)  # {"0": ["n01440764","tench"], ...}
        out = {}
        for k, (wnid, name) in idx.items():
            out[wnid] = (int(k), name)
        return out
    except Exception as e:
        print(f"[Warn] Could not load official ImageNet class index from torchvision: {e}")
        return None

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagenet_dir", type=str, required=True,
                    help="Root containing train/ and/or val/ in standard ImageNet-1k layout.")
    ap.add_argument("--split", type=str, default="train", choices=["train", "val"])
    ap.add_argument("--num_classes", type=int, default=20)
    ap.add_argument("--per_class", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="subset_20x5")
    ap.add_argument("--copy_mode", type=str, default="copy", choices=["none", "copy", "symlink"])
    ap.add_argument("--wnids", type=str, default="",
                    help="Optional comma-separated wnids to use (overrides random class sampling).")
    args = ap.parse_args()

    random.seed(args.seed)

    src_root = Path(args.imagenet_dir).expanduser().resolve()
    split_root = src_root / args.split
    if not split_root.exists():
        raise SystemExit(f"Split folder not found: {split_root}")

    wnid_dirs = [p for p in split_root.iterdir() if p.is_dir()]
    wnids_all = sorted([p.name for p in wnid_dirs])
    if not wnids_all:
        raise SystemExit(f"No class folders found under: {split_root}")

    if args.wnids.strip():
        chosen_wnids = [w.strip() for w in args.wnids.split(",") if w.strip()]
        missing = [w for w in chosen_wnids if w not in wnids_all]
        if missing:
            raise SystemExit(f"These wnids are missing under {split_root}: {missing}")
    else:
        if len(wnids_all) < args.num_classes:
            raise SystemExit(f"Only {len(wnids_all)} classes found, but requested {args.num_classes}.")
        chosen_wnids = random.sample(wnids_all, args.num_classes)

    wnid2official = load_imagenet_class_index()

    out_dir = Path(args.out_dir).resolve()
    ensure_dir(out_dir)
    img_out_root = out_dir / "images" / args.split
    ensure_dir(img_out_root)

    class_map = {}
    if wnid2official is not None:
        for wnid in chosen_wnids:
            cid, cname = wnid2official.get(wnid, (-1, wnid))
            class_map[wnid] = {"wnid": wnid, "class_id": int(cid), "class_name": str(cname)}
    else:
        # fallback stable IDs (NOT official ImageNet IDs)
        for i, wnid in enumerate(sorted(chosen_wnids)):
            class_map[wnid] = {"wnid": wnid, "class_id": int(i), "class_name": wnid}

    manifest_path = out_dir / "subset_manifest.jsonl"
    total = 0

    with open(manifest_path, "w", encoding="utf-8") as f:
        for wnid in chosen_wnids:
            folder = split_root / wnid
            imgs = list_images(folder)
            if len(imgs) < args.per_class:
                raise SystemExit(f"Class {wnid} has only {len(imgs)} images (<{args.per_class}).")

            picked = random.sample(imgs, args.per_class)
            for p in picked:
                item = {
                    "src_path": str(p),
                    "wnid": wnid,
                    "class_id": int(class_map[wnid]["class_id"]),
                    "class_name": class_map[wnid]["class_name"],
                }

                if args.copy_mode != "none":
                    dst_dir = ensure_dir(img_out_root / wnid)
                    dst_path = dst_dir / p.name
                    if args.copy_mode == "copy":
                        shutil.copy2(p, dst_path)
                    elif args.copy_mode == "symlink":
                        if dst_path.exists():
                            dst_path.unlink()
                        os.symlink(p, dst_path)
                    item["subset_path"] = str(dst_path)

                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                total += 1

    with open(out_dir / "class_id_map.json", "w", encoding="utf-8") as f:
        json.dump({
            "split": args.split,
            "seed": args.seed,
            "num_classes": len(chosen_wnids),
            "per_class": args.per_class,
            "total_images": total,
            "classes": class_map,
            "official_class_ids": wnid2official is not None
        }, f, indent=2, ensure_ascii=False)

    print("[Done]")
    print(f"  manifest: {manifest_path}")
    print(f"  class map: {out_dir / 'class_id_map.json'}")
    if args.copy_mode != "none":
        print(f"  subset images: {img_out_root}")

if __name__ == "__main__":
    main()
