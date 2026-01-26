import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")
warnings.filterwarnings("ignore", category=UserWarning)

import os
import sys
import glob
import argparse
import datetime
import torch
from torchvision.utils import save_image

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.stage2.transport import create_transport
from src.semantic.utils import cleanup_memory
from src.semantic.modelManager import ModelManager
from src.semantic.config import DEVICE, DTYPE

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = DEVICE
dtype = DTYPE


def timestamp_str():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def list_pt_files(input_path: str):
    if os.path.isfile(input_path) and input_path.endswith(".pt"):
        return [input_path]
    if os.path.isdir(input_path):
        pts = glob.glob(os.path.join(input_path, "*.pt"))
        pts.sort()
        return pts
    return []


def folder_name_from_pt(pt_path: str):
    filename = os.path.basename(pt_path)
    base_name = os.path.splitext(filename)[0]
    return base_name.replace("fused_results_", "")


def parse_strengths(strength_args):
    """
    Accepts:
      --strengths 0.2 0.3 0.6
    or:
      --strengths 0.2,0.3,0.6
    """
    if not strength_args:
        return None

    # Case A: a single string containing commas
    if len(strength_args) == 1 and "," in strength_args[0]:
        parts = [p.strip() for p in strength_args[0].split(",") if p.strip()]
        return [float(p) for p in parts]

    # Case B: space-separated floats
    return [float(x) for x in strength_args]


@torch.no_grad()
def diffuse_one_file(
    fused_path: str,
    editing_strengths,
    total_steps: int,
    dit_model,
    drift_fn,
    device,
    dtype,
):
    data = torch.load(fused_path, map_location=device)
    base_fused_features = data["fused_features"].to(dtype=dtype)

    B, C, H, W = base_fused_features.shape
    dummy_y = torch.zeros(B, dtype=torch.long, device=device)
    model_kwargs = dict(y=dummy_y, s=None, mask=None)

    # keep your current behavior: initial_noise is per-file random
    initial_noise = torch.randn((B, C, H, W), device=device, dtype=dtype)

    sampled_latents_list = []
    for start_t in editing_strengths:
        latents = (1.0 - start_t) * base_fused_features + start_t * initial_noise

        current_steps = int(total_steps * start_t)
        if current_steps < 1:
            current_steps = 1

        timesteps = torch.linspace(
            start_t, 0.0, current_steps + 1, device=device, dtype=dtype
        )

        for i in range(current_steps):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t_curr

            vec_t = torch.ones(B, device=device, dtype=dtype) * t_curr
            velocity = drift_fn(latents, vec_t, dit_model, **model_kwargs)
            latents = latents + velocity * dt

        final_latent = latents
        if final_latent.dim() == 5 and final_latent.shape[1] == 1:
            final_latent = final_latent.squeeze(1)

        sampled_latents_list.append(final_latent.cpu())

    del base_fused_features, initial_noise, latents
    cleanup_memory()
    return sampled_latents_list


@torch.no_grad()
def decode_and_save(latents_list, save_dir: str, rae, ts: str, device):
    os.makedirs(save_dir, exist_ok=True)

    decoded_imgs = []
    for i, latents in enumerate(latents_list):
        latents = latents.to(device)
        img = rae.decode(latents)
        img_cpu = img[0].cpu()
        decoded_imgs.append(img_cpu)

        out_name = os.path.join(save_dir, f"result_strength_{i}_{ts}.png")
        save_image(img_cpu, out_name)

    if decoded_imgs:
        stitched = torch.cat(decoded_imgs, dim=2)
        stitch_name = os.path.join(save_dir, f"stitched_summary_{ts}.png")
        save_image(stitched, stitch_name)

    cleanup_memory()
    return decoded_imgs


def stage2_process(input_path: str, output_root_dir: str, editing_strengths, total_steps: int, seed=None):
    fused_files = list_pt_files(input_path)
    if not fused_files:
        print(f"No .pt files found from input: {input_path}")
        return

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    print(f"Found {len(fused_files)} file(s): {[os.path.basename(f) for f in fused_files]}")
    os.makedirs(output_root_dir, exist_ok=True)

    manager = ModelManager(device=device)

    # Phase 1
    print("\n[Phase 1] Loading DiT model...")
    dit_model = manager.load_dit()
    if dit_model is None:
        raise RuntimeError("Failed to load DiT model.")

    transport = create_transport(path_type="Linear", prediction="velocity")
    drift_fn = transport.get_drift()

    print(f"[Phase 1] Running diffusion for {len(fused_files)} file(s)...")
    results_buffer = []
    for idx, fused_path in enumerate(fused_files, 1):
        print(f"  -> ({idx}/{len(fused_files)}) {os.path.basename(fused_path)}")
        latents_list = diffuse_one_file(
            fused_path=fused_path,
            editing_strengths=editing_strengths,
            total_steps=total_steps,
            dit_model=dit_model,
            drift_fn=drift_fn,
            device=device,
            dtype=dtype,
        )
        results_buffer.append({"file_path": fused_path, "latents_list": latents_list})

    del dit_model
    cleanup_memory()
    print("[Phase 1] Done.")

    # Phase 2
    print("\n[Phase 2] Loading RAE model...")
    rae = manager.load_rae()
    if rae is None:
        raise RuntimeError("Failed to load RAE model.")

    ts = timestamp_str()
    print(f"[Phase 2] Decoding & saving to: {output_root_dir}")
    for item in results_buffer:
        fused_path = item["file_path"]
        latents_list = item["latents_list"]

        subfolder = folder_name_from_pt(fused_path)
        save_dir = os.path.join(output_root_dir, subfolder)

        print(f"  -> Decoding: {subfolder}")
        decode_and_save(latents_list=latents_list, save_dir=save_dir, rae=rae, ts=ts, device=device)

    del rae
    cleanup_memory()
    print(f"\nAll Done! Results saved in: {output_root_dir}")


def build_argparser():
    p = argparse.ArgumentParser(
        description="Stage2 refactor: supports single .pt file or a directory of .pt files."
    )
    p.add_argument("--input", "-i", required=True, help="Path to a fused .pt file OR a directory containing .pt files")
    p.add_argument("--output", "-o", required=True, help="Output root directory for stage2 results")
    p.add_argument("--steps", type=int, default=50, help="Total steps (keeps your current step scaling by strength)")
    p.add_argument(
        "--strengths",
        nargs="+",
        default=["0.2", "0.3", "0.4", "0.6", "0.8", "0.9", "0.95"],
        help="Editing strengths. e.g. --strengths 0.2 0.3 0.6 OR --strengths 0.2,0.3,0.6",
    )
    p.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    strengths = parse_strengths(args.strengths)
    if strengths is None or len(strengths) == 0:
        raise ValueError("No valid strengths provided.")

    stage2_process(
        input_path=args.input,
        output_root_dir=args.output,
        editing_strengths=strengths,
        total_steps=args.steps,
        seed=args.seed,
    )
