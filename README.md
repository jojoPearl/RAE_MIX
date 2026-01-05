# Semantic Object Insertion & Generation

This project implements a pipeline for semantically inserting objects into background scenes using feature-space fusion and diffusion-based generation. It leverages **CLIPSeg** for semantic masking, **RAE (Regularized Autoencoder)** for feature compression, and **DiT (Diffusion Transformer)** with Flow Matching for high-quality image generation.

## 📂 Directory Structure: `src/semantic/`

The core logic resides in the `src/semantic` directory. Below is a detailed description of each module:

### 1. `segmentation.py`

Handles all CLIP-based semantic understanding and mask generation.

* **`extract_semantic_mask_with_clipseg`**: Uses CLIPSeg to generate a binary mask of the target object based on a text prompt (e.g., "rabbit").
* **`get_text_guided_coords`**: Calculates the optimal `(x, y)` coordinates to place an object within a specific region (ROI) by searching for the highest activation of a text prompt (e.g., "grass").

### 2. `resize.py`

Implements the dual-stage resizing strategy to ensure objects maintain high fidelity at different scales.

* **M1 Scaling (Pixel Space)**: `apply_m1_scaling` resizes and pads the raw object image *before* it enters the RAE encoder. This ensures the encoder receives a standardized input size.
* **M2 Scaling (Latent Space)**: `apply_m2_latent_scaling` performs interpolation in the latent feature space to fine-tune the object's size relative to the background features.

### 3. `utils.py`

General utility functions for image processing and coordinate management.

* **`load_and_transform`**: Standardizes image loading, resizing (Lanczos), and tensor conversion.
* **`cleanup_memory`**: Manages GPU VRAM by clearing caches and garbage collection.

### 4. `modelManager.py`

A centralized class to manage the lifecycle of heavy deep learning models.

* Loads **RAE**, **DiT**, and **CLIP/CLIPSeg** models on demand.
* Ensures models are moved to the correct `DEVICE` (CUDA/CPU) and `DTYPE` (Float16/BFloat16) to optimize memory usage.

### 5. `config.py`

Global configuration settings.

* Defines system constants such as `DEVICE`, `DTYPE`, and paths to model checkpoints.

---

## 🚀 Usage Pipeline

The generation process is split into two stages to allow for feature manipulation before generation.

### Stage 1: Feature Extraction & Fusion (`stage1.py`)

This step processes the base image and the object image, extracts their features using RAE, performs semantic resizing/masking, and fuses them into a single feature map.

**Key Features:**

* **Smart Placement**: Automatically finds the best spot for the object (e.g., "on the grass") within a restricted target area (e.g., "bottom_right").
* **Background Removal**: Uses CLIPSeg mask to remove the background of the inserted object before fusion.

**Example:**

```bash
python stage1.py \
  --base_image "assets/group1/base.png" \
  --replace_image "assets/group1/r.png" \
  --output_dir "assets/group1/stage1_result" \
  --target_text "dog" \
  --target_area "bottom_right" \
  --location_prompt "grass central" \
  --scale_factor 1.0

```

### Stage 2: Diffusion Generation (`stage2.py`)

This step takes the fused features (`.pt` file) from Stage 1 and uses a Diffusion Transformer (DiT) to generate the final photorealistic image.

**Key Features:**

* **Custom Transport Sampling**: Uses a manually controlled Euler loop to solve the ODE.
* **Time-Step Control**: Allows starting the generation from a specific noise level (`t < 1.0`) to preserve the structure of the fused features while harmonizing lighting and texture.

**Example:**

```bash
python stage2.py \
  --output_dir "assets/group1/"

```

### Batch Execution

To run multiple jobs sequentially, use the provided shell script:

```bash
chmod +x run_generate.sh
./run_generate.sh

```

---
