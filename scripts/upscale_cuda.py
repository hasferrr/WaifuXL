#!/usr/bin/env python3
# Copyright 2026 hasferrr
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
WaifuXL Desktop Super-Resolution & Image Tagger CLI
Accelerated via NVIDIA CUDA Tensor Cores with automatic fallback.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Auto-switch to repository virtual environment if available and not already active
_repo_venv_python = Path(__file__).resolve().parent.parent / '.venv' / 'bin' / 'python'
if (
    _repo_venv_python.is_file()
    and sys.executable != str(_repo_venv_python)
    and not os.environ.get('_WAIFUXL_VENV_SWITCHED')
):
    _env = os.environ.copy()
    _env['_WAIFUXL_VENV_SWITCHED'] = '1'
    os.execve(str(_repo_venv_python), [str(_repo_venv_python)] + sys.argv, _env)

# Auto-configure LD_LIBRARY_PATH with all nvidia package lib paths and re-exec if needed
def _ensure_nvidia_ld_path() -> None:
    try:
        import nvidia
        needed_dirs: list[str] = []
        for p in getattr(nvidia, '__path__', []):
            for root, _, files in os.walk(p):
                if 'lib' in root and any(f.endswith('.so') or '.so.' in f for f in files):
                    needed_dirs.append(root)

        if needed_dirs:
            current_ld = os.environ.get('LD_LIBRARY_PATH', '')
            current_parts = set(current_ld.split(':')) if current_ld else set()
            missing = [d for d in needed_dirs if d not in current_parts]
            if missing and not os.environ.get('_WAIFUXL_REEXECED'):
                new_ld = ':'.join(missing + ([current_ld] if current_ld else []))
                env = os.environ.copy()
                env['LD_LIBRARY_PATH'] = new_ld
                env['_WAIFUXL_REEXECED'] = '1'
                os.execve(sys.executable, [sys.executable] + sys.argv, env)
    except Exception:
        pass

_ensure_nvidia_ld_path()

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
import onnxruntime as ort

# Silence informational and warning logs from ONNX Runtime (e.g. conv fallback notices)
ort.set_default_logger_severity(3)

# Constants matching WaifuXL web inference configuration
CHUNK_SIZE = 256
PAD_SIZE = 32
TAG_INDICES = {
    'DESCRIPTORS_START': 0,
    'DESCRIPTORS_END': 2000,
    'CHARACTERS_START': 2000,
    'CHARACTERS_END': 4000,
    'RATINGS_START': 4000,
    'RATINGS_END': 4003,
}


def find_project_root() -> Path:
    """Find the root directory of the WaifuXL project."""
    current = Path(__file__).resolve().parent
    for candidate in [current, current.parent]:
        if (candidate / 'public' / 'models' / 'superRes.onnx').is_file():
            return candidate
    return current.parent


def get_execution_providers(device_choice: str = 'auto') -> list[str | tuple[str, dict]]:
    """Determine available ONNX Runtime execution providers."""
    available = ort.get_available_providers()
    cuda_available = 'CUDAExecutionProvider' in available

    if device_choice.lower() == 'cpu':
        return ['CPUExecutionProvider']

    providers: list[str | tuple[str, dict]] = []
    if cuda_available and device_choice.lower() in ('auto', 'cuda'):
        cuda_options = {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 6 * 1024 * 1024 * 1024,  # 6GB cap to avoid OOM
            'cudnn_conv_algo_search': 'DEFAULT',
            'do_copy_in_default_stream': True,
        }
        providers.append(('CUDAExecutionProvider', cuda_options))

    providers.append('CPUExecutionProvider')
    return providers


def create_session(model_path: Path, providers: list[str | tuple[str, dict]]) -> ort.InferenceSession:
    """Create an ONNX Runtime inference session with session options."""
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.log_severity_level = 3  # Error only to avoid conv fallback spam

    try:
        session = ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)
        return session
    except Exception as e:
        print(f"[Warning] Failed to initialize requested provider ({providers[0]}): {e}", file=sys.stderr)
        print("[Info] Falling back to CPUExecutionProvider...", file=sys.stderr)
        return ort.InferenceSession(str(model_path), sess_options=opts, providers=['CPUExecutionProvider'])


def load_tags(tags_path: Path) -> list[str]:
    """Load tagger vocabulary from tags.json."""
    with open(tags_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return [item[1] for item in data]


def run_tagger(session: ort.InferenceSession, img: Image.Image, tags: list[str]) -> dict[str, list[dict]]:
    """
    Run tagger model on the input image.
    Expects input shape [1, 3, H, W] uint8 RGB.
    """
    rgb_img = img.convert('RGB')
    img_arr = np.array(rgb_img, dtype=np.uint8)  # [H, W, 3]
    tensor_input = np.transpose(img_arr, (2, 0, 1))  # [3, H, W]
    tensor_input = np.expand_dims(tensor_input, axis=0)  # [1, 3, H, W]

    output = session.run(['output'], {'input': tensor_input})[0]  # [1, 4003]
    probs = output[0]

    def get_top_k(start: int, end: int, k: int = 5) -> list[dict]:
        slice_probs = probs[start:end]
        indices = np.argsort(slice_probs)[::-1][:k]
        return [{'name': tags[start + idx], 'prob': float(slice_probs[idx])} for idx in indices]

    return {
        'topDesc': get_top_k(TAG_INDICES['DESCRIPTORS_START'], TAG_INDICES['DESCRIPTORS_END'], 6),
        'topChars': get_top_k(TAG_INDICES['CHARACTERS_START'], TAG_INDICES['CHARACTERS_END'], 3),
        'rating': get_top_k(TAG_INDICES['RATINGS_START'], TAG_INDICES['RATINGS_END'], 3),
    }


def upscale_frame_tiled(
    session: ort.InferenceSession,
    image_array: np.ndarray,
    chunk_size: int = CHUNK_SIZE,
    pad_size: int = PAD_SIZE,
    pass_num: int = 1,
    total_passes: int = 1,
) -> np.ndarray:
    """
    Upscales one frame by a factor of 2x using tiled chunking with overlap padding.
    `image_array` has shape [W, H, 4] uint8 (transposed from PIL).
    """
    in_w, in_h = image_array.shape[:2]
    out_w, out_h = in_w * 2, in_h * 2

    num_chunks_w = math.ceil(in_w / chunk_size)
    num_chunks_h = math.ceil(in_h / chunk_size)
    chunk_w = math.floor(in_w / num_chunks_w)
    chunk_h = math.floor(in_h / num_chunks_h)

    total_chunks = num_chunks_w * num_chunks_h
    out_arr = np.zeros((out_w, out_h, 4), dtype=np.uint8)

    desc = f"Upscaling Pass {pass_num}/{total_passes} (2x)"
    with tqdm(total=total_chunks, desc=desc, unit="tile") as pbar:
        for r in range(num_chunks_h):
            for c in range(num_chunks_w):
                x = c * chunk_w
                y = r * chunk_h
                x_start = max(0, x - pad_size)
                y_start = max(0, y - pad_size)

                in_chunk_w = min(x_start + chunk_w + pad_size * 2, in_w) - x_start
                in_chunk_h = min(y_start + chunk_h + pad_size * 2, in_h) - y_start
                out_chunk_w = 2 * (min(in_w, x + chunk_w) - x)
                out_chunk_h = 2 * (min(in_h, y + chunk_h) - y)

                sub_arr = image_array[x_start : x_start + in_chunk_w, y_start : y_start + in_chunk_h, :]

                # Run super-resolution on chunk
                chunk_output = session.run(['output'], {'input': sub_arr})[0]

                # Crop padding
                crop_x = (x - x_start) * 2
                crop_y = (y - y_start) * 2
                cropped = chunk_output[crop_x : crop_x + out_chunk_w, crop_y : crop_y + out_chunk_h, :]

                out_arr[x * 2 : x * 2 + out_chunk_w, y * 2 : y * 2 + out_chunk_h, :] = cropped
                pbar.update(1)

    return out_arr


def upscale_image(
    session: ort.InferenceSession,
    img: Image.Image,
    scale_factor: int,
    chunk_size: int = CHUNK_SIZE,
    pad_size: int = PAD_SIZE,
) -> Image.Image:
    """
    Multi-pass upscale (2x, 4x, or 8x).
    """
    num_passes = int(math.log2(scale_factor))
    if 2**num_passes != scale_factor:
        raise ValueError(f"Invalid scale factor {scale_factor}. Must be 2, 4, or 8.")

    rgba_img = img.convert('RGBA')
    raw_arr = np.array(rgba_img, dtype=np.uint8)  # [H, W, 4]
    current_arr = np.transpose(raw_arr, (1, 0, 2))  # [W, H, 4]

    for p in range(num_passes):
        current_arr = upscale_frame_tiled(
            session=session,
            image_array=current_arr,
            chunk_size=chunk_size,
            pad_size=pad_size,
            pass_num=p + 1,
            total_passes=num_passes,
        )

    # Transpose back to [H, W, 4] for PIL
    final_arr = np.transpose(current_arr, (1, 0, 2))
    return Image.fromarray(final_arr, mode='RGBA')


def resolve_file_path(p: str | Path | None) -> Path | None:
    """Resolve file path, translating Windows drive paths (e.g. C:\\...) to WSL (/mnt/c/...)."""
    if p is None:
        return None
    p_str = str(p).strip('\'"')
    if not p_str:
        return None

    # Handle Windows paths like C:\... or D:\... on Linux/WSL
    if len(p_str) >= 2 and p_str[1] == ':' and p_str[0].isalpha():
        drive = p_str[0].lower()
        rest = p_str[2:].replace('\\', '/').lstrip('/')
        return Path(f"/mnt/{drive}/{rest}")

    # Handle Windows backslashes in relative paths
    p_str = p_str.replace('\\', '/')
    return Path(p_str).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WaifuXL Desktop Super-Resolution CLI - NVIDIA RTX GPU Accelerated",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('file', nargs='?', default=None, help="Input image file path (positional)")
    parser.add_argument('-i', '--input', default=None, help="Input image file path")
    parser.add_argument('-o', '--output', default=None, help="Output image file path")
    parser.add_argument('-s', '--scale', type=int, choices=[2, 4, 8], default=2, help="Upscale factor (2x, 4x, 8x)")
    parser.add_argument('-d', '--device', choices=['auto', 'cuda', 'cpu'], default='auto', help="Execution provider")
    parser.add_argument('-t', '--tile-size', type=int, default=CHUNK_SIZE, help="Tile chunk size for inference")
    parser.add_argument('-p', '--pad-size', type=int, default=PAD_SIZE, help="Padding size around each tile")
    parser.add_argument('--models-dir', type=Path, default=None, help="Directory containing ONNX models and tags.json")
    parser.add_argument('--no-tag', action='store_true', help="Skip running the image tagger")

    args = parser.parse_args()

    raw_input = args.input or args.file
    if not raw_input:
        parser.print_help()
        sys.exit(1)

    input_path = resolve_file_path(raw_input)
    if not input_path or not input_path.is_file():
        print(f"Error: Input file does not exist: {raw_input}", file=sys.stderr)
        sys.exit(1)

    project_root = find_project_root()
    models_dir = args.models_dir or (project_root / 'public' / 'models')
    tags_file = (project_root / 'public' / 'tags.json') if not args.models_dir else (args.models_dir / 'tags.json')

    super_res_path = models_dir / 'superRes.onnx'
    tagger_path = models_dir / 'tagger.onnx'

    if not super_res_path.is_file():
        print(f"Error: Super-resolution model not found at {super_res_path}", file=sys.stderr)
        sys.exit(1)

    # Determine output path
    if args.output is None:
        input_stem = input_path.stem
        output_path = input_path.parent / f"{input_stem}_{args.scale}x.png"
    else:
        output_path = resolve_file_path(args.output)
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_{args.scale}x.png"

    print("=" * 60)
    print(" WaifuXL Desktop Super-Resolution (NVIDIA RTX GPU / CUDA)")
    print("=" * 60)
    print(f"Input:       {input_path}")
    print(f"Output:      {output_path}")
    print(f"Scale:       {args.scale}x")
    print(f"Tile Size:   {args.tile_size}px (Padding: {args.pad_size}px)")

    providers = get_execution_providers(args.device)

    # Load input image
    t_start = time.perf_counter()
    img = Image.open(input_path)
    orig_w, orig_h = img.size
    print(f"Dimensions:  {orig_w}x{orig_h} -> {orig_w * args.scale}x{orig_h * args.scale}")

    # Initialize Super-Resolution Session
    print("\n[1/3] Initializing Super-Resolution Model...")
    super_session = create_session(super_res_path, providers)
    active_providers = super_session.get_providers()
    active_ep = active_providers[0] if active_providers else 'Unknown'
    is_cuda = 'CUDAExecutionProvider' in active_ep
    print(f"Active Provider: {'⚡ ' if is_cuda else '💻 '}{active_ep}")

    # Run Tagger (if requested and available)
    if not args.no_tag and tagger_path.is_file() and tags_file.is_file():
        print("\n[2/3] Analyzing Image Tags...")
        try:
            tagger_session = create_session(tagger_path, providers)
            tags_vocab = load_tags(tags_file)
            tag_results = run_tagger(tagger_session, img, tags_vocab)

            if tag_results['topChars']:
                char_str = ", ".join(f"{t['name']} ({t['prob']:.1%})" for t in tag_results['topChars'] if t['prob'] > 0.05)
                if char_str:
                    print(f"  • Top Characters:  {char_str}")

            if tag_results['topDesc']:
                desc_str = ", ".join(f"{t['name']} ({t['prob']:.1%})" for t in tag_results['topDesc'][:5])
                print(f"  • Top Descriptors: {desc_str}")

            if tag_results['rating']:
                rating_str = ", ".join(f"{t['name']} ({t['prob']:.1%})" for t in tag_results['rating'])
                print(f"  • Rating:          {rating_str}")
        except Exception as e:
            print(f"  [Notice] Tagger failed: {e}", file=sys.stderr)
    else:
        print("\n[2/3] Skipping Image Tagging.")

    # Run Super-Resolution
    print(f"\n[3/3] Running {args.scale}x Upscaling...")
    upscaled_img = upscale_image(
        session=super_session,
        img=img,
        scale_factor=args.scale,
        chunk_size=args.tile_size,
        pad_size=args.pad_size,
    )

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    upscaled_img.save(str(output_path), format='PNG', optimize=True)
    elapsed = time.perf_counter() - t_start

    print("\n" + "=" * 60)
    print(f"✓ Upscale Complete in {elapsed:.2f}s!")
    print(f"Saved: {output_path} ({upscaled_img.width}x{upscaled_img.height})")
    print("=" * 60)


if __name__ == '__main__':
    main()
