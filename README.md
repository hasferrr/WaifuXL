# ![](https://i.imgur.com/lPsvvh7.png) WaifuXL

[![Upscale Count](https://img.shields.io/endpoint?color=blue&url=https%3A%2F%2Fwaifuxl_upscale_counter.haydnjonest8327.workers.dev%2Fbadge)](waifuxl.com) [![GitHub Super-Linter](https://github.com/TheFutureGadgetsLab/WaifuXL/workflows/Lint%20Code%20Base/badge.svg)](https://github.com/marketplace/actions/super-linter)



# Notice

WaifuXL 1.5 has been released!  
Next we will be working on Desktop GPU/CPU acceleration so users can upscale larger images faster (maybe even videos, depends on the performance we can get out of the model). Some things we're considering for acceleration are PyTorch 2.0 model compilation, ONNX, and AITemplate. When that's available we'll put an announcement on the website.

---

[Check out a full write-up here!](https://haydn.fgl.dev/posts/the-launch-of-waifuxl/)  
WaifuXL is a state of the art super resolution model trained on ~1,200,000 anime style images from the [Danbooru2021](https://www.gwern.net/Danbooru2021) dataset. You can find it online at https://waifuxl.com/. Note that while you _can_ upscale natural (real) images, the model was only trained on anime style drawings, so dont expect to have your socks blown off.

# Comparison to Waifu2x

In general, the [Real-ESRGAN](https://arxiv.org/abs/2107.10833) will outperform the models used on [waifu2x](http://waifu2x.udp.jp/) by a significant margin, without the need for multiple models trained on various noise reduction levels.

![](https://i.imgur.com/BDv4rd9.png)

# How it Works

Using the [Onnx Runtime](https://onnxruntime.ai/), we stream the weights of our ML models directly onto your device to be executed locally via **WebGPU** (for hardware-accelerated discrete/integrated GPU inference) with an automatic fallback to **WebAssembly (WASM)**. Doing so allows us to provide this service solely through a static webpage, no backend for model execution needed. This has the added benefit of enabling the privacy of your images--your images are not, and never will be, sent to us.

## Fork Modifications & Enhancements

This repository includes modifications by **hasferrr** (2026) licensed under the [Apache License 2.0](LICENSE) (see [NOTICE](NOTICE) for full details):

- **WebGPU Acceleration**: High-performance WebGPU execution provider in `onnxruntime-web` with automatic graceful fallback to multi-threaded WASM.
- **Hardware Accelerator Badge**: Live sidebar badge and progress status indicating whether inference is running on **WebGPU** or **WASM (CPU)**.
- **Configurable Model URLs**: Support for overriding ONNX model URLs (`tagger.onnx` and `superRes.onnx`) via `NEXT_PUBLIC_TAGGER_MODEL_URL` and `NEXT_PUBLIC_SUPER_RES_MODEL_URL` environment variables.
- **Desktop NVIDIA CUDA CLI**: Native Python CLI (`scripts/upscale_cuda.py`) for local NVIDIA RTX / CUDA GPU super-resolution and image tagging.

## Models

For our super resolution network we are using the state-of-the-art [Real-ESRGAN](https://arxiv.org/abs/2107.10833) and for our image tagging network we are using a [MobileNetV3](https://arxiv.org/abs/1905.02244). Both were trained on a subset of [Danbooru2021](https://www.gwern.net/Danbooru2021).

## Site

[Onnx Runtime](https://onnxruntime.ai/) supports WebGPU as well as multithreaded SIMD WebAssembly--while upscaling on a phone or a laptop is surprisingly quick, using a beefier computer or discrete GPU will bring noticeable benefits. We're hosted on [Cloudflare Pages](https://pages.cloudflare.com/) which provides unlimited bandwidth. The site is written in React with Next.js and Material UI.

# Running Locally

If you'd like to run the web app locally with [Bun](https://bun.sh/):

```bash
git clone https://github.com/hasferrr/WaifuXL
cd WaifuXL
bun install
bun run dev
```

### Custom Model URLs (Optional)

By default, the web app loads models from `/models/tagger.onnx` and `/models/superRes.onnx` (`public/models/`). To load them from an external CDN or custom URL, copy `.env.example` to `.env` and set:

```env
NEXT_PUBLIC_TAGGER_MODEL_URL=https://example.com/tagger.onnx
NEXT_PUBLIC_SUPER_RES_MODEL_URL=https://example.com/superRes.onnx
```

### Desktop NVIDIA CUDA CLI

To upscale images directly on your desktop using an NVIDIA GPU (CUDA) with automatic CPU fallback:

```bash
# Setup Python virtual environment and install dependencies
python3 -m venv .venv
.venv/bin/pip install -r scripts/requirements.txt

# Run 2x upscale (default) with image tagging via Bun
bun run upscale:cuda -- path/to/image.png

# Custom scale factor (2, 4, or 8) and output path
bun run upscale:cuda -- -i path/to/image.png -o path/to/output.png -s 4
```

A few notes:

- Multithreading in WASM requires `SharedArrayBuffer` because of [Spectre](<https://en.wikipedia.org/wiki/Spectre_(security_vulnerability)>). If WebGPU is unavailable, you can ensure WASM multithreading works locally by launching Chrome from the command line with `--enable-features=SharedArrayBuffer`.

# Contribution Guide

## Code / Site

We're definitely open to code contributions, whether it be code cleanup, new features, or bugfixes. Simply open a discussion so we all can collaborate and discuss the merit of your ideas!

## Ideas

We have plenty of things we'd like to add to [WaifuXL](https://waifuxl.com/), if you have a suggestion simply open a discussion and we can start talking! Here are a few things we have in mind:

- [Style Transfer](https://en.wikipedia.org/wiki/Neural_style_transfer) so you can apply an anime style to real images.
- Easter eggs. We have a tagger that can match thousands of tags, maybe we can have something like a cool fireworks effect if you upscale a top-tier waifu (i.e. Truck-kun).

## Donations

We're open to donations, just head to https://waifuxl.com/donate and see the ways you can contribute. We want to make it clear that we are providing this service to you, free of cost, because it is free of cost to _us_. We have no backend and [Cloudflare Pages](https://pages.cloudflare.com/) provides unlimited bandwidth for free. Please don't feel obligated to donate even if you find yourself using this service frequently.

On top of donations, we're open to compute contributions (GPU's). We'd like to train a better tagger and continue to update the super resolution model as new SOTA models are published. We'd also like to train a model on natural (real images) so we can upscale more than drawings, and maybe a style transfer model. All of this takes a lot of compute which we simply dont have. If you have the means and you're feeling generous, drop us a line.
