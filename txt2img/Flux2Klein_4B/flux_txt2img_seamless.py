"""FLUX.2 Klein text-to-image with circular decoder padding and latent rolling.

This publication-ready script applies:
1) Circular padding patch to all Conv2d layers in the VAE decoder.
2) Per-step latent rolling during denoising.

Outputs:
- Generated image.
- 2x2 tiled preview for seam inspection.
"""

import argparse
import types
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import Flux2KleinPipeline
from PIL import Image


DEFAULTS = {
    "prompt": "seamless tileable texture, high detail, coherent edges",
    "model_dir": "./Model",
    "output": "./outputs/flux-klein-txt2img-seamless.png",
    "tile_output": "./outputs/flux-klein-txt2img-seamless-2x2.png",
    "height": 1024,
    "width": 1024,
    "steps": 30,
    "guidance_scale": 1.0,
    "seed": 0,
    "roll_shift_y": 16,
    "roll_shift_x": 16,
    "debug": False,
}


def make_tile_2x2(image: Image.Image) -> Image.Image:
    """Create a 2x2 tiled preview image."""
    width, height = image.size
    tiled = Image.new("RGB", (width * 2, height * 2))
    tiled.paste(image, (0, 0))
    tiled.paste(image, (width, 0))
    tiled.paste(image, (0, height))
    tiled.paste(image, (width, height))
    return tiled


def infer_grid_from_token_length(length: int) -> tuple[int, int]:
    """Infer token grid shape (H, W) from flattened token length."""
    grid_h = int(np.sqrt(length))
    while grid_h > 1 and length % grid_h != 0:
        grid_h -= 1
    grid_w = max(1, length // max(1, grid_h))
    return grid_h, grid_w


def _conv2d_forward_circular(self: nn.Conv2d, input_tensor: torch.Tensor) -> torch.Tensor:
    """Conv2d forward replacement with explicit circular padding."""
    ph, pw = self._original_padding  # type: ignore[attr-defined]
    x = input_tensor
    if ph > 0 or pw > 0:
        x = F.pad(x, (pw, pw, ph, ph), mode="circular")
    return F.conv2d(x, self.weight, self.bias, self.stride, padding=0, dilation=self.dilation, groups=self.groups)


def patch_circular_padding(module: nn.Module) -> tuple[dict[str, Callable], dict[str, int]]:
    """Patch all Conv2d layers in `module` to use circular padding."""
    original_forwards: dict[str, Callable] = {}
    stats = {"conv2d_total": 0, "conv2d_patched": 0}

    for name, layer in module.named_modules():
        if not isinstance(layer, nn.Conv2d):
            continue
        stats["conv2d_total"] += 1
        padding = layer.padding if isinstance(layer.padding, tuple) else (layer.padding, layer.padding)
        ph, pw = int(padding[0]), int(padding[1])
        original_forwards[name] = layer.forward
        layer._original_padding = (ph, pw)  # type: ignore[attr-defined]
        layer.forward = types.MethodType(_conv2d_forward_circular, layer)
        stats["conv2d_patched"] += 1

    return original_forwards, stats


def restore_patched_layers(module: nn.Module, original_forwards: dict[str, Callable]) -> None:
    """Restore patched layer forwards to their original implementations."""
    modules_by_name = dict(module.named_modules())
    for name, original_forward in original_forwards.items():
        if name not in modules_by_name:
            continue
        modules_by_name[name].forward = original_forward
        if hasattr(modules_by_name[name], "_original_padding"):
            delattr(modules_by_name[name], "_original_padding")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FLUX.2 Klein text-to-image with decoder circular padding and per-step latent rolling."
    )
    parser.add_argument("--prompt", type=str, default=DEFAULTS["prompt"], help="Text prompt.")
    parser.add_argument("--model-dir", type=str, default=DEFAULTS["model_dir"], help="Local model directory.")
    parser.add_argument("--output", type=str, default=DEFAULTS["output"], help="Generated image output path.")
    parser.add_argument("--tile-output", type=str, default=DEFAULTS["tile_output"], help="2x2 tiled preview output path.")
    parser.add_argument("--height", type=int, default=DEFAULTS["height"], help="Output height.")
    parser.add_argument("--width", type=int, default=DEFAULTS["width"], help="Output width.")
    parser.add_argument("--steps", type=int, default=DEFAULTS["steps"], help="Denoising steps.")
    parser.add_argument("--guidance-scale", type=float, default=DEFAULTS["guidance_scale"], help="CFG scale.")
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"], help="Random seed.")
    parser.add_argument("--roll-shift-y", type=int, default=DEFAULTS["roll_shift_y"], help="Latent roll shift on Y axis.")
    parser.add_argument("--roll-shift-x", type=int, default=DEFAULTS["roll_shift_x"], help="Latent roll shift on X axis.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logs.")
    parser.set_defaults(debug=DEFAULTS["debug"])
    return parser.parse_args()


def ensure_output_parent(path: str) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()

    pipeline = Flux2KleinPipeline.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16, local_files_only=True)
    pipeline.enable_model_cpu_offload()

    original_forwards, patch_stats = patch_circular_padding(pipeline.vae.decoder)
    if args.debug:
        print(
            "[debug] decoder Conv2d circular patched: "
            f"{patch_stats['conv2d_patched']}/{patch_stats['conv2d_total']}"
        )

    def latent_roll_callback(_pipe, step, _t, callback_kwargs):
        latents = callback_kwargs["latents"]
        batch_size, seq_len, channels = latents.shape
        grid_h, grid_w = infer_grid_from_token_length(seq_len)

        if grid_h * grid_w == seq_len:
            latents = torch.roll(
                latents.view(batch_size, grid_h, grid_w, channels),
                shifts=(args.roll_shift_y, args.roll_shift_x),
                dims=(1, 2),
            ).view(batch_size, seq_len, channels)

        if args.debug and step == 0:
            print(
                f"[debug] latent rolling enabled: shape={tuple(callback_kwargs['latents'].shape)}, "
                f"grid=({grid_h},{grid_w}), shift=({args.roll_shift_y},{args.roll_shift_x})"
            )
        return {"latents": latents}

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    try:
        output_image = pipeline(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            callback_on_step_end=latent_roll_callback,
            callback_on_step_end_tensor_inputs=["latents"],
        ).images[0]
    finally:
        restore_patched_layers(pipeline.vae.decoder, original_forwards)

    ensure_output_parent(args.output)
    ensure_output_parent(args.tile_output)
    output_image.save(args.output)
    make_tile_2x2(output_image).save(args.tile_output)

    print(f"[ok] image saved: {Path(args.output).expanduser().resolve()}")
    print(f"[ok] 2x2 tile saved: {Path(args.tile_output).expanduser().resolve()}")


if __name__ == "__main__":
    main()
