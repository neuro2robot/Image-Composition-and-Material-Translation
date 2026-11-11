"""Standalone CLI for masked_clip_direction_audit.py.

This loads the same SDXL IP-Adapter components but does not generate images.
It only encodes the existing edited/mask folders and writes audit reports.
"""

import argparse
from pathlib import Path

import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetInpaintPipeline
from ip_adapter.ip_adapter2 import IPAdapterXL

from masked_clip_direction_audit import audit_output_tree


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit existing masked property sequences in CLIP/IP-Adapter space."
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Folder containing object folders such as bomotor_1.",
    )
    parser.add_argument(
        "--report-dir",
        default=None,
        help="Report destination; defaults to OUTPUT_ROOT/_direction_audit.",
    )
    parser.add_argument(
        "--base-model",
        default="SG161222/RealVisXL_V4.0",
    )
    parser.add_argument(
        "--controlnet",
        default="diffusers/controlnet-depth-sdxl-1.0",
    )
    parser.add_argument(
        "--image-encoder",
        default="models/image_encoder",
    )
    parser.add_argument(
        "--ip-ckpt",
        default="sdxl_models/ip-adapter_sdxl_vit-h.bin",
    )
    parser.add_argument(
        "--directions",
        default=None,
        help=(
            "Optional .pt file containing {property_name: direction_tensor}. "
            "Without it, output trajectories are still audited, but learned-"
            "direction alignment columns are NaN."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def load_property_directions(path):
    if path is None:
        return None

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Direction file does not exist: {path}")

    try:
        directions = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        directions = torch.load(path, map_location="cpu")

    if not isinstance(directions, dict):
        raise TypeError("--directions must contain a dictionary of tensors")

    cleaned = {}
    for property_name, direction in directions.items():
        if direction is None:
            cleaned[property_name] = None
        elif torch.is_tensor(direction):
            cleaned[property_name] = direction.detach().cpu().float()
        else:
            raise TypeError(
                f"Direction for {property_name!r} is not a tensor or None"
            )
    return cleaned


def build_ip_adapter(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")

    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print("Loading ControlNet...")
    controlnet = ControlNetModel.from_pretrained(
        args.controlnet,
        variant="fp16" if dtype == torch.float16 else None,
        use_safetensors=True,
        torch_dtype=dtype,
    ).to(device)

    print("Loading SDXL pipeline...")
    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        args.base_model,
        controlnet=controlnet,
        use_safetensors=True,
        torch_dtype=dtype,
        add_watermarker=False,
    ).to(device)

    print("Loading IP-Adapter image encoder and projector...")
    ip_model = IPAdapterXL(
        pipe,
        args.image_encoder,
        args.ip_ckpt,
        device,
    )

    # Diffusion is not used by the audit. Move it out of VRAM after IP-Adapter
    # has installed its weights, while keeping the two required modules active.
    ip_model.pipe.to("cpu")
    ip_model.image_encoder.to(device=device, dtype=dtype).eval()
    ip_model.image_proj_model.to(device=device, dtype=dtype).eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return ip_model, device


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    if not output_root.is_dir():
        raise FileNotFoundError(
            f"Output root does not exist: {output_root.resolve()}"
        )

    property_directions = load_property_directions(args.directions)
    ip_model, device = build_ip_adapter(args)

    audit_output_tree(
        ip_model=ip_model,
        output_root=output_root,
        property_directions=property_directions,
        report_dir=args.report_dir,
        device=device,
    )

    ip_model.image_encoder.to("cpu")
    ip_model.image_proj_model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
