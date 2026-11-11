"""Build global property directions from existing masked output trajectories.

This is a self-distillation diagnostic: each per-object endpoint direction is

    masked_CLIP(last_frame) - masked_CLIP(first_frame)

The directions are normalized before averaging so that one object cannot
dominate merely because it changed more. Leave-one-object-out cosine checks
whether the direction transfers across objects. Inconsistent properties are
saved as None rather than silently used.

This does not generate images.
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from masked_clip_direction_audit import (
    PROPERTY_NAMES,
    _encode_and_project,
    _image_files,
    _load_masked_reference,
    _match_images_and_masks,
    _natural_key,
    _trajectory_metrics,
)
from run_masked_clip_audit import build_ip_adapter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build cross-object CLIP property directions from existing outputs."
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Folder containing boardpin_1, bomotor_1, and other objects.",
    )
    parser.add_argument(
        "--save-path",
        default=None,
        help="Defaults to OUTPUT_ROOT/empirical_property_directions.pt.",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="Defaults to OUTPUT_ROOT/_direction_audit/empirical_direction_report.csv.",
    )
    parser.add_argument(
        "--minimum-examples",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--minimum-path-quality",
        type=float,
        default=0.20,
        help="Minimum efficiency * monotonicity * positive progress correlation.",
    )
    parser.add_argument(
        "--minimum-median-loo-cosine",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--minimum-positive-loo-fraction",
        type=float,
        default=0.60,
    )
    parser.add_argument(
        "--saved-magnitude-scale",
        type=float,
        default=1.0,
        help=(
            "Multiplier applied to the median empirical endpoint norm. Keep 1.0 "
            "here and control strength later with DIRECTION_ENDPOINT_SCALE."
        ),
    )
    parser.add_argument("--base-model", default="SG161222/RealVisXL_V4.0")
    parser.add_argument(
        "--controlnet", default="diffusers/controlnet-depth-sdxl-1.0"
    )
    parser.add_argument("--image-encoder", default="models/image_encoder")
    parser.add_argument(
        "--ip-ckpt", default="sdxl_models/ip-adapter_sdxl_vit-h.bin"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def cosine(first, second):
    first = first.reshape(1, -1).float()
    second = second.reshape(1, -1).float()
    if float(first.norm()) < 1e-12 or float(second.norm()) < 1e-12:
        return float("nan")
    return float(F.cosine_similarity(first, second, dim=-1).item())


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def collect_endpoint_directions(ip_model, output_root, device, minimum_quality):
    output_root = Path(output_root)
    records = {property_name: [] for property_name in PROPERTY_NAMES}

    object_dirs = sorted(
        [path for path in output_root.iterdir() if path.is_dir()],
        key=_natural_key,
    )

    for object_dir in object_dirs:
        if object_dir.name.startswith("_"):
            continue

        print(f"\nOBJECT: {object_dir.name}")
        for property_name in PROPERTY_NAMES:
            edited_dir = object_dir / property_name / "edited"
            mask_dir = object_dir / property_name / "mask"
            if not edited_dir.is_dir() or not mask_dir.is_dir():
                continue

            pairs = _match_images_and_masks(edited_dir, mask_dir)
            if len(pairs) < 2:
                print(f"  {property_name}: skipped; fewer than two frames")
                continue

            clip_vectors = []
            for image_path, mask_path in pairs:
                reference, _ = _load_masked_reference(image_path, mask_path)
                clip_embedding, _ = _encode_and_project(
                    ip_model, reference, device
                )
                clip_vectors.append(clip_embedding.reshape(-1))

            clip_vectors = torch.stack(clip_vectors).float()
            _, summary, endpoint_direction = _trajectory_metrics(clip_vectors)

            progress = max(0.0, summary["progress_correlation"])
            quality = (
                summary["path_efficiency"]
                * summary["monotonic_fraction"]
                * progress
            )
            endpoint_norm = float(endpoint_direction.norm())

            print(
                f"  {property_name}: quality={quality:.3f}, "
                f"endpoint_relative={summary['endpoint_relative_change']:.3%}"
            )

            if endpoint_norm < 1e-8 or quality < minimum_quality:
                print("    excluded from global direction")
                continue

            records[property_name].append(
                {
                    "object_name": object_dir.name,
                    "direction": endpoint_direction.cpu(),
                    "endpoint_norm": endpoint_norm,
                    "quality": quality,
                    **summary,
                }
            )

    return records


def build_global_directions(records, args):
    saved_directions = {}
    report_rows = []

    for property_name in PROPERTY_NAMES:
        property_records = records[property_name]
        count = len(property_records)
        if count < args.minimum_examples:
            saved_directions[property_name] = None
            print(
                f"\n{property_name}: REJECTED; only {count} quality examples"
            )
            continue

        raw_directions = torch.stack(
            [record["direction"].reshape(-1) for record in property_records]
        ).float()
        unit_directions = F.normalize(raw_directions, dim=-1)
        weights = torch.tensor(
            [record["quality"] for record in property_records],
            dtype=torch.float32,
        )
        weights = weights / weights.sum().clamp_min(1e-12)

        weighted_mean = (weights[:, None] * unit_directions).sum(
            dim=0, keepdim=True
        )
        cross_object_coherence = float(weighted_mean.norm())
        global_unit = F.normalize(weighted_mean, dim=-1)

        loo_scores = []
        for index in range(count):
            keep = torch.arange(count) != index
            remaining_weights = weights[keep]
            remaining_weights = remaining_weights / remaining_weights.sum().clamp_min(
                1e-12
            )
            loo_mean = (
                remaining_weights[:, None] * unit_directions[keep]
            ).sum(dim=0, keepdim=True)
            loo_unit = F.normalize(loo_mean, dim=-1)
            loo_scores.append(cosine(unit_directions[index], loo_unit))

        loo_array = np.asarray(loo_scores, dtype=np.float32)
        median_loo = float(np.median(loo_array))
        mean_loo = float(np.mean(loo_array))
        positive_fraction = float(np.mean(loo_array > 0.0))
        above_point_one_fraction = float(np.mean(loo_array >= 0.10))

        accepted = (
            median_loo >= args.minimum_median_loo_cosine
            and positive_fraction >= args.minimum_positive_loo_fraction
        )

        median_norm = float(torch.median(raw_directions.norm(dim=-1)))
        if accepted:
            final_direction = (
                global_unit
                * median_norm
                * args.saved_magnitude_scale
            ).cpu()
            saved_directions[property_name] = final_direction
            status = "ACCEPTED"
        else:
            saved_directions[property_name] = None
            status = "REJECTED"

        print(
            f"\n{property_name}: {status}, examples={count}, "
            f"coherence={cross_object_coherence:.3f}, "
            f"median LOO cosine={median_loo:.3f}, "
            f"positive LOO={positive_fraction:.1%}, "
            f"saved norm={median_norm * args.saved_magnitude_scale:.4f}"
        )

        for record, loo_score in zip(property_records, loo_scores):
            report_rows.append(
                {
                    "property_name": property_name,
                    "object_name": record["object_name"],
                    "property_status": status,
                    "example_count": count,
                    "path_quality": record["quality"],
                    "endpoint_norm": record["endpoint_norm"],
                    "cosine_to_global": cosine(
                        record["direction"], global_unit
                    ),
                    "leave_one_out_cosine": loo_score,
                    "global_coherence": cross_object_coherence,
                    "property_mean_loo_cosine": mean_loo,
                    "property_median_loo_cosine": median_loo,
                    "property_positive_loo_fraction": positive_fraction,
                    "property_above_0.1_loo_fraction": above_point_one_fraction,
                    "path_efficiency": record["path_efficiency"],
                    "monotonic_fraction": record["monotonic_fraction"],
                    "progress_correlation": record["progress_correlation"],
                }
            )

    return saved_directions, report_rows


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary_path)
    os.replace(temporary_path, path)


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    if not output_root.is_dir():
        raise FileNotFoundError(f"Output root does not exist: {output_root}")

    args.save_path = args.save_path or str(
        output_root / "empirical_property_directions.pt"
    )
    args.report_path = args.report_path or str(
        output_root
        / "_direction_audit"
        / "empirical_direction_report.csv"
    )

    ip_model, device = build_ip_adapter(args)
    records = collect_endpoint_directions(
        ip_model,
        output_root,
        device,
        args.minimum_path_quality,
    )
    directions, report_rows = build_global_directions(records, args)

    atomic_torch_save(directions, args.save_path)
    write_csv(args.report_path, report_rows)

    file_size = Path(args.save_path).stat().st_size
    print(f"\nSaved empirical directions: {Path(args.save_path).resolve()}")
    print(f"File size: {file_size} bytes")
    print(f"Report: {Path(args.report_path).resolve()}")

    ip_model.image_encoder.to("cpu")
    ip_model.image_proj_model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
