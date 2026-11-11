"""Audit learned CLIP/IP-Adapter property directions inside object masks.

Add this file beside the generation script, import `audit_output_tree`, and
call it while `ip_model` and `property_directions` are still available:

    from masked_clip_direction_audit import audit_output_tree

    audit_output_tree(
        ip_model=ip_model,
        output_root=output_folder,
        property_directions=property_directions,
        report_dir=os.path.join(output_folder, "_direction_audit"),
    )

The expected folder structure is:

    output_root/
      object_name/
        source/
        albedo/
          edited/
          mask/
        rust/
          edited/
          mask/
        ...

For every property sequence, this script:

1. Removes background information by placing the masked object on neutral gray.
2. Encodes every edited frame with the exact IP-Adapter image encoder.
3. Projects every CLIP embedding through IP-Adapter's image projection model.
4. Measures endpoint movement, monotonicity, straightness and backtracking.
5. Compares the observed output direction with the learned property direction.
6. Writes per-frame and per-property CSV reports plus trajectory plots.
"""

import csv
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


PROPERTY_NAMES = (
    "albedo",
    "discoloration",
    "glow",
    "metalness",
    "paint_degradation",
    "roughness",
    "rust",
    "transparency",
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _natural_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", Path(path).name)
    ]


def _image_files(folder):
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(
        [path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS],
        key=_natural_key,
    )


def _frame_key(path):
    """Normalize names such as rust_03.png and rust_03_mask.png."""
    stem = Path(path).stem.lower()
    for suffix in ("_mask", "-mask", " mask"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def _match_images_and_masks(edited_dir, mask_dir):
    edited = _image_files(edited_dir)
    masks = _image_files(mask_dir)
    mask_by_key = {_frame_key(path): path for path in masks}

    pairs = []
    unmatched = []
    for image_path in edited:
        mask_path = mask_by_key.get(_frame_key(image_path))
        if mask_path is None:
            unmatched.append(image_path)
        else:
            pairs.append((image_path, mask_path))

    # Some datasets use unrelated mask filenames but preserve ordering.
    if unmatched and len(edited) == len(masks):
        print(
            f"  WARNING: matching {len(edited)} images and masks by sorted order "
            "because their stems differ."
        )
        return list(zip(edited, masks))

    if unmatched:
        names = ", ".join(path.name for path in unmatched[:5])
        print(f"  WARNING: no corresponding mask for: {names}")

    return pairs


def _load_masked_reference(image_path, mask_path, neutral_value=127):
    with Image.open(image_path) as handle:
        image = handle.convert("RGB").copy()
    with Image.open(mask_path) as handle:
        mask = handle.convert("L").copy()

    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.NEAREST)

    mask_array = np.asarray(mask, dtype=np.uint8)
    binary_array = np.where(mask_array >= 128, 255, 0).astype(np.uint8)
    mask = Image.fromarray(binary_array, mode="L")

    bbox = mask.getbbox()
    if bbox is None:
        raise ValueError(f"Mask is empty: {mask_path}")

    neutral = Image.new("RGB", image.size, (neutral_value,) * 3)
    masked = Image.composite(image, neutral, mask)

    # Cropping makes CLIP attend to the object rather than mask occupancy.
    cropped_image = masked.crop(bbox)
    cropped_mask = mask.crop(bbox)
    width, height = cropped_image.size
    square_size = max(width, height)

    square_image = Image.new("RGB", (square_size, square_size), (neutral_value,) * 3)
    square_mask = Image.new("L", (square_size, square_size), 0)
    offset = ((square_size - width) // 2, (square_size - height) // 2)
    square_image.paste(cropped_image, offset)
    square_mask.paste(cropped_mask, offset)

    coverage = float(np.mean(np.asarray(square_mask, dtype=np.uint8) > 0))
    return square_image, coverage


@torch.inference_mode()
def _encode_and_project(ip_model, image, device):
    encoder_dtype = next(ip_model.image_encoder.parameters()).dtype
    pixel_values = ip_model.clip_image_processor(
        images=[image], return_tensors="pt"
    ).pixel_values.to(device=device, dtype=encoder_dtype)

    clip_embedding = ip_model.image_encoder(pixel_values).image_embeds.float()
    conditional_tokens, _ = ip_model.get_image_embeds(
        pil_image=None,
        clip_image_embeds=clip_embedding,
    )

    del pixel_values
    return clip_embedding.detach().cpu(), conditional_tokens.float().detach().cpu()


def _cosine(first, second):
    first = first.reshape(1, -1).float()
    second = second.reshape(1, -1).float()
    first_norm = float(first.norm())
    second_norm = float(second.norm())
    if first_norm < 1e-12 or second_norm < 1e-12:
        return float("nan")
    return float(F.cosine_similarity(first, second, dim=-1).item())


def _angle_degrees(first, second):
    cosine = _cosine(first, second)
    if not np.isfinite(cosine):
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _trajectory_metrics(vectors):
    """Return frame metrics and a summary for a [frames, dimensions] tensor."""
    vectors = vectors.float().reshape(vectors.shape[0], -1)
    count = vectors.shape[0]
    if count < 2:
        raise ValueError("A trajectory needs at least two frames")

    origin = vectors[0]
    endpoint_direction = vectors[-1] - origin
    endpoint_norm = float(endpoint_direction.norm())
    origin_norm = float(origin.norm())
    endpoint_relative_change = endpoint_norm / max(origin_norm, 1e-12)

    steps = vectors[1:] - vectors[:-1]
    step_norms = steps.norm(dim=-1)
    path_length = float(step_norms.sum())
    path_efficiency = endpoint_norm / max(path_length, 1e-12)

    deltas = vectors - origin
    delta_norms = deltas.norm(dim=-1)

    if endpoint_norm < 1e-12:
        projections = torch.zeros(count)
        orthogonal_norms = delta_norms.clone()
        step_cosines = torch.full((count - 1,), float("nan"))
    else:
        endpoint_unit = endpoint_direction / endpoint_norm
        projection_distances = deltas @ endpoint_unit
        projections = projection_distances / endpoint_norm
        orthogonal = deltas - projection_distances[:, None] * endpoint_unit[None, :]
        orthogonal_norms = orthogonal.norm(dim=-1)
        step_cosines = F.cosine_similarity(
            steps,
            endpoint_unit.expand_as(steps),
            dim=-1,
        )

    off_axis_fractions = orthogonal_norms / delta_norms.clamp_min(1e-12)
    off_axis_fractions[0] = 0.0

    projection_changes = projections[1:] - projections[:-1]
    tolerance = 0.01
    monotonic_fraction = float((projection_changes >= -tolerance).float().mean())
    backtracking_fraction = float((projection_changes < -tolerance).float().mean())

    indices = np.arange(count, dtype=np.float64)
    projection_array = projections.numpy().astype(np.float64)
    if np.std(projection_array) < 1e-12:
        progress_correlation = 0.0
    else:
        progress_correlation = float(np.corrcoef(indices, projection_array)[0, 1])

    finite_step_cosines = step_cosines[torch.isfinite(step_cosines)]
    mean_step_alignment = (
        float(finite_step_cosines.mean()) if len(finite_step_cosines) else float("nan")
    )

    frame_metrics = []
    for index in range(count):
        step_alignment = float("nan") if index == 0 else float(step_cosines[index - 1])
        frame_metrics.append(
            {
                "frame_index": index,
                "distance_from_first": float(delta_norms[index]),
                "relative_distance_from_first": float(delta_norms[index])
                / max(origin_norm, 1e-12),
                "endpoint_projection": float(projections[index]),
                "off_axis_distance": float(orthogonal_norms[index]),
                "off_axis_fraction": float(off_axis_fractions[index]),
                "step_alignment_with_endpoint": step_alignment,
                "angle_from_first_deg": _angle_degrees(origin, vectors[index]),
            }
        )

    summary = {
        "frame_count": count,
        "endpoint_norm": endpoint_norm,
        "endpoint_relative_change": endpoint_relative_change,
        "path_length": path_length,
        "path_efficiency": path_efficiency,
        "monotonic_fraction": monotonic_fraction,
        "backtracking_fraction": backtracking_fraction,
        "progress_correlation": progress_correlation,
        "mean_step_alignment": mean_step_alignment,
        "median_off_axis_fraction": float(torch.median(off_axis_fractions[1:])),
        "endpoint_angle_deg": _angle_degrees(origin, vectors[-1]),
    }
    return frame_metrics, summary, endpoint_direction


def _classify(clip_summary, token_summary, learned_alignment):
    if clip_summary["endpoint_relative_change"] < 0.002:
        return "NO_CLIP_MOVEMENT"
    if token_summary["endpoint_relative_change"] < 0.002:
        return "PROJECTOR_SUPPRESSES_MOVEMENT"
    if clip_summary["monotonic_fraction"] < 0.70:
        return "BACKTRACKING_NOT_A_STABLE_DIRECTION"
    if clip_summary["path_efficiency"] < 0.35:
        return "HIGH_OFF_AXIS_DRIFT"
    if clip_summary["progress_correlation"] < 0.70:
        return "NOT_GRADED"
    if np.isfinite(learned_alignment) and learned_alignment < 0.10:
        return "OUTPUT_DOES_NOT_FOLLOW_LEARNED_DIRECTION"
    return "CONSISTENT_DIRECTION"


def _write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_plot(path, title, clip_metrics, token_metrics):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    indices = [row["frame_index"] for row in clip_metrics]
    clip_progress = [row["endpoint_projection"] for row in clip_metrics]
    token_progress = [row["endpoint_projection"] for row in token_metrics]
    off_axis = [row["off_axis_fraction"] for row in clip_metrics]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(indices, clip_progress, marker="o", label="CLIP embedding")
    axes[0].plot(indices, token_progress, marker="s", label="IP-Adapter tokens")
    axes[0].plot(indices, np.linspace(0.0, 1.0, len(indices)), "--", label="ideal")
    axes[0].set_xlabel("frame index")
    axes[0].set_ylabel("fraction of endpoint direction")
    axes[0].set_title("Directional progress")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(indices, off_axis, marker="o", color="tab:red")
    axes[1].set_xlabel("frame index")
    axes[1].set_ylabel("off-axis fraction")
    axes[1].set_ylim(bottom=0.0)
    axes[1].set_title("CLIP drift away from endpoint axis")
    axes[1].grid(alpha=0.25)

    figure.suptitle(title)
    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _direction_cosine_rows(object_name, directions):
    rows = []
    names = sorted(directions)
    for first_name in names:
        row = {"object_name": object_name, "property": first_name}
        for second_name in names:
            row[second_name] = _cosine(
                directions[first_name], directions[second_name]
            )
        rows.append(row)
    return rows


@torch.inference_mode()
def audit_output_tree(
    ip_model,
    output_root,
    property_directions=None,
    report_dir=None,
    device=None,
):
    """Audit all object/property folders and return their summary rows.

    Args:
        ip_model: The already-loaded IPAdapterXL instance used for generation.
        output_root: Folder containing one directory per object.
        property_directions: Optional dict from the generation script. When
            supplied, empirical output directions are compared with the learned
            property directions.
        report_dir: Destination for CSV/PNG reports. Defaults to
            output_root/_direction_audit.
        device: Encoder device. Defaults to the image encoder's current device.
    """
    output_root = Path(output_root)
    report_dir = Path(report_dir or output_root / "_direction_audit")
    report_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = next(ip_model.image_encoder.parameters()).device
    device = torch.device(device)

    property_directions = property_directions or {}
    summary_rows = []
    frame_rows = []

    object_dirs = sorted(
        [path for path in output_root.iterdir() if path.is_dir()],
        key=_natural_key,
    )

    for object_dir in object_dirs:
        if object_dir.name.startswith("_"):
            continue

        print(f"\nOBJECT: {object_dir.name}")
        object_empirical_directions = {}

        for property_name in PROPERTY_NAMES:
            property_dir = object_dir / property_name
            edited_dir = property_dir / "edited"
            mask_dir = property_dir / "mask"
            if not edited_dir.is_dir() or not mask_dir.is_dir():
                continue

            print(f"  PROPERTY: {property_name}")
            pairs = _match_images_and_masks(edited_dir, mask_dir)
            if len(pairs) < 2:
                print("    NOT TESTABLE: fewer than two matched frames")
                summary_rows.append(
                    {
                        "object_name": object_dir.name,
                        "property_name": property_name,
                        "status": "NOT_TESTABLE",
                        "frame_count": len(pairs),
                    }
                )
                continue

            clip_vectors = []
            token_vectors = []
            coverages = []

            for image_path, mask_path in pairs:
                reference, coverage = _load_masked_reference(
                    image_path, mask_path
                )
                clip_embedding, tokens = _encode_and_project(
                    ip_model, reference, device
                )
                clip_vectors.append(clip_embedding.reshape(-1))
                token_vectors.append(tokens.reshape(-1))
                coverages.append(coverage)

            clip_vectors = torch.stack(clip_vectors)
            token_vectors = torch.stack(token_vectors)
            clip_metrics, clip_summary, empirical_clip_direction = (
                _trajectory_metrics(clip_vectors)
            )
            token_metrics, token_summary, empirical_token_direction = (
                _trajectory_metrics(token_vectors)
            )
            object_empirical_directions[property_name] = empirical_clip_direction

            learned_alignment = float("nan")
            learned_token_alignment = float("nan")
            learned_direction = property_directions.get(property_name)

            if learned_direction is not None:
                learned_direction = learned_direction.detach().cpu().reshape(-1).float()
                if learned_direction.numel() == empirical_clip_direction.numel():
                    learned_alignment = _cosine(
                        empirical_clip_direction, learned_direction
                    )

                    learned_endpoint = (
                        clip_vectors[0].reshape(1, -1) + learned_direction.reshape(1, -1)
                    )
                    learned_tokens, _ = ip_model.get_image_embeds(
                        pil_image=None,
                        clip_image_embeds=learned_endpoint.to(device).float(),
                    )
                    base_tokens, _ = ip_model.get_image_embeds(
                        pil_image=None,
                        clip_image_embeds=clip_vectors[0].reshape(1, -1).to(device).float(),
                    )
                    learned_token_direction = (
                        learned_tokens.float().cpu() - base_tokens.float().cpu()
                    ).reshape(-1)
                    learned_token_alignment = _cosine(
                        empirical_token_direction, learned_token_direction
                    )

            status = _classify(
                clip_summary, token_summary, learned_alignment
            )

            summary = {
                "object_name": object_dir.name,
                "property_name": property_name,
                "status": status,
                **{f"clip_{key}": value for key, value in clip_summary.items()},
                **{f"token_{key}": value for key, value in token_summary.items()},
                "empirical_vs_learned_clip_cosine": learned_alignment,
                "empirical_vs_learned_token_cosine": learned_token_alignment,
                "mean_mask_coverage": float(np.mean(coverages)),
            }
            summary_rows.append(summary)

            for index, ((image_path, mask_path), clip_row, token_row) in enumerate(
                zip(pairs, clip_metrics, token_metrics)
            ):
                frame_rows.append(
                    {
                        "object_name": object_dir.name,
                        "property_name": property_name,
                        "frame_index": index,
                        "image_path": str(image_path.resolve()),
                        "mask_path": str(mask_path.resolve()),
                        "mask_coverage": coverages[index],
                        **{f"clip_{key}": value for key, value in clip_row.items()},
                        **{f"token_{key}": value for key, value in token_row.items()},
                    }
                )

            print(
                f"    {status}: "
                f"CLIP endpoint={clip_summary['endpoint_relative_change']:.3%}, "
                f"monotonic={clip_summary['monotonic_fraction']:.1%}, "
                f"efficiency={clip_summary['path_efficiency']:.3f}, "
                f"progress r={clip_summary['progress_correlation']:.3f}, "
                f"learned alignment={learned_alignment:.3f}, "
                f"token alignment={learned_token_alignment:.3f}"
            )

            _save_plot(
                report_dir / object_dir.name / f"{property_name}_trajectory.png",
                f"{object_dir.name} / {property_name} / {status}",
                clip_metrics,
                token_metrics,
            )

        cosine_rows = _direction_cosine_rows(
            object_dir.name, object_empirical_directions
        )
        _write_csv(
            report_dir / object_dir.name / "property_direction_cosines.csv",
            cosine_rows,
        )

    _write_csv(report_dir / "property_direction_summary.csv", summary_rows)
    _write_csv(report_dir / "property_frame_metrics.csv", frame_rows)

    print(f"\nDirection audit written to: {report_dir.resolve()}")
    return summary_rows

