# """Leave-one-object-out audit for existing property-scale image folders.

# Expected structure for every supplied view root:

# root/
#   empirical_property_directions.pt       # optional comparison only
#   object_name/
#     base_canvas.png
#     albedo/
#       edited/albedo_00.png ... albedo_09.png
#       mask/...
#     roughness/
#       edited/roughness_00.png ... roughness_09.png
#       mask/...
#     ...

# No new diffusion images are generated. Scale is reconstructed as frame/(N-1).

# The important evaluation is leave-one-canonical-object-out (LOO): directions are
# re-estimated from all other objects, then tested on the held-out object's complete
# trajectory. If two views of the same object are supplied, both are excluded while
# training that fold. This avoids evaluating a saved direction on its own examples.

# Example:

# python audit_existing_property_scale_folders_loo.py \
#   --roots outputs/first_view outputs/second_view \
#   --view-names view_1 view_2 \
#   --image-encoder models/image_encoder \
#   --saved-directions outputs/first_view/empirical_property_directions.pt \
#   --output outputs/existing_property_scale_loo_audit
# """

# import argparse
# import json
# import re
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import torch
# from PIL import Image
# from scipy.stats import kendalltau, pearsonr, spearmanr


# PROPERTY_ORDER = [
#     "albedo",
#     "discoloration",
#     "glow",
#     "metalness",
#     "paint_degradation",
#     "roughness",
#     "rust",
#     "transparency",
# ]
# IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


# def parse_args():
#     parser = argparse.ArgumentParser(
#         description="LOO audit of existing property-scale folders"
#     )
#     parser.add_argument("--roots", nargs="+", required=True)
#     parser.add_argument("--view-names", nargs="*", default=None)
#     parser.add_argument("--image-encoder", default="models/image_encoder")
#     parser.add_argument("--saved-directions", default=None)
#     parser.add_argument("--object-map", default=None)
#     parser.add_argument("--output", default="outputs/existing_property_scale_loo_audit")
#     parser.add_argument("--device", default="cuda")
#     parser.add_argument("--batch-size", type=int, default=16)
#     parser.add_argument("--mask-threshold", type=int, default=128)
#     parser.add_argument("--reference-padding", type=float, default=0.12)
#     parser.add_argument("--bootstrap-repeats", type=int, default=1000)
#     parser.add_argument("--seed", type=int, default=12345)
#     return parser.parse_args()


# def load_object_map(path):
#     if path is None:
#         return {}
#     table = pd.read_csv(path)
#     required = {"view", "object", "canonical_object"}
#     missing = required - set(table.columns)
#     if missing:
#         raise ValueError(f"Object map missing columns: {sorted(missing)}")
#     return {
#         (str(row.view), str(row.object)): str(row.canonical_object)
#         for row in table.itertuples(index=False)
#     }


# def frame_number(path, property_name):
#     match = re.search(
#         rf"^{re.escape(property_name)}[_-](\d+)$", path.stem, flags=re.IGNORECASE
#     )
#     if match:
#         return int(match.group(1))
#     trailing = re.search(r"(\d+)$", path.stem)
#     return int(trailing.group(1)) if trailing else None


# def find_source(object_folder):
#     candidates = [
#         object_folder / "base_canvas.png",
#         object_folder / "source" / "source.jpg",
#         object_folder / "source" / "source.png",
#     ]
#     for path in candidates:
#         if path.is_file():
#             return path
#     return None


# def find_mask(property_folder, image_path, property_name, frame):
#     mask_folder = property_folder / "mask"
#     candidates = [
#         mask_folder / f"{image_path.stem}_mask.png",
#         mask_folder / image_path.name,
#         mask_folder / f"{property_name}_{frame:02d}_mask.png",
#         mask_folder / f"{property_name}_{frame:02d}.png",
#         property_folder.parent / "_debug_actual_inpainting_mask.png",
#     ]
#     for path in candidates:
#         if path.is_file():
#             return path
#     if mask_folder.is_dir():
#         files = sorted(
#             path for path in mask_folder.iterdir()
#             if path.suffix.lower() in IMAGE_EXTENSIONS
#         )
#         if len(files) == 1:
#             return files[0]
#     return None


# def scan_root(root, view, object_map):
#     rows = []
#     root = Path(root).resolve()
#     if not root.is_dir():
#         raise FileNotFoundError(root)
#     for object_folder in sorted(root.iterdir()):
#         if not object_folder.is_dir() or object_folder.name.startswith("_"):
#             continue
#         source_path = find_source(object_folder)
#         if source_path is None:
#             continue
#         object_name = object_folder.name
#         canonical = object_map.get((view, object_name), object_name)
#         for property_name in PROPERTY_ORDER:
#             property_folder = object_folder / property_name
#             edited_folder = property_folder / "edited"
#             if not edited_folder.is_dir():
#                 continue
#             discovered = []
#             for image_path in sorted(edited_folder.iterdir()):
#                 if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
#                     continue
#                 frame = frame_number(image_path, property_name)
#                 if frame is not None:
#                     discovered.append((frame, image_path))
#             discovered.sort(key=lambda item: item[0])
#             if len(discovered) < 2:
#                 continue
#             minimum = discovered[0][0]
#             maximum = discovered[-1][0]
#             denominator = max(maximum - minimum, 1)
#             for frame, image_path in discovered:
#                 mask_path = find_mask(
#                     property_folder, image_path, property_name, frame
#                 )
#                 rows.append(
#                     {
#                         "root": str(root),
#                         "view": view,
#                         "object": object_name,
#                         "canonical_object": canonical,
#                         "property": property_name,
#                         "frame": frame,
#                         "property_scale": (frame - minimum) / denominator,
#                         "image_path": str(image_path),
#                         "mask_path": str(mask_path) if mask_path else "",
#                         "source_path": str(source_path),
#                     }
#                 )
#     return rows


# def scan_all(roots, views, object_map):
#     rows = []
#     for root, view in zip(roots, views):
#         rows.extend(scan_root(root, view, object_map))
#     if not rows:
#         raise RuntimeError("No property-scale images were discovered")
#     return pd.DataFrame(rows).sort_values(
#         ["view", "canonical_object", "property", "frame"]
#     )


# def load_rgb(path):
#     with Image.open(path) as image:
#         return image.convert("RGB").copy()


# def load_binary_mask(path, size, threshold):
#     with Image.open(path) as image:
#         mask = image.convert("L").resize(size, Image.Resampling.NEAREST)
#     return np.asarray(mask, dtype=np.uint8) >= threshold


# def local_reference(image, mask, padding):
#     mask_image = Image.fromarray(
#         np.where(mask, 255, 0).astype(np.uint8), mode="L"
#     )
#     box = mask_image.getbbox()
#     if box is None:
#         raise RuntimeError("Empty mask")
#     image_width, image_height = image.size
#     x0, y0, x1, y1 = box
#     side = int(
#         round(max(x1 - x0, y1 - y0) * (1.0 + 2.0 * float(padding)))
#     )
#     side = max(2, min(side, image_width, image_height))
#     center_x = (x0 + x1) / 2.0
#     center_y = (y0 + y1) / 2.0
#     crop_x0 = int(round(center_x - side / 2.0))
#     crop_y0 = int(round(center_y - side / 2.0))
#     crop_x0 = max(0, min(crop_x0, image_width - side))
#     crop_y0 = max(0, min(crop_y0, image_height - side))
#     return image.crop(
#         (crop_x0, crop_y0, crop_x0 + side, crop_y0 + side)
#     ).resize((1024, 1024), Image.Resampling.LANCZOS)


# def load_encoder(path, device):
#     try:
#         from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
#     except ImportError as error:
#         raise RuntimeError("The transformers package is required") from error
#     model = CLIPVisionModelWithProjection.from_pretrained(path).to(device).eval()
#     try:
#         processor = CLIPImageProcessor.from_pretrained(path)
#     except (OSError, ValueError):
#         processor = CLIPImageProcessor()
#     return processor, model


# @torch.inference_mode()
# def encode_images(images, processor, model, device, batch_size):
#     outputs = []
#     for start in range(0, len(images), batch_size):
#         batch = images[start : start + batch_size]
#         pixels = processor(images=batch, return_tensors="pt").pixel_values.to(device)
#         outputs.append(model(pixel_values=pixels).image_embeds.float().cpu())
#     return torch.cat(outputs, dim=0)


# def encode_inventory(
#     inventory, processor, model, device, batch_size, threshold, padding
# ):
#     records = []
#     references = []
#     valid_record_indices = []
#     for row in inventory.to_dict("records"):
#         image_path = Path(row["image_path"])
#         mask_path = Path(row["mask_path"]) if row["mask_path"] else None
#         source_path = Path(row["source_path"])
#         if mask_path is None or not mask_path.is_file():
#             row["audit_error"] = "mask missing"
#             records.append(row)
#             continue
#         image = load_rgb(image_path)
#         source = load_rgb(source_path).resize(image.size, Image.Resampling.LANCZOS)
#         mask = load_binary_mask(mask_path, image.size, threshold)
#         if not mask.any():
#             row["audit_error"] = "empty mask"
#             records.append(row)
#             continue
#         generated_array = np.asarray(image, dtype=np.uint8)
#         source_array = np.asarray(source, dtype=np.uint8)
#         difference = np.abs(
#             generated_array.astype(np.int16) - source_array.astype(np.int16)
#         )
#         outside = difference[~mask]
#         inside = difference[mask]
#         row["mask_coverage"] = float(mask.mean())
#         row["outside_max_delta"] = int(outside.max()) if outside.size else 0
#         row["inside_mean_delta"] = float(inside.mean()) if inside.size else np.nan
#         records.append(row)
#         references.append(local_reference(image, mask, padding))
#         valid_record_indices.append(len(records) - 1)

#     if not references:
#         raise RuntimeError("No image had a usable mask")
#     embeddings = encode_images(
#         references, processor, model, device, batch_size
#     ).float()
#     for embedding_index, record_index in enumerate(valid_record_indices):
#         records[record_index]["embedding"] = embeddings[embedding_index].numpy()
#         records[record_index]["embedding_norm"] = float(
#             embeddings[embedding_index].norm()
#         )
#     return pd.DataFrame(records)


# def endpoint_table(scored):
#     rows = []
#     group_columns = ["view", "canonical_object", "object", "property"]
#     for keys, group in scored.groupby(group_columns):
#         group = group.sort_values("property_scale")
#         valid = group[group["embedding"].notna()]
#         if len(valid) < 2:
#             continue
#         first = np.asarray(valid.iloc[0]["embedding"], dtype=np.float32)
#         last = np.asarray(valid.iloc[-1]["embedding"], dtype=np.float32)
#         delta = last - first
#         view, canonical_object, object_name, property_name = keys
#         rows.append(
#             {
#                 "view": view,
#                 "canonical_object": canonical_object,
#                 "object": object_name,
#                 "property": property_name,
#                 "endpoint_delta": delta,
#                 "endpoint_delta_norm": float(np.linalg.norm(delta)),
#                 "frames": len(valid),
#             }
#         )
#     return pd.DataFrame(rows)


# def unit(vector):
#     vector = np.asarray(vector, dtype=np.float32)
#     norm = float(np.linalg.norm(vector))
#     return vector / max(norm, 1e-8)


# def train_loo_directions(endpoints, held_out_object):
#     training = endpoints[endpoints["canonical_object"] != held_out_object]
#     directions = {}
#     for property_name, group in training.groupby("property"):
#         deltas = np.stack(group["endpoint_delta"].to_list())
#         mean_delta = deltas.mean(axis=0)
#         if np.linalg.norm(mean_delta) > 1e-8:
#             directions[property_name] = unit(mean_delta)
#     return directions


# def safe_corr(function, x, y):
#     x = np.asarray(x, dtype=np.float64)
#     y = np.asarray(y, dtype=np.float64)
#     keep = np.isfinite(x) & np.isfinite(y)
#     x, y = x[keep], y[keep]
#     if len(x) < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
#         return np.nan
#     try:
#         result = function(x, y)
#         return float(result.statistic if hasattr(result, "statistic") else result[0])
#     except Exception:
#         return np.nan


# def linear_r2(x, y):
#     x = np.asarray(x, dtype=np.float64)
#     y = np.asarray(y, dtype=np.float64)
#     if len(x) < 3 or np.allclose(y, y[0]):
#         return np.nan
#     prediction = np.polyval(np.polyfit(x, y, 1), x)
#     denominator = np.sum((y - y.mean()) ** 2)
#     return (
#         float(1.0 - np.sum((y - prediction) ** 2) / denominator)
#         if denominator > 1e-12
#         else np.nan
#     )


# def evaluate_loo(scored, endpoints):
#     rows = []
#     response_rows = []
#     canonical_objects = sorted(endpoints["canonical_object"].unique())
#     for held_out in canonical_objects:
#         directions = train_loo_directions(endpoints, held_out)
#         held_frames = scored[scored["canonical_object"] == held_out]
#         for keys, group in held_frames.groupby(["view", "object", "property"]):
#             view, object_name, target_property = keys
#             if target_property not in directions:
#                 continue
#             group = group.sort_values("property_scale")
#             group = group[group["embedding"].notna()]
#             if len(group) < 3:
#                 continue
#             embeddings = np.stack(group["embedding"].to_list())
#             scales = group["property_scale"].to_numpy(float)
#             endpoint_responses = {}
#             target_curve = None
#             for measured_property, direction in directions.items():
#                 curve = embeddings @ direction
#                 endpoint_response = float(curve[-1] - curve[0])
#                 endpoint_responses[measured_property] = endpoint_response
#                 response_rows.append(
#                     {
#                         "held_out_object": held_out,
#                         "view": view,
#                         "object": object_name,
#                         "target_property": target_property,
#                         "measured_property": measured_property,
#                         "endpoint_response": endpoint_response,
#                     }
#                 )
#                 if measured_property == target_property:
#                     target_curve = curve
#             if target_curve is None:
#                 continue
#             target_response = endpoint_responses[target_property]
#             off_target = {
#                 name: value for name, value in endpoint_responses.items()
#                 if name != target_property
#             }
#             largest_name = max(off_target, key=lambda name: abs(off_target[name]))
#             largest_value = off_target[largest_name]
#             total = sum(abs(value) for value in endpoint_responses.values())
#             path_length = float(np.sum(np.abs(np.diff(target_curve))))
#             rows.append(
#                 {
#                     "held_out_object": held_out,
#                     "view": view,
#                     "object": object_name,
#                     "property": target_property,
#                     "frames": len(group),
#                     "loo_target_spearman": safe_corr(
#                         spearmanr, scales, target_curve
#                     ),
#                     "loo_target_kendall": safe_corr(kendalltau, scales, target_curve),
#                     "loo_target_linear_r2": linear_r2(scales, target_curve),
#                     "loo_monotonic_fraction": float(
#                         np.mean(np.diff(target_curve) >= -1e-8)
#                     ),
#                     "loo_target_endpoint_response": target_response,
#                     "loo_path_efficiency": (
#                         abs(target_response) / path_length
#                         if path_length > 1e-12 else np.nan
#                     ),
#                     "loo_selectivity": (
#                         abs(target_response) / total if total > 1e-12 else np.nan
#                     ),
#                     "loo_target_rank": 1 + sum(
#                         abs(value) > abs(target_response)
#                         for value in off_target.values()
#                     ),
#                     "largest_off_target_property": largest_name,
#                     "largest_off_target_response": largest_value,
#                     "target_to_largest_off_target_ratio": (
#                         abs(target_response) / abs(largest_value)
#                         if abs(largest_value) > 1e-12 else np.inf
#                     ),
#                     "maximum_outside_delta": float(
#                         np.nanmax(group["outside_max_delta"].to_numpy(float))
#                     ),
#                 }
#             )
#     return pd.DataFrame(rows), pd.DataFrame(response_rows)


# def full_directions(endpoints):
#     result = {}
#     for property_name, group in endpoints.groupby("property"):
#         result[property_name] = unit(
#             np.stack(group["endpoint_delta"].to_list()).mean(axis=0)
#         )
#     return result


# def direction_stability(endpoints, repeats, rng):
#     full = full_directions(endpoints)
#     rows = []
#     learning_rows = []
#     for property_name, group in endpoints.groupby("property"):
#         by_object = []
#         for canonical_object, object_group in group.groupby("canonical_object"):
#             by_object.append(
#                 (
#                     canonical_object,
#                     np.stack(object_group["endpoint_delta"].to_list()).mean(axis=0),
#                 )
#             )
#         if len(by_object) < 2:
#             continue
#         reference = full[property_name]
#         for held_out, _ in by_object:
#             training = [delta for name, delta in by_object if name != held_out]
#             loo = unit(np.stack(training).mean(axis=0))
#             rows.append(
#                 {
#                     "property": property_name,
#                     "held_out_object": held_out,
#                     "loo_to_full_cosine": float(np.dot(loo, reference)),
#                     "training_objects": len(training),
#                 }
#             )
#         maximum_n = len(by_object)
#         deltas = np.stack([delta for _, delta in by_object])
#         for sample_size in range(2, maximum_n + 1):
#             cosines = []
#             for _ in range(repeats):
#                 indices = rng.choice(maximum_n, size=sample_size, replace=True)
#                 estimate = unit(deltas[indices].mean(axis=0))
#                 cosines.append(float(np.dot(estimate, reference)))
#             learning_rows.append(
#                 {
#                     "property": property_name,
#                     "sample_size": sample_size,
#                     "median_cosine": float(np.median(cosines)),
#                     "ci_low": float(np.percentile(cosines, 2.5)),
#                     "ci_high": float(np.percentile(cosines, 97.5)),
#                 }
#             )
#     return pd.DataFrame(rows), pd.DataFrame(learning_rows), full


# def load_saved_directions(path):
#     if path is None:
#         return {}
#     path = Path(path)
#     if not path.is_file():
#         raise FileNotFoundError(path)
#     try:
#         saved = torch.load(path, map_location="cpu", weights_only=True)
#     except TypeError:
#         saved = torch.load(path, map_location="cpu")
#     result = {}
#     for property_name in PROPERTY_ORDER:
#         tensor = saved.get(property_name) if isinstance(saved, dict) else None
#         if tensor is not None:
#             result[property_name] = unit(tensor.detach().float().reshape(-1).numpy())
#     return result


# def compare_saved(full, saved):
#     rows = []
#     for property_name in sorted(set(full) & set(saved)):
#         rows.append(
#             {
#                 "property": property_name,
#                 "recomputed_to_saved_cosine": float(
#                     np.dot(full[property_name], saved[property_name])
#                 ),
#             }
#         )
#     return pd.DataFrame(rows)


# def response_matrices(response_rows, properties):
#     if response_rows.empty:
#         return pd.DataFrame(), pd.DataFrame()
#     matrix = response_rows.pivot_table(
#         index="target_property",
#         columns="measured_property",
#         values="endpoint_response",
#         aggfunc="median",
#     ).reindex(index=properties, columns=properties)
#     normalized = matrix.copy()
#     for target in normalized.index:
#         values = normalized.loc[target].to_numpy(float)
#         denominator = np.nansum(np.abs(values))
#         if denominator > 1e-12:
#             normalized.loc[target] = values / denominator
#     return matrix, normalized


# def property_summary(loo_metrics, stability):
#     rows = []
#     for property_name, group in loo_metrics.groupby("property"):
#         stable = stability[stability["property"] == property_name]
#         rows.append(
#             {
#                 "property": property_name,
#                 "held_out_sweeps": len(group),
#                 "held_out_objects": group["held_out_object"].nunique(),
#                 "views": group["view"].nunique(),
#                 "median_loo_spearman": float(np.nanmedian(group["loo_target_spearman"])),
#                 "median_loo_kendall": float(np.nanmedian(group["loo_target_kendall"])),
#                 "median_loo_monotonic_fraction": float(
#                     np.nanmedian(group["loo_monotonic_fraction"])
#                 ),
#                 "median_loo_selectivity": float(np.nanmedian(group["loo_selectivity"])),
#                 "target_rank_1_fraction": float(
#                     np.mean(group["loo_target_rank"].to_numpy(float) == 1)
#                 ),
#                 "positive_endpoint_fraction": float(
#                     np.mean(group["loo_target_endpoint_response"].to_numpy(float) > 0)
#                 ),
#                 "median_target_to_off_target_ratio": float(
#                     np.nanmedian(group["target_to_largest_off_target_ratio"])
#                 ),
#                 "median_loo_direction_cosine": (
#                     float(np.nanmedian(stable["loo_to_full_cosine"]))
#                     if len(stable) else np.nan
#                 ),
#                 "outside_mask_pass_fraction": float(
#                     np.mean(group["maximum_outside_delta"].to_numpy(float) == 0)
#                 ),
#             }
#         )
#     return pd.DataFrame(rows).sort_values("property")


# def view_consistency(scored, endpoints):
#     rows = []
#     for (canonical_object, property_name), group in scored.groupby(
#         ["canonical_object", "property"]
#     ):
#         views = sorted(group["view"].unique())
#         if len(views) < 2:
#             continue
#         directions = train_loo_directions(endpoints, canonical_object)
#         if property_name not in directions:
#             continue
#         direction = directions[property_name]
#         for first_index in range(len(views)):
#             for second_index in range(first_index + 1, len(views)):
#                 view_1, view_2 = views[first_index], views[second_index]
#                 curves = []
#                 for view in [view_1, view_2]:
#                     subset = group[group["view"] == view].sort_values("property_scale")
#                     subset = subset[subset["embedding"].notna()]
#                     curve = np.stack(subset["embedding"].to_list()) @ direction
#                     curves.append((subset["property_scale"].to_numpy(float), curve))
#                 scale_1, curve_1 = curves[0]
#                 scale_2, curve_2 = curves[1]
#                 common = sorted(set(scale_1.tolist()) & set(scale_2.tolist()))
#                 if len(common) < 3:
#                     continue
#                 values_1 = np.array([curve_1[np.where(scale_1 == x)[0][0]] for x in common])
#                 values_2 = np.array([curve_2[np.where(scale_2 == x)[0][0]] for x in common])
#                 delta_1 = values_1 - values_1[0]
#                 delta_2 = values_2 - values_2[0]
#                 denominator = max(
#                     np.max(np.abs(delta_1)), np.max(np.abs(delta_2)), 1e-8
#                 )
#                 rows.append(
#                     {
#                         "canonical_object": canonical_object,
#                         "property": property_name,
#                         "view_1": view_1,
#                         "view_2": view_2,
#                         "paired_levels": len(common),
#                         "curve_spearman": safe_corr(spearmanr, delta_1, delta_2),
#                         "curve_pearson": safe_corr(pearsonr, delta_1, delta_2),
#                         "normalized_curve_rmse": float(
#                             np.sqrt(np.mean((delta_1 - delta_2) ** 2)) / denominator
#                         ),
#                         "endpoint_sign_agreement": bool(
#                             np.sign(delta_1[-1]) == np.sign(delta_2[-1])
#                         ),
#                     }
#                 )
#     return pd.DataFrame(rows)


# def save_full_direction_cosines(full, output_folder):
#     names = sorted(full)
#     matrix = np.stack([full[name] for name in names])
#     table = pd.DataFrame(matrix @ matrix.T, index=names, columns=names)
#     table.to_csv(output_folder / "recomputed_direction_cosine_matrix.csv")
#     return table


# def write_report(
#     output_folder,
#     inventory,
#     summary,
#     view_table,
#     direction_table,
#     saved_comparison,
# ):
#     lines = [
#         "# Existing Property-Scale Data: Leave-One-Object-Out Audit",
#         "",
#         "## Why this test",
#         "",
#         "The saved directions appear to have been estimated from these property-scale images. Scoring the same images with the saved direction is training-set evaluation. This report instead removes the held-out canonical object (including all supplied views), learns directions from the remaining objects, and measures the held-out trajectory.",
#         "",
#         "This establishes reproducibility, monotonicity, cross-property leakage, direction stability, and limited two-view consistency. It does not establish physical BRDF validity without Blender ground-truth shader values.",
#         "",
#         "## Inventory",
#         "",
#         f"- Image levels discovered: {len(inventory)}",
#         f"- Views: {inventory['view'].nunique()}",
#         f"- Canonical objects: {inventory['canonical_object'].nunique()}",
#         f"- Properties: {sorted(inventory['property'].unique().tolist())}",
#         "",
#         "## LOO property summary",
#         "",
#         "```text",
#         summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"),
#         "```",
#         "",
#         "Recommended interpretation:",
#         "",
#         "- median LOO Spearman near 1: held-out progression follows the learned axis;",
#         "- rank-1 fraction near 1: target response dominates other learned properties;",
#         "- selectivity near 1: little cross-property leakage;",
#         "- LOO direction cosine near 1: direction is not dominated by one object;",
#         "- outside-mask pass fraction 1: final composites are spatially confined.",
#         "",
#         "## Two-view consistency",
#         "",
#     ]
#     if view_table.empty:
#         lines.append(
#             "No paired canonical objects were detected across view roots. If object folder names differ, provide --object-map."
#         )
#     else:
#         lines.extend(
#             [
#                 f"- Paired curves: {len(view_table)}",
#                 f"- Median curve Spearman: {np.nanmedian(view_table['curve_spearman']):.3f}",
#                 f"- Endpoint sign agreement: {100*np.mean(view_table['endpoint_sign_agreement']):.1f}%",
#             ]
#         )
#     lines.extend(
#         [
#             "",
#             "## Direction geometry",
#             "",
#             "```text",
#             direction_table.to_string(float_format=lambda value: f"{value:.3f}"),
#             "```",
#             "",
#         ]
#     )
#     if not saved_comparison.empty:
#         lines.extend(
#             [
#                 "## Recomputed versus saved directions",
#                 "",
#                 "```text",
#                 saved_comparison.to_string(
#                     index=False, float_format=lambda value: f"{value:.3f}"
#                 ),
#                 "```",
#                 "",
#             ]
#         )
#     lines.extend(
#         [
#             "## Do you need more views now?",
#             "",
#             "No. First use these results to reject unstable or entangled properties. Two views support only a limited paired-view consistency statement. Generate additional held-out views later only for properties that pass LOO monotonicity and selectivity, and only if the paper keeps a viewpoint-generalization claim.",
#             "",
#             "## Remaining reviewer requirement",
#             "",
#             "To answer physical credibility, merge the original Blender shader parameter values with these images or with the original controlled renders. CLIP-on-CLIP LOO consistency is valuable but not an independent physical measurement.",
#         ]
#     )
#     (output_folder / "loo_audit_report.md").write_text(
#         "\n".join(lines), encoding="utf-8"
#     )


# def main():
#     args = parse_args()
#     roots = [str(Path(path)) for path in args.roots]
#     if args.view_names:
#         if len(args.view_names) != len(roots):
#             raise ValueError("--view-names must match --roots")
#         views = [str(value) for value in args.view_names]
#     else:
#         views = [Path(path).name for path in roots]

#     output_folder = Path(args.output)
#     output_folder.mkdir(parents=True, exist_ok=True)
#     rng = np.random.default_rng(args.seed)

#     inventory = scan_all(roots, views, load_object_map(args.object_map))
#     inventory.to_csv(output_folder / "discovered_property_scales.csv", index=False)

#     device = args.device
#     if device.startswith("cuda") and not torch.cuda.is_available():
#         print("CUDA unavailable; using CPU")
#         device = "cpu"
#     processor, model = load_encoder(args.image_encoder, device)
#     scored = encode_inventory(
#         inventory,
#         processor,
#         model,
#         device,
#         args.batch_size,
#         args.mask_threshold,
#         args.reference_padding,
#     )
#     # Embedding vectors are stored separately because array-valued CSV cells are
#     # unwieldy. The scalar inventory remains human-readable.
#     embedding_rows = scored[scored["embedding"].notna()].copy()
#     embedding_matrix = np.stack(embedding_rows["embedding"].to_list())
#     np.save(output_folder / "existing_clip_embeddings.npy", embedding_matrix)
#     scalar_scored = scored.drop(columns=["embedding"])
#     scalar_scored.to_csv(output_folder / "per_frame_inventory_and_pixels.csv", index=False)
#     embedding_index = embedding_rows[
#         ["view", "canonical_object", "object", "property", "frame", "property_scale"]
#     ].copy()
#     embedding_index["embedding_row"] = np.arange(len(embedding_index))
#     embedding_index.to_csv(output_folder / "embedding_index.csv", index=False)

#     endpoints = endpoint_table(scored)
#     endpoints.drop(columns=["endpoint_delta"]).to_csv(
#         output_folder / "endpoint_inventory.csv", index=False
#     )
#     loo_metrics, response_rows = evaluate_loo(scored, endpoints)
#     loo_metrics.to_csv(output_folder / "loo_per_sweep_metrics.csv", index=False)
#     response_rows.to_csv(output_folder / "loo_all_responses.csv", index=False)

#     stability, learning_curve, full = direction_stability(
#         endpoints, args.bootstrap_repeats, rng
#     )
#     stability.to_csv(output_folder / "loo_direction_stability.csv", index=False)
#     learning_curve.to_csv(
#         output_folder / "direction_sample_size_curve.csv", index=False
#     )

#     properties = [name for name in PROPERTY_ORDER if name in full]
#     response, normalized_response = response_matrices(response_rows, properties)
#     response.to_csv(output_folder / "loo_intervention_matrix.csv")
#     normalized_response.to_csv(
#         output_folder / "loo_intervention_matrix_row_normalized.csv"
#     )
#     summary = property_summary(loo_metrics, stability)
#     summary.to_csv(output_folder / "loo_property_summary.csv", index=False)

#     views_table = view_consistency(scored, endpoints)
#     views_table.to_csv(output_folder / "loo_two_view_consistency.csv", index=False)
#     direction_table = save_full_direction_cosines(full, output_folder)
#     saved_comparison = compare_saved(
#         full, load_saved_directions(args.saved_directions)
#     )
#     saved_comparison.to_csv(
#         output_folder / "recomputed_vs_saved_directions.csv", index=False
#     )

#     write_report(
#         output_folder,
#         inventory,
#         summary,
#         views_table,
#         direction_table,
#         saved_comparison,
#     )
#     with (output_folder / "audit_configuration.json").open(
#         "w", encoding="utf-8"
#     ) as handle:
#         json.dump(
#             {
#                 "roots": roots,
#                 "views": views,
#                 "image_encoder": args.image_encoder,
#                 "saved_directions": args.saved_directions,
#                 "canonical_objects": sorted(
#                     inventory["canonical_object"].unique().tolist()
#                 ),
#                 "properties": sorted(inventory["property"].unique().tolist()),
#                 "levels": int(len(inventory)),
#             },
#             handle,
#             indent=2,
#         )

#     print("\nLOO audit complete; no images were generated")
#     print(f"Report: {(output_folder / 'loo_audit_report.md').resolve()}")
#     print(f"Summary: {(output_folder / 'loo_property_summary.csv').resolve()}")


# if __name__ == "__main__":
#     main()


"""Deconfound four CLIP material directions using already-computed embeddings.

This script consumes the output of audit_existing_property_scale_folders_loo.py:

    existing_clip_embeddings.npy
    embedding_index.csv

It does not load diffusion, run IP-Adapter, encode images, or generate images.

For every object/view/property sweep, it estimates a CLIP tangent using all ten
property-scale levels. It then compares four direction constructions:

1. raw
   Mean normalized property tangent.

2. dual_ridge
   B (B^T B + lambda I)^-1. This is the regularized dual basis: movement along
   one exported vector is designed to have small projection on the other raw
   property axes.

3. residual_ridge
   For each property, regress its raw direction on the other directions and use
   the normalized residual.

4. lowdin_ridge
   Symmetric/Löwdin-style regularized orthogonalization, the closest symmetric
   decorrelation of the complete basis.

All methods are evaluated leave-one-canonical-object-out. The requested method
is then fitted using all current objects and saved as a plain dictionary of
(1, 1024) tensors compatible with the existing generation code.

Example:

python deconfound_four_clip_property_directions.py \
  --audit-root outputs/existing_property_scale_loo_audit \
  --properties albedo metalness rust paint_degradation \
  --ridge 0.05 \
  --export-method dual_ridge \
  --output outputs/four_property_deconfounding
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau, spearmanr


DEFAULT_PROPERTIES = [
    "albedo",
    "metalness",
    "rust",
    "paint_degradation",
]
METHODS = ["raw", "dual_ridge", "residual_ridge", "lowdin_ridge"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute and LOO-evaluate deconfounded CLIP directions"
    )
    parser.add_argument(
        "--audit-root",
        default="outputs/existing_property_scale_loo_audit",
    )
    parser.add_argument(
        "--properties",
        nargs="+",
        default=DEFAULT_PROPERTIES,
    )
    parser.add_argument("--ridge", type=float, default=0.05)
    parser.add_argument(
        "--export-method",
        choices=METHODS,
        default="dual_ridge",
    )
    parser.add_argument(
        "--output",
        default="outputs/four_property_deconfounding",
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Cannot normalize a zero/invalid vector")
    return vector / norm


def normalize_columns(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=0, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1e-12):
        raise ValueError("Direction construction produced a zero/invalid column")
    return matrix / norms


def safe_corr(function, x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return np.nan
    try:
        result = function(x, y)
        return float(result.statistic if hasattr(result, "statistic") else result[0])
    except Exception:
        return np.nan


def linear_r2(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return np.nan
    coefficients = np.polyfit(x, y, 1)
    prediction = np.polyval(coefficients, x)
    denominator = float(np.sum((y - y.mean()) ** 2))
    if denominator <= 1e-12:
        return np.nan
    return float(1.0 - np.sum((y - prediction) ** 2) / denominator)


def bootstrap_ci(values, repeats, rng):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    estimate = float(np.median(values))
    if len(values) == 1:
        return estimate, estimate, estimate
    samples = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        draw = rng.choice(values, size=len(values), replace=True)
        samples[index] = np.median(draw)
    low, high = np.percentile(samples, [2.5, 97.5])
    return estimate, float(low), float(high)


def load_audit_embeddings(audit_root, properties):
    audit_root = Path(audit_root)
    embedding_path = audit_root / "existing_clip_embeddings.npy"
    index_path = audit_root / "embedding_index.csv"
    if not embedding_path.is_file():
        raise FileNotFoundError(embedding_path)
    if not index_path.is_file():
        raise FileNotFoundError(index_path)

    embeddings = np.load(embedding_path).astype(np.float64)
    index = pd.read_csv(index_path)
    required = {
        "view",
        "canonical_object",
        "object",
        "property",
        "frame",
        "property_scale",
        "embedding_row",
    }
    missing = required - set(index.columns)
    if missing:
        raise ValueError(f"embedding_index.csv missing columns: {sorted(missing)}")
    index["embedding_row"] = pd.to_numeric(
        index["embedding_row"], errors="raise"
    ).astype(int)
    if index["embedding_row"].min() < 0:
        raise ValueError("Negative embedding row")
    if index["embedding_row"].max() >= len(embeddings):
        raise ValueError("Embedding index exceeds matrix length")

    index = index[index["property"].isin(properties)].copy()
    if index.empty:
        raise RuntimeError("None of the requested properties exist in the audit")
    missing_properties = sorted(set(properties) - set(index["property"].unique()))
    if missing_properties:
        raise RuntimeError(
            f"Requested properties absent from audit: {missing_properties}"
        )
    index["property_scale"] = pd.to_numeric(
        index["property_scale"], errors="raise"
    )
    return index, embeddings


def fit_sweep_slope(scales, embeddings):
    """OLS CLIP tangent d embedding / d property_scale."""

    scales = np.asarray(scales, dtype=np.float64)
    embeddings = np.asarray(embeddings, dtype=np.float64)
    order = np.argsort(scales)
    scales = scales[order]
    embeddings = embeddings[order]
    centered_scale = scales - scales.mean()
    denominator = float(np.sum(centered_scale ** 2))
    if denominator <= 1e-12:
        raise ValueError("A property sweep has no scale variation")
    centered_embeddings = embeddings - embeddings.mean(axis=0, keepdims=True)
    return (centered_scale[:, None] * centered_embeddings).sum(axis=0) / denominator


def build_sweeps(index, embeddings):
    sweep_rows = []
    frame_rows = []
    group_columns = ["view", "canonical_object", "object", "property"]
    for keys, group in index.groupby(group_columns):
        group = group.sort_values("property_scale")
        matrix = embeddings[group["embedding_row"].to_numpy(int)]
        scales = group["property_scale"].to_numpy(float)
        if len(group) < 3:
            continue
        slope = fit_sweep_slope(scales, matrix)
        endpoint = matrix[-1] - matrix[0]
        view, canonical_object, object_name, property_name = keys
        sweep_id = len(sweep_rows)
        sweep_rows.append(
            {
                "sweep_id": sweep_id,
                "view": str(view),
                "canonical_object": str(canonical_object),
                "object": str(object_name),
                "property": str(property_name),
                "slope": slope,
                "slope_norm": float(np.linalg.norm(slope)),
                "endpoint": endpoint,
                "endpoint_norm": float(np.linalg.norm(endpoint)),
                "frames": len(group),
            }
        )
        for local_index, (_, frame) in enumerate(group.iterrows()):
            frame_rows.append(
                {
                    "sweep_id": sweep_id,
                    "view": str(view),
                    "canonical_object": str(canonical_object),
                    "object": str(object_name),
                    "property": str(property_name),
                    "frame": int(frame["frame"]),
                    "property_scale": float(frame["property_scale"]),
                    "embedding": matrix[local_index],
                }
            )
    if not sweep_rows:
        raise RuntimeError("No complete property sweeps were found")
    return pd.DataFrame(sweep_rows), pd.DataFrame(frame_rows)


def mean_raw_basis(sweeps, properties, excluded_object=None):
    data = sweeps
    if excluded_object is not None:
        data = data[data["canonical_object"] != excluded_object]
    columns = []
    counts = {}
    for property_name in properties:
        group = data[data["property"] == property_name]
        if group.empty:
            raise RuntimeError(
                f"No training sweeps for {property_name} after excluding "
                f"{excluded_object}"
            )
        # First average multiple views per canonical object, then average objects.
        object_slopes = []
        for _, object_group in group.groupby("canonical_object"):
            object_slopes.append(np.stack(object_group["slope"].to_list()).mean(axis=0))
        mean_slope = np.stack(object_slopes).mean(axis=0)
        columns.append(unit(mean_slope))
        counts[property_name] = len(object_slopes)
    return np.stack(columns, axis=1), counts


def deconfound_bases(raw_basis, ridge):
    """Return every candidate basis as dimension x property matrices."""

    raw_basis = normalize_columns(raw_basis)
    property_count = raw_basis.shape[1]
    identity = np.eye(property_count, dtype=np.float64)
    gram = raw_basis.T @ raw_basis

    dual = raw_basis @ np.linalg.solve(gram + ridge * identity, identity)
    dual = normalize_columns(dual)

    residual_columns = []
    for target_index in range(property_count):
        other_indices = [
            index for index in range(property_count) if index != target_index
        ]
        target = raw_basis[:, target_index]
        others = raw_basis[:, other_indices]
        coefficients = np.linalg.solve(
            others.T @ others + ridge * np.eye(len(other_indices)),
            others.T @ target,
        )
        residual = target - others @ coefficients
        residual_columns.append(unit(residual))
    residual = np.stack(residual_columns, axis=1)

    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    inverse_sqrt = (
        eigenvectors
        @ np.diag(1.0 / np.sqrt(eigenvalues + ridge))
        @ eigenvectors.T
    )
    lowdin = normalize_columns(raw_basis @ inverse_sqrt)

    return {
        "raw": raw_basis,
        "dual_ridge": dual,
        "residual_ridge": residual,
        "lowdin_ridge": lowdin,
    }


def basis_diagnostics(raw_basis, candidate_basis, properties, method):
    cosine = candidate_basis.T @ candidate_basis
    response = raw_basis.T @ candidate_basis
    rows = []
    for target_index, target_property in enumerate(properties):
        retained_alignment = float(
            np.dot(raw_basis[:, target_index], candidate_basis[:, target_index])
        )
        off_response = np.delete(response[:, target_index], target_index)
        diagonal_response = float(response[target_index, target_index])
        largest_off = float(np.max(np.abs(off_response))) if len(off_response) else 0.0
        rows.append(
            {
                "method": method,
                "property": target_property,
                "raw_to_candidate_cosine": retained_alignment,
                "training_diagonal_response": diagonal_response,
                "training_largest_off_response": largest_off,
                "training_target_to_off_ratio": (
                    abs(diagonal_response) / largest_off
                    if largest_off > 1e-12 else np.inf
                ),
            }
        )
    return pd.DataFrame(rows), cosine, response


def evaluate_basis_on_object(
    frame_table,
    held_out_object,
    properties,
    method,
    basis,
):
    held_out = frame_table[
        frame_table["canonical_object"] == held_out_object
    ]
    metric_rows = []
    response_rows = []
    for keys, group in held_out.groupby(["view", "object", "property"]):
        view, object_name, target_property = keys
        if target_property not in properties:
            continue
        group = group.sort_values("property_scale")
        scales = group["property_scale"].to_numpy(float)
        embedding_matrix = np.stack(group["embedding"].to_list())
        score_matrix = embedding_matrix @ basis
        target_index = properties.index(target_property)
        target_curve = score_matrix[:, target_index]
        endpoint_responses = score_matrix[-1] - score_matrix[0]
        for measured_index, measured_property in enumerate(properties):
            response_rows.append(
                {
                    "method": method,
                    "held_out_object": held_out_object,
                    "view": view,
                    "object": object_name,
                    "target_property": target_property,
                    "measured_property": measured_property,
                    "endpoint_response": float(endpoint_responses[measured_index]),
                }
            )
        target_response = float(endpoint_responses[target_index])
        off_indices = [
            index for index in range(len(properties)) if index != target_index
        ]
        off_values = endpoint_responses[off_indices]
        largest_local_index = int(np.argmax(np.abs(off_values)))
        largest_index = off_indices[largest_local_index]
        largest_value = float(endpoint_responses[largest_index])
        total_absolute = float(np.sum(np.abs(endpoint_responses)))
        path_length = float(np.sum(np.abs(np.diff(target_curve))))
        metric_rows.append(
            {
                "method": method,
                "held_out_object": held_out_object,
                "view": view,
                "object": object_name,
                "property": target_property,
                "frames": len(group),
                "target_spearman": safe_corr(spearmanr, scales, target_curve),
                "target_kendall": safe_corr(kendalltau, scales, target_curve),
                "target_linear_r2": linear_r2(scales, target_curve),
                "monotonic_fraction": float(
                    np.mean(np.diff(target_curve) >= -1e-8)
                ),
                "target_endpoint_response": target_response,
                "selectivity": (
                    abs(target_response) / total_absolute
                    if total_absolute > 1e-12 else np.nan
                ),
                "target_rank": 1 + int(
                    np.sum(np.abs(off_values) > abs(target_response))
                ),
                "largest_off_target_property": properties[largest_index],
                "largest_off_target_response": largest_value,
                "target_to_largest_off_ratio": (
                    abs(target_response) / abs(largest_value)
                    if abs(largest_value) > 1e-12 else np.inf
                ),
                "path_efficiency": (
                    abs(target_response) / path_length
                    if path_length > 1e-12 else np.nan
                ),
            }
        )
    return metric_rows, response_rows


def outer_loo_evaluation(sweeps, frame_table, properties, ridge):
    metric_rows = []
    response_rows = []
    fold_direction_rows = []
    canonical_objects = sorted(sweeps["canonical_object"].unique())
    if len(canonical_objects) < 3:
        raise RuntimeError("At least three canonical objects are required for LOO")

    for held_out_object in canonical_objects:
        raw_basis, counts = mean_raw_basis(
            sweeps, properties, excluded_object=held_out_object
        )
        candidates = deconfound_bases(raw_basis, ridge)
        for method, basis in candidates.items():
            metrics, responses = evaluate_basis_on_object(
                frame_table,
                held_out_object,
                properties,
                method,
                basis,
            )
            metric_rows.extend(metrics)
            response_rows.extend(responses)
            for property_index, property_name in enumerate(properties):
                fold_direction_rows.append(
                    {
                        "held_out_object": held_out_object,
                        "method": method,
                        "property": property_name,
                        "training_objects": counts[property_name],
                        "direction": basis[:, property_index],
                    }
                )
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(response_rows),
        pd.DataFrame(fold_direction_rows),
    )


def aggregate_method_property(metrics, repeats, rng):
    rows = []
    grouped = metrics.groupby(["method", "property"])
    continuous_metrics = [
        "target_spearman",
        "target_kendall",
        "target_linear_r2",
        "monotonic_fraction",
        "selectivity",
        "target_to_largest_off_ratio",
        "path_efficiency",
    ]
    for (method, property_name), group in grouped:
        row = {
            "method": method,
            "property": property_name,
            "held_out_sweeps": len(group),
            "held_out_objects": group["held_out_object"].nunique(),
            "views": group["view"].nunique(),
            "target_rank_1_fraction": float(
                np.mean(group["target_rank"].to_numpy(float) == 1)
            ),
            "positive_endpoint_fraction": float(
                np.mean(group["target_endpoint_response"].to_numpy(float) > 0)
            ),
        }
        for metric in continuous_metrics:
            estimate, low, high = bootstrap_ci(
                group[metric].to_numpy(float), repeats, rng
            )
            row[f"median_{metric}"] = estimate
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["method", "property"])


def aggregate_method_summary(metrics):
    rows = []
    for method, group in metrics.groupby("method"):
        rows.append(
            {
                "method": method,
                "sweeps": len(group),
                "median_target_spearman": float(
                    np.nanmedian(group["target_spearman"])
                ),
                "median_monotonic_fraction": float(
                    np.nanmedian(group["monotonic_fraction"])
                ),
                "median_selectivity": float(np.nanmedian(group["selectivity"])),
                "target_rank_1_fraction": float(
                    np.mean(group["target_rank"].to_numpy(float) == 1)
                ),
                "positive_endpoint_fraction": float(
                    np.mean(group["target_endpoint_response"].to_numpy(float) > 0)
                ),
                "median_target_to_off_ratio": float(
                    np.nanmedian(group["target_to_largest_off_ratio"])
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["median_selectivity", "median_target_spearman"],
        ascending=False,
    )


def fold_direction_stability(fold_directions, full_candidates, properties):
    rows = []
    for record in fold_directions.to_dict("records"):
        property_index = properties.index(record["property"])
        reference = full_candidates[record["method"]][:, property_index]
        estimate = np.asarray(record["direction"], dtype=np.float64)
        rows.append(
            {
                "held_out_object": record["held_out_object"],
                "method": record["method"],
                "property": record["property"],
                "training_objects": record["training_objects"],
                "loo_to_full_cosine": float(np.dot(estimate, reference)),
            }
        )
    return pd.DataFrame(rows)


def response_matrix(response_rows, method, properties):
    subset = response_rows[response_rows["method"] == method]
    return subset.pivot_table(
        index="target_property",
        columns="measured_property",
        values="endpoint_response",
        aggfunc="median",
    ).reindex(index=properties, columns=properties)


def save_basis_files(full_candidates, properties, output_folder):
    for method, basis in full_candidates.items():
        payload = {
            property_name: torch.from_numpy(
                basis[:, property_index].astype(np.float32)
            ).reshape(1, -1)
            for property_index, property_name in enumerate(properties)
        }
        torch.save(payload, output_folder / f"{method}_property_directions.pt")


def write_report(
    output_folder,
    properties,
    ridge,
    export_method,
    method_summary,
    property_summary,
    diagnostics,
    stability,
):
    lines = [
        "# Four-Property CLIP Direction Deconfounding",
        "",
        "## Scope",
        "",
        f"Properties: {', '.join(properties)}",
        f"Regularization: {ridge}",
        f"Exported method: {export_method}",
        "",
        "This study uses only existing CLIP embeddings. It introduces no text prompts, RGB corrections, background compositing, diffusion generations, or physical labels.",
        "",
        "The LOO evaluation excludes every view of the held-out canonical object while estimating and deconfounding the directions.",
        "",
        "## Method comparison",
        "",
        "```text",
        method_summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"),
        "```",
        "",
        "## Per-property comparison",
        "",
        "```text",
        property_summary[
            [
                "method",
                "property",
                "median_target_spearman",
                "median_monotonic_fraction",
                "median_selectivity",
                "target_rank_1_fraction",
                "median_target_to_largest_off_ratio",
                "positive_endpoint_fraction",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.3f}"),
        "```",
        "",
        "## Semantic retention versus training separation",
        "",
        "```text",
        diagnostics.to_string(index=False, float_format=lambda value: f"{value:.3f}"),
        "```",
        "",
        "`raw_to_candidate_cosine` measures how much of the original property direction remains. A very small value means decorrelation may have removed much of the semantic signal. `training_target_to_off_ratio` measures separation on training objects; only LOO metrics should be used as evidence of generalization.",
        "",
        "## LOO direction stability",
        "",
        "```text",
        stability.groupby(["method", "property"])["loo_to_full_cosine"]
        .median()
        .reset_index()
        .to_string(index=False, float_format=lambda value: f"{value:.3f}"),
        "```",
        "",
        "## Export",
        "",
        f"The generation-compatible file is `{export_method}_property_directions.pt`. It is also copied to `deconfounded_property_directions.pt`.",
        "",
        "Do not claim physical BRDF validation from this result. It tests whether CLIP directions can be made more selective within the existing representation. Ground-truth Blender parameters are still required for physical correlation.",
    ]
    (output_folder / "deconfounding_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main():
    args = parse_args()
    properties = list(dict.fromkeys(args.properties))
    if len(properties) < 2:
        raise ValueError("At least two properties are required for deconfounding")
    if args.ridge < 0:
        raise ValueError("--ridge must be nonnegative")

    output_folder = Path(args.output)
    output_folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    index, embeddings = load_audit_embeddings(args.audit_root, properties)
    sweeps, frame_table = build_sweeps(index, embeddings)
    sweep_scalars = sweeps.drop(columns=["slope", "endpoint"])
    sweep_scalars.to_csv(output_folder / "sweep_tangent_inventory.csv", index=False)

    raw_full, object_counts = mean_raw_basis(sweeps, properties)
    full_candidates = deconfound_bases(raw_full, args.ridge)

    diagnostic_tables = []
    for method, candidate in full_candidates.items():
        diagnostics, cosine, training_response = basis_diagnostics(
            raw_full, candidate, properties, method
        )
        diagnostic_tables.append(diagnostics)
        pd.DataFrame(cosine, index=properties, columns=properties).to_csv(
            output_folder / f"{method}_direction_cosine_matrix.csv"
        )
        pd.DataFrame(
            training_response, index=properties, columns=properties
        ).to_csv(output_folder / f"{method}_training_response_matrix.csv")
    diagnostics = pd.concat(diagnostic_tables, ignore_index=True)
    diagnostics.to_csv(output_folder / "basis_diagnostics.csv", index=False)

    metrics, responses, fold_directions = outer_loo_evaluation(
        sweeps, frame_table, properties, args.ridge
    )
    metrics.to_csv(output_folder / "loo_all_sweep_metrics.csv", index=False)
    responses.to_csv(output_folder / "loo_all_cross_responses.csv", index=False)

    method_property_summary = aggregate_method_property(
        metrics, args.bootstrap_repeats, rng
    )
    method_property_summary.to_csv(
        output_folder / "loo_method_property_summary.csv", index=False
    )
    method_summary = aggregate_method_summary(metrics)
    method_summary.to_csv(output_folder / "loo_method_summary.csv", index=False)

    for method in METHODS:
        matrix = response_matrix(responses, method, properties)
        matrix.to_csv(output_folder / f"{method}_loo_response_matrix.csv")
        normalized = matrix.copy()
        for target in normalized.index:
            values = normalized.loc[target].to_numpy(float)
            denominator = np.nansum(np.abs(values))
            if denominator > 1e-12:
                normalized.loc[target] = values / denominator
        normalized.to_csv(
            output_folder / f"{method}_loo_response_matrix_row_normalized.csv"
        )

    stability = fold_direction_stability(
        fold_directions, full_candidates, properties
    )
    stability.to_csv(output_folder / "loo_deconfounded_direction_stability.csv", index=False)

    save_basis_files(full_candidates, properties, output_folder)
    selected_payload = {
        property_name: torch.from_numpy(
            full_candidates[args.export_method][:, property_index].astype(np.float32)
        ).reshape(1, -1)
        for property_index, property_name in enumerate(properties)
    }
    torch.save(
        selected_payload,
        output_folder / "deconfounded_property_directions.pt",
    )

    write_report(
        output_folder,
        properties,
        args.ridge,
        args.export_method,
        method_summary,
        method_property_summary,
        diagnostics,
        stability,
    )

    configuration = {
        "audit_root": str(Path(args.audit_root).resolve()),
        "properties": properties,
        "ridge": args.ridge,
        "export_method": args.export_method,
        "objects_per_property": object_counts,
        "embedding_dimension": int(embeddings.shape[1]),
        "canonical_objects": sorted(sweeps["canonical_object"].unique().tolist()),
        "views": sorted(sweeps["view"].unique().tolist()),
    }
    with (output_folder / "deconfounding_configuration.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(configuration, handle, indent=2)

    print("\nFour-property CLIP deconfounding complete")
    print("No images were generated and no non-CLIP correction was applied")
    print(f"Report: {(output_folder / 'deconfounding_report.md').resolve()}")
    print(f"Comparison: {(output_folder / 'loo_method_summary.csv').resolve()}")
    print(
        f"Exported directions: "
        f"{(output_folder / 'deconfounded_property_directions.pt').resolve()}"
    )


if __name__ == "__main__":
    main()
