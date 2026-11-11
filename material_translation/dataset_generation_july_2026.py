import argparse
import csv
import os
import random
import sys

import bpy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    import dataset_generator as dg
except ModuleNotFoundError:
    parent_dir = os.path.dirname(SCRIPT_DIR)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    import dataset_generator as dg


argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []

parser = argparse.ArgumentParser()
parser.add_argument("--object_root_dir", default="./blender_files/objects/all", help="Directory containing object folders")
parser.add_argument("--output_root_dir", default="./demo_assets/texture_dir/blender_output_july_2026", help="Where to save the generated images")
parser.add_argument("--hdr_path", default="./blender_files/scenes/studio_small_03_1k.hdr", help="Path to the HDR environment")
parser.add_argument("--render_resolution", type=int, default=512, help="Render resolution")
args = parser.parse_args(argv)

random.seed(42)

object_root_dir = args.object_root_dir
output_root_dir = args.output_root_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "incremental_dataset")
hdr_path = args.hdr_path
render_resolution = args.render_resolution

# Ordered least-to-most-affected (idx 0 -> 9 == value 0.0 -> 1.0); rust (and every other
# property here) walks less -> more of the effect as idx increases.
PROPERTY_NAMES = ["rust", "metalness", "roughness", "albedo", "glow", "transparency", "discoloration", "paint_degradation"]


def write_metadata(metadata_path, row):
    file_exists = os.path.exists(metadata_path)
    with open(metadata_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not file_exists:
            writer.writerow([
                "object_id",
                "source_image_path",
                "edited_image_path",
                "masked_image_path",
                "property_name",
                "source_property_value",
                "target_property_value",
                "view_id",
            ])
        writer.writerow(row)


def apply_property(main_obj, texture_dir, property_name, value):
    if property_name == "rust":
        dg.apply_rust(main_obj, texture_dir, value)
    elif property_name == "metalness":
        dg.apply_metal(main_obj, texture_dir, value)
    elif property_name == "roughness":
        dg.apply_roughness(main_obj, texture_dir, value)
    elif property_name == "albedo":
        dg.apply_albedo(main_obj, texture_dir, value)
    elif property_name == "glow":
        dg.apply_glow(main_obj, texture_dir, value)
    elif property_name == "transparency":
        dg.apply_transparency(main_obj, value)
    elif property_name == "discoloration":
        dg.apply_discoloration(main_obj, texture_dir, value)
    elif property_name == "paint_degradation":
        dg.apply_paint_degradation(main_obj, texture_dir, value)
    elif property_name == "scratches":
        dg.apply_scratches(main_obj, texture_dir, value)
    else:
        dg.apply_base_material(main_obj, texture_dir)


def process_object(object_dir):
    dir_name = os.path.basename(object_dir)
    blend_files = [entry for entry in sorted(os.listdir(object_dir)) if entry.endswith(".blend")]
    if not blend_files:
        print(f"[Skip] No .blend file found in {object_dir}")
        return

    blend_file_path = os.path.join(object_dir, blend_files[0])
    texture_dir = os.path.join(object_dir, "textures")
    if not os.path.exists(blend_file_path):
        print(f"[Skip] Missing blend file: {blend_file_path}")
        return
    if not os.path.isdir(texture_dir):
        print(f"[Skip] Missing texture directory: {texture_dir}")
        return
    if not hdr_path or not os.path.exists(hdr_path):
        print(f"[Skip] Missing HDR: {hdr_path}")
        return

    object_output_dir = os.path.join(output_root_dir, dir_name)
    os.makedirs(object_output_dir, exist_ok=True)

    print(f"Processing object: {dir_name}")

    bpy.ops.wm.read_homefile(use_empty=True)

    imported_objects = []
    try:
        with bpy.data.libraries.load(blend_file_path, link=False) as (data_from, data_to):
            data_to.objects = [name for name in data_from.objects if name]

        for obj in data_to.objects:
            if obj:
                bpy.context.collection.objects.link(obj)
                imported_objects.append(obj)
    except Exception as exc:
        print(f"[Warning] Failed to load blend file {blend_file_path}: {exc}")
        return

    main_obj = next((obj for obj in imported_objects if obj.type == "MESH"), None)
    if main_obj is None:
        print(f"[Skip] No mesh object found in {blend_file_path}")
        return

    dg.prepare_object(main_obj)
    camera = None
    light_obj = None
    bg_node = None
    camera, light_obj, bg_node = dg.ensure_scene_setup(main_obj, texture_dir, hdr_path, camera, light_obj)

    scene = bpy.context.scene
    scene.render.resolution_x = render_resolution
    scene.render.resolution_y = render_resolution
    scene.render.image_settings.file_format = "JPEG"
    scene.render.engine = "CYCLES"

    # The base/unedited material is identical for every property and every step of this
    # object (camera and lighting are also fixed once above), so render it exactly once
    # and reuse that single source image for every row in the metadata below.
    source_dir = os.path.join(object_output_dir, "source")
    os.makedirs(source_dir, exist_ok=True)
    source_path = os.path.join(source_dir, "source.jpg")

    dg.apply_base_material(main_obj, texture_dir)
    if not dg.render_image(scene, source_path):
        print(f"[Warning] Source render failed for {dir_name}")
        return
    source_property_value = 0.0

    for property_name in PROPERTY_NAMES:
        prop_dir = os.path.join(object_output_dir, property_name)
        edited_dir = os.path.join(prop_dir, "edited")
        mask_dir = os.path.join(prop_dir, "mask")
        os.makedirs(edited_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        for idx in range(10):
            # idx 0 -> 9 maps to value 0.0 -> 1.0, i.e. least -> most of the effect.
            value = idx / 9.0
            edited_path = os.path.join(edited_dir, f"{property_name}_{idx:02d}.jpg")
            mask_path = os.path.join(mask_dir, f"{property_name}_{idx:02d}_mask.jpg")

            dg.apply_base_material(main_obj, texture_dir)
            apply_property(main_obj, texture_dir, property_name, value)

            if dg.render_image(scene, edited_path):
                mask_rendered = dg.render_mask_image(scene, main_obj, mask_path, property_name, value)
                if not mask_rendered:
                    print(f"[Warning] Mask render failed for {dir_name}/{property_name}/{idx}")

                print(f"[{property_name}] {idx + 1}/10 -> {value:.2f} -> {edited_path}")
                write_metadata(
                    os.path.join(output_root_dir, "metadata.csv"),
                    [
                        dir_name,
                        os.path.abspath(source_path),
                        os.path.abspath(edited_path),
                        os.path.abspath(mask_path) if mask_rendered else "",
                        property_name,
                        f"{source_property_value:.2f}",
                        f"{value:.4f}",
                        idx,
                    ],
                )
            else:
                print(f"[Warning] Render failed for {dir_name}/{property_name}/{idx}")


def main():
    os.makedirs(output_root_dir, exist_ok=True)
    metadata_path = os.path.join(output_root_dir, "metadata.csv")
    if os.path.exists(metadata_path):
        os.remove(metadata_path)

    if not os.path.isdir(object_root_dir):
        print(f"[Skip] Object root directory not found: {object_root_dir}")
        return

    object_dirs = [
        os.path.join(object_root_dir, entry)
        for entry in sorted(os.listdir(object_root_dir))
        if os.path.isdir(os.path.join(object_root_dir, entry))
    ]

    for object_dir in object_dirs:
        process_object(object_dir)

    print("[Done] Incremental texture generation completed.")


if __name__ == "__main__":
    main()