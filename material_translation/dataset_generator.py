import argparse
import csv
import math
import os
import random
import sys
from mathutils import Vector

import bpy


argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []

parser = argparse.ArgumentParser()
parser.add_argument("--R", type=int, default=0, help="Red value (0-255)")
parser.add_argument("--G", type=int, default=0, help="Green value (0-255)")
parser.add_argument("--B", type=int, default=255, help="Blue value (0-255)")
parser.add_argument("--object_root_dir", default="./blender_files/objects/all", help="Directory containing .blend object folders")
parser.add_argument("--output_root_dir", default=None, help="Root output directory for the generated dataset")
parser.add_argument("--hdr_path", default="./blender_files/scenes/studio_small_03_1k.hdr", help="Path to the HDR environment texture")
parser.add_argument("--render_resolution", type=int, default=512, help="Render resolution")
parser.add_argument("--num_samples", type=int, default=20, help="Number of image pairs to generate")
args = parser.parse_args(argv)

random.seed(42)

NUM_SAMPLES = 20

rgb_color = (args.R, args.G, args.B)

# ====== CONFIGURATION ======
object_root_dir = args.object_root_dir
output_root_dir = args.output_root_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
render_resolution = args.render_resolution
use_hdr = True
hdr_path = args.hdr_path
NUM_SAMPLES = args.num_samples if args.num_samples is not None else NUM_SAMPLES

albedo_min_strength = 0.2
albedo_max_strength = 3.0
roughness_min_strength = 0.0
roughness_max_strength = 1.0
metal_texture_dir = "./blender_files/textures/Metal044A_1K-JPG"
metalness_min_strength = 0.0
metalness_max_strength = 5.0
rust_texture_dir = "./blender_files/textures/Rust_1K-JPG"
rust_min_strength = 0.0
rust_max_strength = 5.0

PROPERTY_NAMES = ["rust", "metalness", "roughness", "albedo", "glow", "transparency", "color", "discoloration", "paint_degradation", "scratches"]


def rgb255_to_float(rgb255):
    """Convert RGB values from 0-255 to 0.0-1.0 scale (Blender-friendly)."""
    return tuple(channel / 255.0 for channel in rgb255)


override_object_color = True
override_rgb_color = rgb255_to_float(rgb_color)


# ====== HELPERS ======
def interpolate(min_val, max_val, i, n):
    if n == 1:
        return (min_val + max_val) / 2
    return min_val + (max_val - min_val) * i / (n - 1)


def find_texture_file(texture_dir, keywords, valid_exts={".jpg", ".jpeg", ".png", ".tif", ".tiff", ".exr"}):
    if not texture_dir or not os.path.isdir(texture_dir):
        return None
    try:
        for file in os.listdir(texture_dir):
            if not any(file.lower().endswith(ext) for ext in valid_exts):
                continue
            if all(k.lower() in file.lower() for k in keywords):
                return os.path.join(texture_dir, file)
    except FileNotFoundError:
        return None
    return None


def setup_white_background():
    if not bpy.context.scene.world:
        bpy.context.scene.world = bpy.data.worlds.new("World")
    bpy.context.scene.world.use_nodes = True
    nodes = bpy.context.scene.world.node_tree.nodes
    links = bpy.context.scene.world.node_tree.links
    for n in nodes:
        nodes.remove(n)
    bg = nodes.new(type="ShaderNodeBackground")
    bg.inputs[0].default_value = (1, 1, 1, 1)
    out = nodes.new(type="ShaderNodeOutputWorld")
    links.new(bg.outputs[0], out.inputs[0])


def add_texture_node(nodes, path, label, color_space="sRGB"):
    if not path:
        return None
    node = nodes.new("ShaderNodeTexImage")
    node.label = label
    node.image = bpy.data.images.load(path)
    node.image.colorspace_settings.name = color_space
    return node


def setup_hdr_background(hdr_path):
    if not hdr_path or not os.path.exists(hdr_path):
        return None
    if not bpy.context.scene.world:
        bpy.context.scene.world = bpy.data.worlds.new("World")
    bpy.context.scene.world.use_nodes = True
    nodes = bpy.context.scene.world.node_tree.nodes
    links = bpy.context.scene.world.node_tree.links
    for n in nodes:
        nodes.remove(n)

    env_tex = nodes.new(type="ShaderNodeTexEnvironment")
    env_tex.image = bpy.data.images.load(hdr_path)

    bg = nodes.new(type="ShaderNodeBackground")
    out = nodes.new(type="ShaderNodeOutputWorld")
    links.new(env_tex.outputs["Color"], bg.inputs["Color"])
    links.new(bg.outputs["Background"], out.inputs["Surface"])

    return bg


def setup_material(texture_dir, albedo_strength=1.0, roughness_strength=1.0, override_base_color=None, color_strength=1.0):
    mat = bpy.data.materials.new(name="PBRMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    for node in list(nodes):
        if node.name not in {"Principled BSDF", "Material Output"}:
            nodes.remove(node)
    bsdf = nodes.get("Principled BSDF")

    base_path = find_texture_file(texture_dir, ["diff"])
    base_node = add_texture_node(nodes, base_path, "Base", "sRGB")

    if override_base_color is not None:
        rgb_node = nodes.new(type="ShaderNodeRGB")
        rgb_node.outputs[0].default_value = (*override_base_color, 1.0)
        bpy.context.scene.view_settings.view_transform = "Standard"
        bpy.context.scene.view_settings.look = "None"

        if base_node:
            mix_node = nodes.new(type="ShaderNodeMixRGB")
            mix_node.blend_type = "MIX"
            mix_node.inputs["Fac"].default_value = color_strength
            links.new(rgb_node.outputs["Color"], mix_node.inputs[1])
            links.new(mix_node.outputs["Color"], bsdf.inputs["Base Color"])
        else:
            links.new(rgb_node.outputs["Color"], bsdf.inputs["Base Color"])

    elif base_node:
        mult_node = nodes.new(type="ShaderNodeMixRGB")
        mult_node.blend_type = "MULTIPLY"
        mult_node.inputs[0].default_value = 1.0
        mult_node.inputs[2].default_value = (albedo_strength, albedo_strength, albedo_strength, 1)
        links.new(base_node.outputs["Color"], mult_node.inputs[1])
        links.new(mult_node.outputs["Color"], bsdf.inputs["Base Color"])

    rough_path = find_texture_file(texture_dir, ["rough"])
    rough_node = add_texture_node(nodes, rough_path, "Roughness", "Non-Color")
    if rough_node:
        mult_rough = nodes.new(type="ShaderNodeMath")
        mult_rough.operation = "MULTIPLY"
        mult_rough.inputs[1].default_value = roughness_strength
        links.new(rough_node.outputs["Color"], mult_rough.inputs[0])
        links.new(mult_rough.outputs["Value"], bsdf.inputs["Roughness"])

    metal = add_texture_node(nodes, find_texture_file(texture_dir, ["metal"]), "Metallic", "Non-Color")
    if metal:
        links.new(metal.outputs["Color"], bsdf.inputs["Metallic"])

    normal = add_texture_node(nodes, find_texture_file(texture_dir, ["nor", "gl"]), "Normal", "Non-Color")
    if normal:
        nrm_map = nodes.new("ShaderNodeNormalMap")
        links.new(normal.outputs["Color"], nrm_map.inputs["Color"])
        links.new(nrm_map.outputs["Normal"], bsdf.inputs["Normal"])

    return mat


def setup_glass_material(name="GlassMaterial", base_color=(0.8, 0.9, 1.0), transparency=0.1, roughness=0.0, ior=1.45):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    for node in list(nodes):
        if node.name != "Material Output":
            nodes.remove(node)

    output_node = nodes.get("Material Output")
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (-200, 0)

    rgba = (*base_color, 1.0 - transparency)
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Alpha"].default_value = rgba[3]
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["IOR"].default_value = ior

    if "Transmission" in bsdf.inputs:
        bsdf.inputs["Transmission"].default_value = 1.0
    elif "Transmission Weight" in bsdf.inputs:
        bsdf.inputs["Transmission Weight"].default_value = 1.0

    links.new(bsdf.outputs["BSDF"], output_node.inputs["Surface"])

    mat.blend_method = "BLEND"
    mat.use_backface_culling = False

    if bpy.context.scene.render.engine == "BLENDER_EEVEE" and hasattr(mat, "shadow_method"):
        mat.shadow_method = "HASHED"

    return mat


def setup_material_metal(texture_dir, metalness_strength=1.0):
    mat = bpy.data.materials.new(name="PBRMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    for node in list(nodes):
        if node.name not in {"Principled BSDF", "Material Output"}:
            nodes.remove(node)
    bsdf = nodes.get("Principled BSDF")

    base_color_path = find_texture_file(texture_dir, ["Color"])
    roughness_path = find_texture_file(texture_dir, ["Roughness"])
    metalness_path = find_texture_file(texture_dir, ["Metalness"])
    normal_path = find_texture_file(texture_dir, ["NormalGL", "Normal"])

    if base_color_path:
        base_node = add_texture_node(nodes, base_color_path, "Base Color", "sRGB")
        links.new(base_node.outputs["Color"], bsdf.inputs["Base Color"])

    if roughness_path:
        rough_node = add_texture_node(nodes, roughness_path, "Roughness", "Non-Color")
        mult_rough = nodes.new(type="ShaderNodeMath")
        mult_rough.operation = "MULTIPLY"
        mult_rough.inputs[1].default_value = metalness_strength
        links.new(rough_node.outputs["Color"], mult_rough.inputs[0])
        links.new(mult_rough.outputs["Value"], bsdf.inputs["Roughness"])

    if metalness_path:
        metal_node = add_texture_node(nodes, metalness_path, "Metallic", "Non-Color")
        links.new(metal_node.outputs["Color"], bsdf.inputs["Metallic"])

    if normal_path:
        normal_node = add_texture_node(nodes, normal_path, "Normal", "Non-Color")
        normal_map_node = nodes.new(type="ShaderNodeNormalMap")
        links.new(normal_node.outputs["Color"], normal_map_node.inputs["Color"])
        links.new(normal_map_node.outputs["Normal"], bsdf.inputs["Normal"])

    return mat


def compute_rust_material_state(rust_strength):
    strength = max(0.0, min(float(rust_strength), 5.0)) / 5.0
    return {
        "base_color_mix": strength,
        "roughness": 0.35 + strength * 0.45,
        "metallic": 0.10 + strength * 0.25,
        "normal_strength": strength,
    }


def setup_material_rust(texture_dir, rust_strength=1.0):
    mat = bpy.data.materials.new(name="RustMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    for node in list(nodes):
        if node.name not in {"Principled BSDF", "Material Output"}:
            nodes.remove(node)

    bsdf = nodes.get("Principled BSDF")
    rust_state = compute_rust_material_state(rust_strength)

    base_color_path = os.path.join(texture_dir, "Rust009_1K-JPG_Color.jpg")
    if os.path.exists(base_color_path):
        base_node = add_texture_node(nodes, base_color_path, "RustColor", "sRGB")
        rust_color_node = nodes.new(type="ShaderNodeRGB")
        rust_color_node.outputs[0].default_value = (0.4, 0.2, 0.1, 1.0)
        mix_node = nodes.new(type="ShaderNodeMixRGB")
        mix_node.blend_type = "MIX"
        mix_node.inputs["Fac"].default_value = rust_state["base_color_mix"]
        links.new(rust_color_node.outputs["Color"], mix_node.inputs[1])
        links.new(base_node.outputs["Color"], mix_node.inputs[2])
        links.new(mix_node.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        bsdf.inputs["Base Color"].default_value = (0.4, 0.2, 0.1, 1.0)

    roughness_path = os.path.join(texture_dir, "Rust009_1K-JPG_Roughness.jpg")
    if os.path.exists(roughness_path):
        rough_node = add_texture_node(nodes, roughness_path, "RustRoughness", "Non-Color")
        links.new(rough_node.outputs["Color"], bsdf.inputs["Roughness"])
    else:
        bsdf.inputs["Roughness"].default_value = rust_state["roughness"]

    normal_path = os.path.join(texture_dir, "Rust009_1K-JPG_NormalGL.jpg")
    if os.path.exists(normal_path):
        normal_node = add_texture_node(nodes, normal_path, "RustNormal", "Non-Color")
        normal_map_node = nodes.new(type="ShaderNodeNormalMap")
        links.new(normal_node.outputs["Color"], normal_map_node.inputs["Color"])

        scale_node = nodes.new(type="ShaderNodeVectorMath")
        scale_node.operation = "SCALE"
        scale_node.inputs[1].default_value = (
            rust_state["normal_strength"],
            rust_state["normal_strength"],
            rust_state["normal_strength"],
        )
        links.new(normal_map_node.outputs["Normal"], scale_node.inputs[0])
        links.new(scale_node.outputs["Vector"], bsdf.inputs["Normal"])
    else:
        print(f"[Warning] Rust normal map not found in {texture_dir}.")

    bsdf.inputs["Metallic"].default_value = rust_state["metallic"]
    return mat


def setup_material_glow(texture_dir, glow_strength=10.0):
    mat = bpy.data.materials.new(name="PBR_Glow")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    for node in list(nodes):
        if node.name not in {"Material Output", "Principled BSDF"}:
            nodes.remove(node)

    output = nodes.get("Material Output")
    bsdf = nodes.get("Principled BSDF")

    base_path = find_texture_file(texture_dir, ["diff"])
    base_node = add_texture_node(nodes, base_path, "Albedo", "sRGB")
    if base_node:
        links.new(base_node.outputs["Color"], bsdf.inputs["Base Color"])

    rough_path = find_texture_file(texture_dir, ["rough"])
    if rough_path:
        rough_node = add_texture_node(nodes, rough_path, "Roughness", "Non-Color")
        links.new(rough_node.outputs["Color"], bsdf.inputs["Roughness"])

    normal_path = find_texture_file(texture_dir, ["nor", "gl"])
    if normal_path:
        normal_node = add_texture_node(nodes, normal_path, "Normal", "Non-Color")
        normal_map = nodes.new(type="ShaderNodeNormalMap")
        links.new(normal_node.outputs["Color"], normal_map.inputs["Color"])
        links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])

    emission_node = nodes.new(type="ShaderNodeEmission")
    emission_node.inputs["Strength"].default_value = glow_strength
    if base_node:
        links.new(base_node.outputs["Color"], emission_node.inputs["Color"])

    add_shader = nodes.new(type="ShaderNodeAddShader")
    links.new(bsdf.outputs["BSDF"], add_shader.inputs[0])
    links.new(emission_node.outputs["Emission"], add_shader.inputs[1])
    links.new(add_shader.outputs["Shader"], output.inputs["Surface"])

    return mat


def build_discoloration_mask(nodes, links, strength):
    """Build the noise-based blotch mask used by discoloration; returns a 0..1 Value output socket."""
    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    noise = nodes.new(type="ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 4.0
    noise.inputs["Detail"].default_value = 6.0
    links.new(tex_coord.outputs["Object"], noise.inputs["Vector"])

    ramp = nodes.new(type="ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.35
    ramp.color_ramp.elements[1].position = 0.65
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])

    mask_scale = nodes.new(type="ShaderNodeMath")
    mask_scale.operation = "MULTIPLY"
    mask_scale.inputs[1].default_value = strength
    links.new(ramp.outputs["Color"], mask_scale.inputs[0])

    return mask_scale.outputs["Value"]


def setup_material_discoloration(texture_dir, discoloration_strength=0.0):
    """Blotchy, uneven staining/yellowing layered over the base material; grows with strength."""
    strength = max(0.0, min(float(discoloration_strength), 1.0))

    mat = bpy.data.materials.new(name="DiscolorationMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    for node in list(nodes):
        if node.name not in {"Principled BSDF", "Material Output"}:
            nodes.remove(node)
    bsdf = nodes.get("Principled BSDF")

    base_path = find_texture_file(texture_dir, ["diff"])
    base_node = add_texture_node(nodes, base_path, "Base", "sRGB")

    mask_value = build_discoloration_mask(nodes, links, strength)

    stain_color = nodes.new(type="ShaderNodeRGB")
    stain_color.outputs[0].default_value = (0.42, 0.35, 0.18, 1.0)

    mix_node = nodes.new(type="ShaderNodeMixRGB")
    mix_node.blend_type = "MIX"
    links.new(mask_value, mix_node.inputs["Fac"])
    if base_node:
        links.new(base_node.outputs["Color"], mix_node.inputs[1])
    else:
        mix_node.inputs[1].default_value = (0.8, 0.8, 0.8, 1.0)
    links.new(stain_color.outputs["Color"], mix_node.inputs[2])
    links.new(mix_node.outputs["Color"], bsdf.inputs["Base Color"])

    rough_path = find_texture_file(texture_dir, ["rough"])
    rough_node = add_texture_node(nodes, rough_path, "Roughness", "Non-Color")
    if rough_node:
        links.new(rough_node.outputs["Color"], bsdf.inputs["Roughness"])
    else:
        bsdf.inputs["Roughness"].default_value = 0.5 + 0.3 * strength

    return mat


def build_paint_degradation_mask(nodes, links, strength):
    """Build the Voronoi chip mask used by paint_degradation; returns a 0..1 Color output socket."""
    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    voronoi = nodes.new(type="ShaderNodeTexVoronoi")
    voronoi.inputs["Scale"].default_value = 8.0
    links.new(tex_coord.outputs["Object"], voronoi.inputs["Vector"])

    ramp = nodes.new(type="ShaderNodeValToRGB")
    low_pos = max(0.0, 0.75 - strength * 0.6)
    ramp.color_ramp.elements[0].position = low_pos
    ramp.color_ramp.elements[1].position = min(1.0, low_pos + 0.08)
    links.new(voronoi.outputs["Distance"], ramp.inputs["Fac"])

    return ramp.outputs["Color"]


def setup_material_paint_degradation(texture_dir, degradation_strength=0.0):
    """Chipped/peeling paint revealing a worn substrate beneath; chipped area grows with strength."""
    strength = max(0.0, min(float(degradation_strength), 1.0))

    mat = bpy.data.materials.new(name="PaintDegradationMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    for node in list(nodes):
        if node.name not in {"Principled BSDF", "Material Output"}:
            nodes.remove(node)
    bsdf = nodes.get("Principled BSDF")

    base_path = find_texture_file(texture_dir, ["diff"])
    base_node = add_texture_node(nodes, base_path, "PaintBase", "sRGB")

    mask_color = build_paint_degradation_mask(nodes, links, strength)

    substrate_color = nodes.new(type="ShaderNodeRGB")
    substrate_color.outputs[0].default_value = (0.32, 0.30, 0.28, 1.0)

    mix_node = nodes.new(type="ShaderNodeMixRGB")
    mix_node.blend_type = "MIX"
    links.new(mask_color, mix_node.inputs["Fac"])
    if base_node:
        links.new(base_node.outputs["Color"], mix_node.inputs[1])
    else:
        mix_node.inputs[1].default_value = (0.75, 0.75, 0.78, 1.0)
    links.new(substrate_color.outputs["Color"], mix_node.inputs[2])
    links.new(mix_node.outputs["Color"], bsdf.inputs["Base Color"])

    rough_mix = nodes.new(type="ShaderNodeMath")
    rough_mix.operation = "MULTIPLY_ADD"
    rough_mix.inputs[1].default_value = 0.5
    rough_mix.inputs[2].default_value = 0.3
    links.new(mask_color, rough_mix.inputs[0])
    links.new(rough_mix.outputs["Value"], bsdf.inputs["Roughness"])

    bsdf.inputs["Metallic"].default_value = 0.05 + strength * 0.15

    return mat


def build_scratch_mask(nodes, links, strength):
    """Build the Voronoi edge-distance scratch-line mask; returns a 0..1 Value output socket."""
    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    voronoi = nodes.new(type="ShaderNodeTexVoronoi")
    if hasattr(voronoi, "feature"):
        voronoi.feature = "DISTANCE_TO_EDGE"
    voronoi.inputs["Scale"].default_value = 15.0 + strength * 25.0
    links.new(tex_coord.outputs["Object"], voronoi.inputs["Vector"])

    ramp = nodes.new(type="ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[1].position = max(0.015, 0.12 - strength * 0.1)
    links.new(voronoi.outputs["Distance"], ramp.inputs["Fac"])

    invert = nodes.new(type="ShaderNodeInvert")
    links.new(ramp.outputs["Color"], invert.inputs["Color"])

    mask_scale = nodes.new(type="ShaderNodeMath")
    mask_scale.operation = "MULTIPLY"
    mask_scale.inputs[1].default_value = strength
    links.new(invert.outputs["Color"], mask_scale.inputs[0])

    return mask_scale.outputs["Value"]


def setup_material_scratch(texture_dir, scratch_strength=0.0):
    """Thin bright scratch lines (using Voronoi cell-edge distance) cutting through the base material; density and visibility grow with strength."""
    strength = max(0.0, min(float(scratch_strength), 1.0))

    mat = bpy.data.materials.new(name="ScratchMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    for node in list(nodes):
        if node.name not in {"Principled BSDF", "Material Output"}:
            nodes.remove(node)
    bsdf = nodes.get("Principled BSDF")

    base_path = find_texture_file(texture_dir, ["diff"])
    base_node = add_texture_node(nodes, base_path, "ScratchBase", "sRGB")

    mask_value = build_scratch_mask(nodes, links, strength)

    scratch_color = nodes.new(type="ShaderNodeRGB")
    scratch_color.outputs[0].default_value = (0.85, 0.85, 0.85, 1.0)

    color_mix = nodes.new(type="ShaderNodeMixRGB")
    color_mix.blend_type = "MIX"
    links.new(mask_value, color_mix.inputs["Fac"])
    if base_node:
        links.new(base_node.outputs["Color"], color_mix.inputs[1])
    else:
        color_mix.inputs[1].default_value = (0.6, 0.6, 0.6, 1.0)
    links.new(scratch_color.outputs["Color"], color_mix.inputs[2])
    links.new(color_mix.outputs["Color"], bsdf.inputs["Base Color"])

    rough_path = find_texture_file(texture_dir, ["rough"])
    rough_node = add_texture_node(nodes, rough_path, "Roughness", "Non-Color")
    rough_mix = nodes.new(type="ShaderNodeMixRGB")
    rough_mix.blend_type = "MIX"
    links.new(mask_value, rough_mix.inputs["Fac"])
    if rough_node:
        links.new(rough_node.outputs["Color"], rough_mix.inputs[1])
    else:
        rough_mix.inputs[1].default_value = (0.5, 0.5, 0.5, 1.0)
    rough_mix.inputs[2].default_value = (0.1, 0.1, 0.1, 1.0)
    links.new(rough_mix.outputs["Color"], bsdf.inputs["Roughness"])

    return mat


def compute_object_center_and_radius(obj):
    local_bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    center = sum(local_bbox, Vector()) / 8
    radius = max((v - center).length for v in local_bbox)
    return center, radius


def setup_camera_dynamic(obj, fov_deg=50):
    from math import radians, tan

    center, radius = compute_object_center_and_radius(obj)

    cam_data = bpy.data.cameras.new(name="Camera")
    cam_data.sensor_fit = "AUTO"
    cam_data.lens_unit = "FOV"
    cam_data.angle = radians(fov_deg)

    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    fov_rad = cam_data.angle
    distance = radius / tan(fov_rad / 2)

    cam.location = center + Vector((0.0, -distance, 0.0))

    constraint = cam.constraints.new(type="TRACK_TO")
    constraint.target = obj
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"

    return cam


def setup_light(center, energy=1000):
    light_data = bpy.data.lights.new(name="Light", type="POINT")
    light_data.energy = energy
    light = bpy.data.objects.new(name="Light", object_data=light_data)
    light.location = center + Vector((2, -2, 3))
    bpy.context.collection.objects.link(light)
    return light


def configure_cycles():
    try:
        prefs = bpy.context.preferences
        if "cycles" in prefs.addons:
            cycles_prefs = prefs.addons["cycles"].preferences
            cycles_prefs.compute_device_type = "CUDA"
            cycles_prefs.get_devices()
            for device in cycles_prefs.devices:
                device.use = True
            bpy.context.scene.cycles.device = "GPU"
    except Exception:
        pass

    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.render.resolution_x = render_resolution
    bpy.context.scene.render.resolution_y = render_resolution
    bpy.context.scene.render.image_settings.file_format = "JPEG"


def prepare_object(obj):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    obj.location = (0, 0, 0)
    obj.scale = (1, 1, 1)


def assign_material(obj, material):
    if not material:
        return
    if len(obj.data.materials):
        obj.data.materials[0] = material
    else:
        obj.data.materials.append(material)


def ensure_scene_setup(obj, texture_dir, hdr_path, camera, light):
    if camera is None:
        camera = setup_camera_dynamic(obj)
    if light is None:
        center, _ = compute_object_center_and_radius(obj)
        light = setup_light(center, energy=1000)

    if use_hdr and hdr_path and os.path.exists(hdr_path):
        bg_node = setup_hdr_background(hdr_path)
    else:
        setup_white_background()
        bg_node = None

    configure_cycles()
    return camera, light, bg_node


def render_image(scene, output_path):
    scene.render.filepath = output_path
    try:
        bpy.ops.render.render(write_still=True)
        return os.path.exists(output_path)
    except Exception as exc:
        print(f"[Error] Render failed for {output_path}: {exc}")
        return False


def get_or_create_black_world():
    """A dedicated black-background world used for mask renders, kept separate from the HDR/white world."""
    world = bpy.data.worlds.get("MaskBlackWorld")
    if world is None:
        world = bpy.data.worlds.new("MaskBlackWorld")
        world.use_nodes = True
        nodes = world.node_tree.nodes
        links = world.node_tree.links
        for n in list(nodes):
            nodes.remove(n)
        bg = nodes.new(type="ShaderNodeBackground")
        bg.inputs[0].default_value = (0, 0, 0, 1)
        out = nodes.new(type="ShaderNodeOutputWorld")
        links.new(bg.outputs[0], out.inputs[0])
    return world


def build_mask_material(property_name, target_property_value):
    """Black/white mask material for the given edit: a soft grayscale mask for spatially-patchy
    properties (using the same node graph as the edit itself), or a flat white silhouette for
    properties that uniformly affect the whole object."""
    mat = bpy.data.materials.new(name=f"Mask_{property_name}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for node in list(nodes):
        nodes.remove(node)

    output_node = nodes.new(type="ShaderNodeOutputMaterial")
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.0

    try:
        strength = max(0.0, min(float(target_property_value), 1.0))
    except (TypeError, ValueError):
        strength = 0.0

    if property_name == "discoloration":
        mask_socket = build_discoloration_mask(nodes, links, strength)
        links.new(mask_socket, emission.inputs["Color"])
    elif property_name == "paint_degradation":
        mask_socket = build_paint_degradation_mask(nodes, links, strength)
        links.new(mask_socket, emission.inputs["Color"])
    elif property_name == "scratches":
        mask_socket = build_scratch_mask(nodes, links, strength)
        links.new(mask_socket, emission.inputs["Color"])
    else:
        # Whole-object edits (rust, metalness, roughness, albedo, glow, transparency, color):
        # the entire silhouette is the affected region.
        emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)

    links.new(emission.outputs["Emission"], output_node.inputs["Surface"])
    return mat


def render_mask_image(scene, obj, output_path, property_name, target_property_value):
    """Render a black/white mask of the region affected by the given edit, then restore the
    object's material and the scene's world so subsequent renders are unaffected."""
    original_material = obj.data.materials[0] if obj.data.materials else None
    original_world = scene.world

    mask_material = build_mask_material(property_name, target_property_value)
    assign_material(obj, mask_material)
    scene.world = get_or_create_black_world()

    success = render_image(scene, output_path)

    if original_material is not None:
        obj.data.materials[0] = original_material
    scene.world = original_world

    return success


def apply_base_material(obj, texture_dir):
    material = setup_material(texture_dir, albedo_strength=1.0, roughness_strength=1.0, override_base_color=None)
    assign_material(obj, material)
    return material


def apply_rust(obj, texture_dir, level):
    rust_dir = rust_texture_dir if rust_texture_dir and os.path.isdir(rust_texture_dir) else texture_dir
    material = setup_material_rust(rust_dir, rust_strength=level * 5.0)
    assign_material(obj, material)
    return material


def apply_metal(obj, texture_dir, level):
    metal_dir = metal_texture_dir if metal_texture_dir and os.path.isdir(metal_texture_dir) else texture_dir
    material = setup_material_metal(metal_dir, metalness_strength=level * 5.0)
    assign_material(obj, material)
    return material


def apply_roughness(obj, texture_dir, level):
    material = setup_material(texture_dir, albedo_strength=1.0, roughness_strength=level, override_base_color=None)
    assign_material(obj, material)
    return material


def apply_glow(obj, texture_dir, level):
    material = setup_material_glow(texture_dir, glow_strength=level * 50.0)
    assign_material(obj, material)
    return material


def apply_transparency(obj, level):
    material = setup_glass_material(base_color=(1.0, 1.0, 1.0), transparency=level, roughness=0.0)
    assign_material(obj, material)
    return material


def apply_albedo(obj, texture_dir, level):
    actual_albedo = 0.2 + level * (3.0 - 0.2)
    material = setup_material(texture_dir, albedo_strength=actual_albedo, roughness_strength=1.0, override_base_color=None)
    assign_material(obj, material)
    return material


def apply_color(obj, texture_dir, level):
    random_color = (random.random(), random.random(), random.random())
    material = setup_material(
        texture_dir,
        albedo_strength=1.0,
        roughness_strength=1.0,
        override_base_color=random_color if override_object_color else None,
        color_strength=1.0,
    )
    assign_material(obj, material)
    return random_color


def apply_discoloration(obj, texture_dir, level):
    material = setup_material_discoloration(texture_dir, discoloration_strength=level)
    assign_material(obj, material)
    return material


def apply_paint_degradation(obj, texture_dir, level):
    material = setup_material_paint_degradation(texture_dir, degradation_strength=level)
    assign_material(obj, material)
    return material


def apply_scratches(obj, texture_dir, level):
    material = setup_material_scratch(texture_dir, scratch_strength=level)
    assign_material(obj, material)
    return material


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


# ====== MAIN PROCESS ======
def main():
    os.makedirs(output_root_dir, exist_ok=True)

    metadata_path = os.path.join(output_root_dir, "metadata.csv")

    if not os.path.isdir(object_root_dir):
        print(f"[Skip] Object root directory not found: {object_root_dir}")
        return

    object_dirs = [
        os.path.join(object_root_dir, entry)
        for entry in sorted(os.listdir(object_root_dir))
        if os.path.isdir(os.path.join(object_root_dir, entry))
    ]

    if not object_dirs:
        print(f"[Skip] No object directories found in {object_root_dir}")
        return

    for object_dir in object_dirs:
        dir_name = os.path.basename(object_dir)
        blend_files = [entry for entry in sorted(os.listdir(object_dir)) if entry.endswith(".blend")]
        if not blend_files:
            print(f"[Skip] No .blend file found in {object_dir}")
            continue

        blend_file_path = os.path.join(object_dir, blend_files[0])
        texture_dir = os.path.join(object_dir, "textures")
        if not os.path.exists(blend_file_path):
            print(f"[Skip] Missing blend file: {blend_file_path}")
            continue
        if not os.path.isdir(texture_dir):
            print(f"[Skip] Missing texture directory: {texture_dir}")
            continue

        if use_hdr and (not hdr_path or not os.path.exists(hdr_path)):
            print(f"[Skip] Missing HDR: {hdr_path}")
            continue

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
            continue

        main_obj = next((obj for obj in imported_objects if obj.type == "MESH"), None)
        if main_obj is None:
            print(f"[Skip] No mesh object found in {blend_file_path}")
            continue

        prepare_object(main_obj)

        camera = None
        light_obj = None
        bg_node = None
        camera, light_obj, bg_node = ensure_scene_setup(main_obj, texture_dir, hdr_path, camera, light_obj)

        if not main_obj.data.materials:
            apply_base_material(main_obj, texture_dir)
        if not main_obj.data.materials:
            print(f"[Skip] No material could be created for {dir_name}")
            continue

        scene = bpy.context.scene
        original_energy = light_obj.data.energy if light_obj and light_obj.data else 1000

        for sample_idx in range(1, NUM_SAMPLES + 1):
            property_name = random.choice(PROPERTY_NAMES)
            source_property_value = 0.0

            source_path = os.path.join(object_output_dir, f"source_{sample_idx:06d}.jpg")
            edit_path = os.path.join(object_output_dir, f"edit_{sample_idx:06d}.jpg")
            mask_path = os.path.join(object_output_dir, f"mask_{sample_idx:06d}.jpg")

            if property_name == "rust":
                target_property_value = random.uniform(0.0, 1.0)
                target_value_for_log = target_property_value
            elif property_name == "metalness":
                target_property_value = random.uniform(0.0, 1.0)
                target_value_for_log = target_property_value
            elif property_name == "roughness":
                target_property_value = random.uniform(0.0, 1.0)
                target_value_for_log = target_property_value
            elif property_name == "albedo":
                target_property_value = random.uniform(0.0, 1.0)
                target_value_for_log = target_property_value
            elif property_name == "glow":
                target_property_value = random.uniform(0.0, 1.0)
                target_value_for_log = target_property_value
            elif property_name == "transparency":
                target_property_value = random.uniform(0.0, 1.0)
                target_value_for_log = target_property_value
            elif property_name == "color":
                color_tuple = (random.random(), random.random(), random.random())
                target_property_value = f"({color_tuple[0]:.4f},{color_tuple[1]:.4f},{color_tuple[2]:.4f})"
                target_value_for_log = target_property_value
            elif property_name == "discoloration":
                target_property_value = random.uniform(0.0, 1.0)
                target_value_for_log = target_property_value
            elif property_name == "paint_degradation":
                target_property_value = random.uniform(0.0, 1.0)
                target_value_for_log = target_property_value
            elif property_name == "scratches":
                target_property_value = random.uniform(0.0, 1.0)
                target_value_for_log = target_property_value
            else:
                target_property_value = 0.0
                target_value_for_log = target_property_value

            print(f"[{sample_idx}/{NUM_SAMPLES}] Object: {dir_name} Property: {property_name} Target Value: {target_value_for_log}")

            apply_base_material(main_obj, texture_dir)
            if not render_image(scene, source_path):
                print(f"[Warning] Source render failed for {dir_name} sample {sample_idx}")
                continue

            if property_name == "rust":
                apply_rust(main_obj, texture_dir, target_property_value)
            elif property_name == "metalness":
                apply_metal(main_obj, texture_dir, target_property_value)
            elif property_name == "roughness":
                apply_roughness(main_obj, texture_dir, target_property_value)
            elif property_name == "albedo":
                apply_albedo(main_obj, texture_dir, target_property_value)
            elif property_name == "glow":
                apply_glow(main_obj, texture_dir, target_property_value)
            elif property_name == "transparency":
                apply_transparency(main_obj, target_property_value)
            elif property_name == "color":
                apply_color(main_obj, texture_dir, target_property_value)
            elif property_name == "discoloration":
                apply_discoloration(main_obj, texture_dir, target_property_value)
            elif property_name == "paint_degradation":
                apply_paint_degradation(main_obj, texture_dir, target_property_value)
            elif property_name == "scratches":
                apply_scratches(main_obj, texture_dir, target_property_value)

            if render_image(scene, edit_path):
                mask_rendered = render_mask_image(scene, main_obj, mask_path, property_name, target_property_value)
                if not mask_rendered:
                    print(f"[Warning] Mask render failed for {dir_name} sample {sample_idx}")

                row = [
                    dir_name,
                    os.path.abspath(source_path),
                    os.path.abspath(edit_path),
                    os.path.abspath(mask_path) if mask_rendered else "",
                    property_name,
                    f"{source_property_value:.2f}",
                    str(target_property_value),
                    sample_idx - 1,
                ]
                write_metadata(metadata_path, row)
            else:
                print(f"[Warning] Edit render failed for {dir_name} sample {sample_idx}")

            if light_obj and light_obj.data:
                light_obj.data.energy = original_energy
            if bg_node:
                bg_node.inputs[1].default_value = 1.0

    print("[Done] Dataset generation completed.")


if __name__ == "__main__":
    main()