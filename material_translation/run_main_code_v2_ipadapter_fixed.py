
import csv
import gc
import inspect
import os
from collections import Counter

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import ControlNetModel, StableDiffusionXLControlNetInpaintPipeline
from ip_adapter.ip_adapter2 import IPAdapterXL
from PIL import Image, ImageFilter, ImageOps
from tqdm import tqdm


# =============================================================================
# PATHS
# =============================================================================

DEVICE = torch.device("cuda")
DTYPE = torch.float16

INPUT_FOLDER = "demo_assets/input_imgs/tools_result_input"
MASK_FOLDER = "demo_assets/input_imgs/tools_result_input_sam2_masks"
DEPTH_FOLDER = "demo_assets/depths/tools_result_input_depth"
OUTPUT_FOLDER = "outputs/tool_output_empirical_slerp_ipadapter_pushed1_final"
DIRECTIONS_PATH = (
    "outputs/tool_output_updated_july_2026/empirical_property_directions.pt"
)

BASE_MODEL_PATH = "SG161222/RealVisXL_V4.0"
CONTROLNET_PATH = "diffusers/controlnet-depth-sdxl-1.0"
IMAGE_ENCODER_PATH = "models/image_encoder"
IP_CHECKPOINT_PATH = "sdxl_models/ip-adapter_sdxl_vit-h.bin"


# =============================================================================
# CONTROLLED PUSH SETTINGS
# =============================================================================

IMAGE_SIZE = (1024, 1024)
SEED = 42
NUM_INFERENCE_STEPS = 60
INTENSITY_STEPS = 10

# Every property endpoint is exactly this fraction of ||base_embedding|| away.
# This replaces the incomparable global multiplication of raw direction norms.
TARGET_RELATIVE_ENDPOINT = 3.0

GUIDANCE_SCALE =7.5
CONTROLNET_SCALE = 1.20

# The adapter begins gently while layout is established, then grows while the
# material/texture is resolved. If callbacks are unavailable, the fallback is
# used as a constant scale.
USE_LATE_IP_SCALE_SCHEDULE = True
IP_SCALE_LOW = 0.80
IP_SCALE_HIGH = 2.40
IP_SCALE_KNEE = 0.35
IP_SCALE_FALLBACK = 2.00

# alpha = progress ** 0.65 gives the early/middle frames more separation.
ALPHA_EXPONENT = 0.65
TRAVERSAL_MODE = "slerp"  # "slerp" or "linear"

# Keep False for a direction-only causal experiment.
USE_PROPERTY_TEXT_PROMPT = False

# Mask safety. White/opaque object is editable; background is protected.
MASK_THRESHOLD = 128
MAX_EXPECTED_FOREGROUND_COVERAGE = 0.50
MASK_EROSION_PIXELS = 1
INWARD_FEATHER_PIXELS = 3.0

# Tight reference crop with genuine local context.
REFERENCE_PADDING = 0.12
REFERENCE_KEEP_CONTEXT = True

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg")

# Set these for a quick test, then return both to None for the complete run.
ONLY_OBJECT = None
ONLY_PROPERTIES = None
# ONLY_OBJECT = None
# ONLY_PROPERTIES = None
# app.ONLY_OBJECT = "nut_4"
# app.ONLY_PROPERTIES = {"rust", "transparency"}

def property_config(strength, prompt, negative):
    return {
        "max_alpha": 1.0,
        "inpaint_strength": strength,
        "prompt": prompt,
        "negative": negative,
    }


# 0.45 gives 27 effective denoising steps out of 60. Geometry remains anchored
# by depth ControlNet and by the exact final mask composite.
PROPERTY_CONFIGS = {
    "albedo": property_config(
        0.42,
        "changed diffuse base color while preserving exact surface detail",
        "changed geometry, excessive reflection, deep shadows",
    ),
    "metalness": property_config(
        0.44,
        "metallic material, reflections and specular highlights",
        "plastic, fabric, wood, nonmetallic matte material",
    ),
    "paint_degradation": property_config(
        0.45,
        "chipped paint, peeling coating, worn edges and exposed underlayer",
        "perfect new paint, flawless coating, freshly painted",
    ),
    
    "rust": property_config(
        0.55,
        "realistic orange-brown rust, corrosion spots and oxidized metal",
        "clean polished metal, rust-free",
    ),
}

PROPERTY_NAMES = list(PROPERTY_CONFIGS)
METADATA_PATH = os.path.join(OUTPUT_FOLDER, "metadata.csv")


# =============================================================================
# MODEL SETUP -- ATTENTION ORDER MUST NOT CHANGE
# =============================================================================


def build_ip_adapter():
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_PATH,
        variant="fp16",
        use_safetensors=True,
        torch_dtype=DTYPE,
    ).to(DEVICE)

    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        BASE_MODEL_PATH,
        controlnet=controlnet,
        use_safetensors=True,
        torch_dtype=DTYPE,
        add_watermarker=False,
    ).to(DEVICE)

    # Configure standard processors first. IPAdapterXL must install its custom
    # processors last or enable_attention_slicing() will replace all of them.
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()

    ip_model = IPAdapterXL(
        pipe,
        IMAGE_ENCODER_PATH,
        IP_CHECKPOINT_PATH,
        DEVICE,
    )

    pipe.enable_model_cpu_offload()
    ip_model.image_encoder.to(DEVICE, dtype=DTYPE).eval()
    ip_model.image_proj_model.to(DEVICE, dtype=DTYPE).eval()
    return ip_model


def get_ip_processors(ip_model):
    processors = {}
    for name, processor in ip_model.pipe.unet.attn_processors.items():
        class_name = processor.__class__.__name__.lower()
        if "ipattn" in class_name and hasattr(processor, "scale"):
            processors[name] = processor
    return processors


def set_processor_scale(processor, scale):
    scale = float(scale)
    current = processor.scale
    if torch.is_tensor(current):
        current.fill_(scale)
    elif isinstance(current, list):
        processor.scale = [scale] * len(current)
    else:
        processor.scale = scale


def force_ip_scale(ip_model, scale):
    ip_model.set_scale(float(scale))
    processors = get_ip_processors(ip_model)
    if not processors:
        raise RuntimeError("IP-Adapter processors are absent")
    for processor in processors.values():
        set_processor_scale(processor, scale)


def verify_processors(ip_model):
    classes = Counter(
        processor.__class__.__name__
        for processor in ip_model.pipe.unet.attn_processors.values()
    )
    print("\nPROCESSORS AFTER CORRECTED CONSTRUCTION:")
    for class_name, count in classes.items():
        print(f"  {class_name}: {count}")
    processors = get_ip_processors(ip_model)
    print(f"Detected IP processors: {len(processors)}")
    if not processors:
        raise RuntimeError(
            "No IP attention processors. Never call attention slicing after "
            "constructing IPAdapterXL."
        )


def pipeline_supports_step_callback(ip_model):
    try:
        signature = inspect.signature(ip_model.pipe.__call__)
    except (TypeError, ValueError):
        return False
    return "callback_on_step_end" in signature.parameters


def make_ip_scale_callback(ip_model):
    processors = list(get_ip_processors(ip_model).values())
    if not processors:
        raise RuntimeError("Cannot schedule an absent IP-Adapter")

    def callback(pipe, step_index, timestep, callback_kwargs):
        total = int(getattr(pipe, "_num_timesteps", NUM_INFERENCE_STEPS))
        fraction = float(step_index) / max(total - 1, 1)
        if fraction <= IP_SCALE_KNEE:
            scale = IP_SCALE_LOW
        else:
            interpolation = (fraction - IP_SCALE_KNEE) / max(
                1.0 - IP_SCALE_KNEE, 1e-8
            )
            scale = IP_SCALE_LOW + interpolation * (
                IP_SCALE_HIGH - IP_SCALE_LOW
            )
        for processor in processors:
            set_processor_scale(processor, scale)
        return callback_kwargs

    return callback


# =============================================================================
# FILES, DEPTH AND ALPHA-AWARE MASKS
# =============================================================================


def find_image(folder, stem):
    for extension in SUPPORTED_EXTENSIONS:
        path = os.path.join(folder, stem + extension)
        if os.path.isfile(path):
            return path
    return None


def resize_rgb(image):
    return image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.LANCZOS)


def threshold_mask(mask):
    return mask.point(
        lambda value: 255 if value >= MASK_THRESHOLD else 0,
        mode="L",
    )


def load_mask(object_name):
    path = find_image(MASK_FOLDER, object_name)
    if path is None:
        raise FileNotFoundError(f"Mask missing for {object_name}")

    with Image.open(path) as handle:
        image = ImageOps.exif_transpose(handle)
        rgba = image.convert("RGBA")
        alpha = rgba.getchannel("A")
        alpha_values = np.asarray(alpha, dtype=np.uint8)
        alpha_fraction = float((alpha_values >= MASK_THRESHOLD).mean())
        useful_alpha = (
            "A" in image.getbands()
            and int(alpha_values.min()) < int(alpha_values.max())
            and 0.001 < alpha_fraction < 0.999
        )
        if useful_alpha:
            mask = alpha.copy()
            source_kind = "PNG alpha"
        else:
            mask = ImageOps.grayscale(rgba.convert("RGB"))
            source_kind = "luminance"

    mask = mask.resize(IMAGE_SIZE, Image.Resampling.NEAREST)
    mask = threshold_mask(mask)
    initial_coverage = float((np.asarray(mask) > 0).mean())
    inverted = False
    if initial_coverage > MAX_EXPECTED_FOREGROUND_COVERAGE:
        mask = threshold_mask(ImageOps.invert(mask))
        inverted = True

    if MASK_EROSION_PIXELS > 0:
        size = 2 * int(MASK_EROSION_PIXELS) + 1
        mask = threshold_mask(mask.filter(ImageFilter.MinFilter(size)))

    final_coverage = float((np.asarray(mask) > 0).mean())
    print(
        f"{object_name}: mask={source_kind}, initial={initial_coverage:.2%}, "
        f"inverted={inverted}, FINAL EDITABLE={final_coverage:.2%}"
    )
    if final_coverage < 0.001 or final_coverage > MAX_EXPECTED_FOREGROUND_COVERAGE:
        raise RuntimeError(
            f"{object_name}: unsafe mask coverage {final_coverage:.2%}"
        )
    return mask


def load_depth(path):
    with Image.open(path) as handle:
        values = np.asarray(handle)
    if values.ndim == 3:
        values = values[..., 0]
    if values.dtype == np.uint8:
        depth = values
    elif values.dtype == np.uint16:
        depth = np.round(values.astype(np.float32) / 257.0).astype(np.uint8)
    else:
        values = np.nan_to_num(values.astype(np.float32))
        if values.min() >= 0.0 and values.max() <= 1.0:
            values *= 255.0
        depth = np.clip(values, 0, 255).round().astype(np.uint8)
    return (
        Image.fromarray(depth, mode="L")
        .resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        .convert("RGB")
    )


def local_reference(source, mask):
    box = mask.getbbox()
    if box is None:
        raise RuntimeError("Empty object mask")
    x0, y0, x1, y1 = box
    width = x1 - x0
    height = y1 - y0
    side = int(round(max(width, height) * (1.0 + 2.0 * REFERENCE_PADDING)))
    side = max(2, min(side, IMAGE_SIZE[0], IMAGE_SIZE[1]))
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    crop_x0 = int(round(center_x - side / 2.0))
    crop_y0 = int(round(center_y - side / 2.0))
    crop_x0 = max(0, min(crop_x0, IMAGE_SIZE[0] - side))
    crop_y0 = max(0, min(crop_y0, IMAGE_SIZE[1] - side))
    crop_box = (crop_x0, crop_y0, crop_x0 + side, crop_y0 + side)

    if REFERENCE_KEEP_CONTEXT:
        reference_source = source
    else:
        neutral = Image.new("RGB", IMAGE_SIZE, (127, 127, 127))
        reference_source = Image.composite(source, neutral, mask)
    return reference_source.crop(crop_box).resize(
        IMAGE_SIZE, Image.Resampling.LANCZOS
    )


def save_mask_debug(object_folder, source, mask):
    mask.save(os.path.join(object_folder, "_debug_actual_inpainting_mask.png"))
    source_array = np.asarray(source, dtype=np.uint8)
    selected = np.asarray(mask, dtype=np.uint8) > 0
    overlay = source_array.copy()
    overlay[selected] = (
        0.45 * overlay[selected].astype(np.float32)
        + 0.55 * np.array([255.0, 0.0, 0.0])
    ).round().astype(np.uint8)
    Image.fromarray(overlay, mode="RGB").save(
        os.path.join(object_folder, "_debug_mask_overlay.png")
    )


# =============================================================================
# NORMALIZED EMPIRICAL DIRECTIONS AND TRAVERSAL
# =============================================================================


def load_directions():
    if not os.path.isfile(DIRECTIONS_PATH):
        raise FileNotFoundError(DIRECTIONS_PATH)
    if os.path.getsize(DIRECTIONS_PATH) == 0:
        raise RuntimeError(f"Direction file is empty: {DIRECTIONS_PATH}")
    try:
        saved = torch.load(DIRECTIONS_PATH, map_location="cpu", weights_only=True)
    except TypeError:
        saved = torch.load(DIRECTIONS_PATH, map_location="cpu")
    if not isinstance(saved, dict):
        raise TypeError("Direction file must contain a dictionary")

    directions = {}
    for property_name in PROPERTY_NAMES:
        direction = saved.get(property_name)
        if direction is None:
            print(f"{property_name}: missing/rejected")
            directions[property_name] = None
            continue
        direction = direction.detach().float().reshape(1, -1)
        if direction.shape != (1, 1024):
            raise ValueError(
                f"{property_name}: expected (1, 1024), got {direction.shape}"
            )
        if not torch.isfinite(direction).all() or float(direction.norm()) < 1e-8:
            raise ValueError(f"{property_name}: invalid direction")
        directions[property_name] = direction.to(DEVICE)
        print(f"{property_name}: raw empirical norm={float(direction.norm()):.4f}")
    return directions


@torch.inference_mode()
def encode_reference(ip_model, reference):
    pixels = ip_model.clip_image_processor(
        images=[reference], return_tensors="pt"
    ).pixel_values.to(DEVICE, dtype=DTYPE)
    return ip_model.image_encoder(pixels).image_embeds.float()


def angle_deg(first, second):
    cosine = F.cosine_similarity(first.float(), second.float(), dim=-1)
    return float(torch.rad2deg(torch.arccos(cosine.clamp(-1.0, 1.0))).mean())


def norm_aware_slerp(first, second, alpha, threshold=0.9995):
    first = first.float()
    second = second.float()
    alpha = float(alpha)
    first_norm = first.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    second_norm = second.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    first_unit = first / first_norm
    second_unit = second / second_norm
    dot = (first_unit * second_unit).sum(-1, keepdim=True).clamp(-1.0, 1.0)
    if bool((dot < -threshold).any()):
        raise RuntimeError("SLERP endpoints are nearly antipodal")
    if bool((dot > threshold).all()):
        return (1.0 - alpha) * first + alpha * second
    theta = torch.acos(dot)
    denominator = torch.sin(theta).clamp_min(1e-8)
    unit = (
        torch.sin((1.0 - alpha) * theta) / denominator * first_unit
        + torch.sin(alpha * theta) / denominator * second_unit
    )
    radius = (1.0 - alpha) * first_norm + alpha * second_norm
    return unit * radius


def make_traversal(base, raw_direction):
    direction = raw_direction.to(base.device, dtype=base.dtype).reshape(1, -1)
    direction_unit = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    endpoint = (
        base
        + TARGET_RELATIVE_ENDPOINT
        * base.norm(dim=-1, keepdim=True)
        * direction_unit
    )
    endpoint_relative = float(
        (endpoint - base).norm() / base.norm().clamp_min(1e-8)
    )
    endpoint_angle = angle_deg(base, endpoint)

    def traverse(alpha):
        if TRAVERSAL_MODE == "slerp":
            embedding = norm_aware_slerp(base, endpoint, alpha)
        elif TRAVERSAL_MODE == "linear":
            embedding = base + float(alpha) * (endpoint - base)
        else:
            raise ValueError("TRAVERSAL_MODE must be slerp or linear")
        return embedding, angle_deg(base, embedding)

    return traverse, endpoint_angle, endpoint_relative


# =============================================================================
# GENERATION, STRICT COMPOSITE AND METRICS
# =============================================================================


def prompts(config, object_label):
    positive = (
        f"the same exact {object_label} shown in the input image, preserve exact "
        "silhouette, proportions, component positions, camera, pose and geometry"
    )
    negative = (
        "different object, changed geometry, altered silhouette, deformation, "
        "missing parts, extra parts, blurry, distorted, text, watermark"
    )
    if USE_PROPERTY_TEXT_PROMPT:
        positive += ", " + config["prompt"]
        negative += ", " + config["negative"]
    return positive, negative


def final_inward_composite(source, generated, mask):
    source_array = np.asarray(source.convert("RGB"), dtype=np.uint8)
    generated_array = np.asarray(generated.convert("RGB"), dtype=np.uint8)
    mask_array = np.asarray(mask.convert("L"), dtype=np.uint8)
    hard_inside = mask_array >= MASK_THRESHOLD

    if INWARD_FEATHER_PIXELS > 0:
        soft_mask = mask.convert("L").filter(
            ImageFilter.GaussianBlur(radius=float(INWARD_FEATHER_PIXELS))
        )
        weight = np.asarray(soft_mask, dtype=np.float32) / 255.0
        # Critical: discard the outward half of the Gaussian blur.
        weight[~hard_inside] = 0.0
    else:
        weight = hard_inside.astype(np.float32)

    weight = weight[..., None]
    result = np.round(
        source_array.astype(np.float32) * (1.0 - weight)
        + generated_array.astype(np.float32) * weight
    ).clip(0, 255).astype(np.uint8)
    result[~hard_inside] = source_array[~hard_inside]

    outside_difference = np.abs(
        result.astype(np.int16) - source_array.astype(np.int16)
    )[~hard_inside]
    outside_max = int(outside_difference.max()) if outside_difference.size else 0
    if outside_max != 0:
        raise AssertionError(f"Outside-mask maximum is {outside_max}, expected 0")
    return Image.fromarray(result, mode="RGB"), outside_max


@torch.inference_mode()
def generate(ip_model, source, mask, depth, embedding, config, object_label):
    positive, negative = prompts(config, object_label)
    common = dict(
        image=source.convert("RGB"),
        clip_image_embeds=embedding.float(),
        control_image=depth.convert("RGB"),
        mask_image=threshold_mask(mask.convert("L")),
        width=IMAGE_SIZE[0],
        height=IMAGE_SIZE[1],
        controlnet_conditioning_scale=CONTROLNET_SCALE,
        prompt=positive,
        negative_prompt=negative,
        num_samples=1,
        strength=config["inpaint_strength"],
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        seed=SEED,
    )

    scheduled = USE_LATE_IP_SCALE_SCHEDULE and pipeline_supports_step_callback(ip_model)
    if scheduled:
        force_ip_scale(ip_model, IP_SCALE_LOW)
        try:
            result = ip_model.generate(
                **common,
                scale=IP_SCALE_LOW,
                callback_on_step_end=make_ip_scale_callback(ip_model),
                callback_on_step_end_tensor_inputs=["latents"],
            )[0]
            scale_mode = f"scheduled {IP_SCALE_LOW:.2f}->{IP_SCALE_HIGH:.2f}"
        except TypeError as error:
            print(
                "IPAdapterXL wrapper did not forward callback kwargs; "
                f"using constant scale {IP_SCALE_FALLBACK:.2f}. Error: {error}"
            )
            force_ip_scale(ip_model, IP_SCALE_FALLBACK)
            result = ip_model.generate(
                **common,
                scale=IP_SCALE_FALLBACK,
            )[0]
            scale_mode = f"constant fallback {IP_SCALE_FALLBACK:.2f}"
    else:
        force_ip_scale(ip_model, IP_SCALE_FALLBACK)
        result = ip_model.generate(
            **common,
            scale=IP_SCALE_FALLBACK,
        )[0]
        scale_mode = f"constant {IP_SCALE_FALLBACK:.2f}"

    generated = result.convert("RGB").resize(IMAGE_SIZE)
    final, outside_max = final_inward_composite(source, generated, mask)
    return final, scale_mode, outside_max


def difference_stats(first_array, second_array, mask_array):
    difference = np.abs(
        second_array.astype(np.float32) - first_array.astype(np.float32)
    )
    global_mean = float(difference.mean())
    selected = mask_array > 0
    mask_mean = float(difference[selected].mean())
    changed = float((difference.max(axis=2)[selected] > 2.0).mean() * 100.0)
    return global_mean, mask_mean, changed


def append_metadata(row):
    exists = os.path.isfile(METADATA_PATH)
    with open(METADATA_PATH, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(
                [
                    "object", "property", "frame", "linear_progress", "alpha",
                    "angle_deg", "embedding_delta", "token_delta",
                    "mask_from_source", "mask_from_previous", "mask_from_first",
                    "mask_changed_from_first_pct", "outside_mask_max_delta",
                    "ip_scale_mode", "image_path", "mask_path",
                ]
            )
        writer.writerow(row)


def process_object(filename, ip_model, directions):
    object_name = os.path.splitext(filename)[0]
    if ONLY_OBJECT is not None and object_name != ONLY_OBJECT:
        return
    depth_path = os.path.join(DEPTH_FOLDER, object_name + ".png")
    if not os.path.isfile(depth_path):
        print(f"Skipping {object_name}: depth missing")
        return

    with Image.open(os.path.join(INPUT_FOLDER, filename)) as handle:
        source = resize_rgb(handle)
    mask = load_mask(object_name)
    depth = load_depth(depth_path)
    reference = local_reference(source, mask)
    source_array = np.asarray(source, dtype=np.uint8)
    mask_array = np.asarray(mask, dtype=np.uint8)

    object_folder = os.path.join(OUTPUT_FOLDER, object_name)
    os.makedirs(object_folder, exist_ok=True)
    source.save(os.path.join(object_folder, "base_canvas.png"))
    reference.save(os.path.join(object_folder, "base_reference.png"))
    save_mask_debug(object_folder, source, mask)

    base_embedding = encode_reference(ip_model, reference)
    base_tokens, _ = ip_model.get_image_embeds(
        pil_image=None, clip_image_embeds=base_embedding.float()
    )
    base_norm = max(float(base_embedding.norm()), 1e-8)
    token_denominator = max(float(base_tokens.abs().mean()), 1e-8)
    object_label = object_name.replace("_", " ")

    for property_name, config in PROPERTY_CONFIGS.items():
        if ONLY_PROPERTIES is not None and property_name not in ONLY_PROPERTIES:
            continue
        direction = directions.get(property_name)
        if direction is None:
            continue

        traverse, endpoint_angle, endpoint_relative = make_traversal(
            base_embedding, direction
        )
        print(
            f"\n{object_name}/{property_name}: endpoint angle={endpoint_angle:.3f}, "
            f"normalized endpoint={endpoint_relative:.2%}, "
            f"strength={config['inpaint_strength']}, guidance={GUIDANCE_SCALE}"
        )

        edited_folder = os.path.join(object_folder, property_name, "edited")
        mask_folder = os.path.join(object_folder, property_name, "mask")
        os.makedirs(edited_folder, exist_ok=True)
        os.makedirs(mask_folder, exist_ok=True)
        first_array = None
        previous_array = None

        for frame in range(INTENSITY_STEPS):
            linear_progress = frame / (INTENSITY_STEPS - 1)
            alpha = linear_progress ** ALPHA_EXPONENT
            alpha *= config["max_alpha"]
            embedding, angle = traverse(alpha)
            image_tokens, _ = ip_model.get_image_embeds(
                pil_image=None, clip_image_embeds=embedding.float()
            )
            embedding_delta = float((embedding - base_embedding).norm()) / base_norm
            token_delta = float((image_tokens - base_tokens).abs().mean()) / token_denominator

            final, scale_mode, outside_max = generate(
                ip_model, source, mask, depth, embedding, config, object_label
            )
            final_array = np.asarray(final, dtype=np.uint8)
            _, mask_source, _ = difference_stats(
                source_array, final_array, mask_array
            )
            if previous_array is None:
                mask_previous = 0.0
            else:
                _, mask_previous, _ = difference_stats(
                    previous_array, final_array, mask_array
                )
            if first_array is None:
                mask_first = 0.0
                changed_first = 0.0
                first_array = final_array.copy()
            else:
                _, mask_first, changed_first = difference_stats(
                    first_array, final_array, mask_array
                )
            previous_array = final_array.copy()

            image_path = os.path.join(
                edited_folder, f"{property_name}_{frame:02d}.png"
            )
            mask_path = os.path.join(
                mask_folder, f"{property_name}_{frame:02d}_mask.png"
            )
            final.save(image_path)
            mask.save(mask_path)

            print(
                f"{object_name}/{property_name}/idx={frame}: "
                f"progress={linear_progress:.3f}, alpha={alpha:.3f}, "
                f"angle={angle:.3f}, embedding={embedding_delta:.2%}, "
                f"tokens={token_delta:.2%}, MASK_source={mask_source:.3f}/255, "
                f"MASK_previous={mask_previous:.3f}/255, "
                f"MASK_first={mask_first:.3f}/255, changed={changed_first:.1f}%, "
                f"OUTSIDE_MAX={outside_max}/255, IP={scale_mode}"
            )

            append_metadata(
                [
                    object_name, property_name, frame,
                    f"{linear_progress:.4f}", f"{alpha:.4f}", f"{angle:.4f}",
                    f"{embedding_delta:.8f}", f"{token_delta:.8f}",
                    f"{mask_source:.8f}", f"{mask_previous:.8f}",
                    f"{mask_first:.8f}", f"{changed_first:.4f}", outside_max,
                    scale_mode, os.path.abspath(image_path), os.path.abspath(mask_path),
                ]
            )

            del final, final_array, embedding, image_tokens
            gc.collect()
            torch.cuda.empty_cache()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This fp16 SDXL pipeline requires a CUDA GPU")
    if INTENSITY_STEPS < 2:
        raise ValueError("INTENSITY_STEPS must be at least 2")
    if not 0.0 < TARGET_RELATIVE_ENDPOINT:
        raise ValueError("TARGET_RELATIVE_ENDPOINT must be positive")
    if not 0.0 < ALPHA_EXPONENT:
        raise ValueError("ALPHA_EXPONENT must be positive")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    if os.path.exists(METADATA_PATH):
        os.remove(METADATA_PATH)

    directions = load_directions()
    ip_model = build_ip_adapter()
    verify_processors(ip_model)
    force_ip_scale(ip_model, IP_SCALE_LOW)
    print(
        f"Step callback supported by pipeline: "
        f"{pipeline_supports_step_callback(ip_model)}"
    )

    files = sorted(
        filename for filename in os.listdir(INPUT_FOLDER)
        if filename.lower().endswith(SUPPORTED_EXTENSIONS)
    )
    for filename in tqdm(files, desc="Objects"):
        try:
            process_object(filename, ip_model, directions)
        except Exception as error:
            print(f"FAILED {filename}: {type(error).__name__}: {error}")
        gc.collect()
        torch.cuda.empty_cache()

    ip_model.image_encoder.to("cpu")
    ip_model.image_proj_model.to("cpu")
    ip_model.pipe.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\nFinished. Results: {OUTPUT_FOLDER}")
    print(f"Metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()
