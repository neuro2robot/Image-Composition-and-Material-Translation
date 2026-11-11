# Material Translation

Training-free, graded material editing of a masked real object (rust, metalness, paint degradation, albedo)
by moving its CLIP image embedding along a property direction and generating with a depth-conditioned
SDXL inpainting pipeline through IP-Adapter.

Built on [ZeST](https://github.com/ttchengab/zest_code) and [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter);
inspired by [MARBLE](https://marble-neurips.github.io/) / Alchemist / TexSliders (CLIP-space material control).

## Pipeline

```
build_empirical_direction.py  ->  empirical_property_directions.pt  ->  run_main_code_v2_ipadapter_fixed.py  ->  graded images
```

### 1. Build property directions: `build_empirical_direction.py`
Reads an output tree of graded trajectories (`<root>/<object>/<property>/edited/*.png` with matching `mask/*.png`),
encodes each frame as a masked object on a gray background with the IP-Adapter CLIP encoder, and takes the
first-to-last-frame embedding displacement per object. Directions are unit-normalized, weighted by trajectory
quality (path efficiency x monotonicity x positive progress correlation) and averaged across objects.
Leave-one-object-out cosines decide whether a property direction transfers; inconsistent properties are saved as `None`.
```
python build_empirical_direction.py --output-root outputs/<run> --save-path outputs/<run>/empirical_property_directions.pt
```
No images are generated in this step. A report is written to `<root>/_direction_audit/empirical_direction_report.csv`.

### 2. Generate graded edits: `run_main_code_v2_ipadapter_fixed.py`
For every object and property (albedo, metalness, paint_degradation, rust):
1. masked, gray-background reference crop -> base CLIP embedding `e_base`
2. endpoint = `e_base + TARGET_RELATIVE_ENDPOINT * ||e_base|| * d_p/||d_p||`
3. 10 levels by norm-aware SLERP, `alpha = progress ** ALPHA_EXPONENT`
4. IP-Adapter projection -> depth-ControlNet SDXL inpainting restricted to the mask
5. feathered mask composite, so the background stays unchanged

Main settings (top of the file): `NUM_INFERENCE_STEPS=60`, `INTENSITY_STEPS=10`, `GUIDANCE_SCALE=7.5`,
`CONTROLNET_SCALE=1.20`, IP scale 0.80 -> 2.40 (late schedule), `TRAVERSAL_MODE="slerp"`,
`USE_PROPERTY_TEXT_PROMPT=False` (direction-only), `TARGET_RELATIVE_ENDPOINT` (endpoint distance relative to `||e_base||`).
Set `ONLY_OBJECT` / `ONLY_PROPERTIES` for a quick test; leave both `None` for the full run.
```
python run_main_code_v2_ipadapter_fixed.py
```

## Expected files
```
demo_assets/input_imgs/tools_result_input/<object>.png|jpg          # real object photos
demo_assets/input_imgs/tools_result_input_sam2_masks/<object>.png   # object masks (white = object)
demo_assets/depths/tools_result_input_depth/<object>.png            # depth maps
models/image_encoder/                                               # IP-Adapter CLIP ViT-H image encoder
sdxl_models/ip-adapter_sdxl_vit-h.bin                               # IP-Adapter SDXL checkpoint
outputs/tool_output_updated_july_2026/empirical_property_directions.pt
```
Image encoder and checkpoint come from [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter). Base models
(`SG161222/RealVisXL_V4.0`, `diffusers/controlnet-depth-sdxl-1.0`) download from Hugging Face. Paths are set at the top of the scripts.
Output: `<OUTPUT_FOLDER>/<object>/<property>/edited/<property>_00..09.png` plus masks and per-frame metrics CSV.

## Other scripts
- `dataset_generator.py`, `dataset_generation_july_2026.py` - Blender (bpy) renders of objects with graded property values
  (reference and modified renders). Default paths inside are machine-specific; pass `--object_root_dir`, `--output_root_dir`, `--hdr_path`.
- `audit_properties.py` - leave-one-object-out evaluation of directions on generated folders.
- `masked_clip_direction_audit.py`, `run_masked_clip_audit.py` - masked CLIP trajectory metrics used by the direction builder.
- `ip_adapter/` - IP-Adapter code.

Dependencies: `requirements.txt` (from ZeST).
