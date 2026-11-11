#!/bin/bash
set -e
export PATH="/home/docker/miniconda3/bin:$PATH"

print_info() { echo -e "\033[1;34m[INFO]\033[0m $1"; }
print_success() { echo -e "\033[1;32m[SUCCESS]\033[0m $1"; }
print_error() { echo -e "\033[1;31m[ERROR]\033[0m $1"; }
execute() {
    eval "$1"
    if [ $? -ne 0 ]; then print_error "Command failed: $1"; return 1; else print_success "Command succeeded: $1"; return 0; fi
}

print_info "Initializing Conda..."
execute 'eval "$(conda shell.bash hook)"'
execute 'conda activate base'

ROOT_PATH="../data/test_new"
OUTPUT_PATH="../data/test_new/output"
BG_PATH="$ROOT_PATH/bg"
FG_PATH="$ROOT_PATH/fg"
SAMPLE_JSON="$ROOT_PATH/sample.json"

print_info "Generating prompts for BG folders"
execute "python estimate_prompt.py --path \"$BG_PATH\""
print_info "Generating prompts for FG folders"
execute "python estimate_prompt.py --path \"$FG_PATH\""

print_info "Generating masks for FG folders"
execute "python estimate_mask.py --path \"$FG_PATH\" --model birefnet-general"

cd ../ml-depth-pro || exit
print_info "Activating Conda environment: depth-pro"
execute 'conda activate depth-pro'

print_info "Generating depth for BG folders"
execute "python run_multi.py --path \"$BG_PATH\" --image_name image.png --output_name depth.png"
print_info "Generating depth for FG folders"
execute "python run_multi.py --path \"$FG_PATH\" --image_name image.png --output_name depth.png"

execute 'conda activate diffusionhandles'
cd ../test || exit

mkdir -p "$OUTPUT_PATH"

# # --- PHASE 1: CREATE ALL COMBINATIONS ---
for bg_folder in "$BG_PATH"/*; do
    bg_name=$(basename "$bg_folder")
    if [ ! -d "$bg_folder" ]; then continue; fi
    bg_label="${bg_name%_*}"
    bg_label="${bg_label//_/ }"

    for fg_folder in "$FG_PATH"/*; do
        fg_name=$(basename "$fg_folder")
        if [ ! -d "$fg_folder" ]; then continue; fi
        fg_label="${fg_name%_*}"
        fg_label="${fg_label//_/ }"

        combo_folder="$OUTPUT_PATH/${bg_name}-${fg_name}"
        mkdir -p "$combo_folder"

        {
            [ -f "$fg_folder/image.png" ] && cp "$fg_folder/image.png" "$combo_folder/fg.png"
            [ -f "$fg_folder/depth.png" ] && cp "$fg_folder/depth.png" "$combo_folder/fg_depth.png"
            [ -f "$fg_folder/depth.exr" ] && cp "$fg_folder/depth.exr" "$combo_folder/fg_depth.exr"
            [ -f "$fg_folder/depth.npy" ] && cp "$fg_folder/depth.npy" "$combo_folder/fg_depth_intrinsics.npy"
            [ -f "$fg_folder/prompt.txt" ] && cp "$fg_folder/prompt.txt" "$combo_folder/fg_prompt.txt"
            [ -f "$fg_folder/prompt.txt" ] && cp "$fg_folder/prompt.txt" "$combo_folder/fg_harmonize_prompt.txt"
            [ -f "$fg_folder/mask.png" ] && cp "$fg_folder/mask.png" "$combo_folder/fg_mask.png"
            [ -f "$fg_folder/mask.png" ] && cp "$fg_folder/mask.png" "$combo_folder/composite_mask.png"

            [ -f "$bg_folder/image.png" ] && cp "$bg_folder/image.png" "$combo_folder/bg.png"
            [ -f "$bg_folder/depth.png" ] && cp "$bg_folder/depth.png" "$combo_folder/bg_depth.png"
            [ -f "$bg_folder/depth.exr" ] && cp "$bg_folder/depth.exr" "$combo_folder/bg_depth.exr"
            [ -f "$bg_folder/depth.npy" ] && cp "$bg_folder/depth.npy" "$combo_folder/bg_depth_intrinsics.npy"
            [ -f "$bg_folder/prompt.txt" ] && cp "$bg_folder/prompt.txt" "$combo_folder/bg_prompt.txt"
            [ -f "$bg_folder/prompt.txt" ] && cp "$bg_folder/prompt.txt" "$combo_folder/bg_harmonize_prompt.txt"

            echo "a ${fg_label} on a ${bg_label}" > "$combo_folder/composite_prompt.txt"

            cat > "$combo_folder/transforms.json" << EOL
{
    "edit_000": {
        "translation": [0.0, 0.0, 0.0],
        "rotation_axis": [0.0, 1.0, 0.0],
        "rotation_angle": 0.0
    }
}
EOL

            cat > "$combo_folder/sample.json" << EOL
{
    "${bg_name}-${fg_name}": ["edit_000"]
}
EOL

            cat > "$combo_folder/config.yaml" << EOL
guided_diffuser:
  bg_weight: 0.1
  fg_weight: 0.15 # 0.15 worked well for pin
  activation_loss_weight: 1.0
  bg_latent_loss_weight: 1.0
  fg_latent_loss_weight: 5.0 # 2.0Worked well for pin
  fg_patch_size: 1
  bg_patch_size: 1
  use_depth: true
  save_denoising_steps: false
  bg_loss_type: 'global_avg'
  num_timesteps: 50
  num_optsteps: 3
  guidance_max_step: 38
  guidance_schedule_type: 'constant'
  bg_erosion: 0
  seed: 2773
depth_transform_mode: 'pc'
EOL

            convert "$combo_folder/fg.png" "$combo_folder/bg.png" -mask "$combo_folder/fg_mask.png" -composite "$combo_folder/composite.png"

            # execute "python refine_img.py --img_path \"$combo_folder/composite.png\" --output_path \"$combo_folder/composite_refined.png\" --prompt_path \"$combo_folder/composite_prompt.txt\" --strength 0.1 --guidance_scale 7.5 --device cuda > /dev/null 2>&1"

            print_success "Created combination: ${bg_name}-${fg_name}"
        } || { print_error "Failed combination: ${bg_name}-${fg_name}, skipping..."; continue; }
    done
done

execute "python refine_img_multi.py --root \"$OUTPUT_PATH\" --strength 0.05 --guidance_scale 5.0 --device cuda"

print_info "Generating depth for composite images..."
cd ../ml-depth-pro || exit
execute 'conda activate depth-pro'
execute "python run_multi.py --path \"$OUTPUT_PATH\" --image_name composite_refined.png --output_name composite_depth.png"
execute 'conda activate diffusionhandles'
cd ../test || exit

# --- PHASE 2: RUN TEST ON ALL COMBINATIONS ---
print_info "Running test_diffusion_handles.py for all combinations..."
for combo_folder in "$OUTPUT_PATH"/*; do
    [ -d "$combo_folder" ] || continue
    [ "$(basename "$combo_folder")" = "results" ] && continue
    { 
        execute "python test_diffusion_handles.py --mode custom --config_path \"$combo_folder/config.yaml\" --test_set_path \"$combo_folder/sample.json\" --input_dir \"$OUTPUT_PATH\" --output_dir \"$OUTPUT_PATH/results\" --device cuda" # --cache_input_image_identity"  


    } || { 
        print_error "test_diffusion_handles.py failed for $combo_folder, skipping..."; continue; }
done
