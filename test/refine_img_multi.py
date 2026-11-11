import argparse, torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image
from pathlib import Path

def refine_image(pipe, input_image_path, output_image_path, prompt, strength=0.3, guidance_scale=7.5):
    init_image = Image.open(input_image_path).convert("RGB")
    output = pipe(prompt=prompt, image=init_image, strength=strength, guidance_scale=guidance_scale)
    output.images[0].save(output_image_path)
    print(f"[SUCCESS] Saved refined image to {output_image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch refine images using SDXL Refiner")
    parser.add_argument("--root", required=True, help="Root folder containing subfolders with composite.png and composite_prompt.txt")
    parser.add_argument("--strength", type=float, default=0.35)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"No folder found at {root}")

    # Load pipeline once
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to(args.device)

    # Iterate over subfolders
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        img_path = folder / "composite.png"
        prompt_path = folder / "composite_prompt.txt"
        out_path = folder / "composite_refined.png"

        if img_path.exists() and prompt_path.exists():
            try:
                prompt = prompt_path.read_text().strip()
                refine_image(pipe, str(img_path), str(out_path), prompt, args.strength, args.guidance_scale)
            except Exception as e:
                print(f"[ERROR] Failed for {folder.name}: {e}")
        else:
            print(f"[SKIP] Missing composite.png or composite_prompt.txt in {folder}")
