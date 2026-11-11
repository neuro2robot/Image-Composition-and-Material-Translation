import argparse, torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image
from pathlib import Path

def refine_image(input_image_path, output_image_path, prompt, strength=0.3, guidance_scale=7.5, device="cuda"):
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to(device)
    init_image = Image.open(input_image_path).convert("RGB")
    output = pipe(prompt=prompt, image=init_image, strength=strength, guidance_scale=guidance_scale)
    output.images[0].save(output_image_path)
    print(f"[SUCCESS] Saved refined image to {output_image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refine an image using SDXL Refiner")
    parser.add_argument("--img_path", required=True, help="Path to input image")
    parser.add_argument("--output_path", required=True, help="Path to save refined image")
    parser.add_argument("--prompt_path", required=True, help="Path to prompt text file")
    parser.add_argument("--strength", type=float, default=0.35)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    img_path = Path(args.img_path)
    if not img_path.exists():
        raise FileNotFoundError(f"No file found at {img_path}")
    prompt = Path(args.prompt_path).read_text().strip()
    refine_image(str(img_path), args.output_path, prompt, args.strength, args.guidance_scale, args.device)
