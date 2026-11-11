import os
import argparse
from PIL import Image
import depth_pro
import numpy as np
import torch
import imageio
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, transform = depth_pro.create_model_and_transforms()
model = model.to(device)
model.eval()

def generate_depth(image_path, save_path):
    image, _, f_px = depth_pro.load_rgb(image_path)
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model.infer(image, f_px=f_px)
    depth = prediction["depth"]
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_8bit = (depth_normalized * 255).astype(np.uint8)
    Image.fromarray(depth_8bit).save(save_path)
    imageio.imwrite(save_path.replace(".png", ".exr"), depth.astype(np.float32))

def main():
    parser = argparse.ArgumentParser(description="Generate depth maps for all subfolders")
    parser.add_argument("--path", required=True, help="Root directory containing subfolders with image.png")
    parser.add_argument("--image_name", default="image.png", help="Name of the image file in each subfolder")
    parser.add_argument("--output_name", default="depth.png", help="Name of the output image file in each subfolder")
    args = parser.parse_args()

    root_dir = args.path
    subs = [s for s in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, s))]

    for sub in tqdm(subs, desc="Processing folders", unit="folder"):
        sub_path = os.path.join(root_dir, sub)
        img_path = os.path.join(sub_path, args.image_name)
        depth_path = os.path.join(sub_path, args.output_name)
        if os.path.exists(img_path):
            try:
                generate_depth(img_path, depth_path)
            except Exception as e:
                print(f"⚠️ Skipping {img_path} due to error: {e}")
        else:
            print(f"⚠️ Skipping {img_path} — file not found.")

if __name__ == "__main__":
    main()
