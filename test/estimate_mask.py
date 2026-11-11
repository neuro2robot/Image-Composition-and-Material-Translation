import os
import argparse
from PIL import Image
from rembg import new_session, remove
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Generate masks for images using rembg")
    parser.add_argument("--path", required=True, help="Root directory containing subfolders with image.png")
    parser.add_argument("--model", required=True, help="Model name to use (e.g., u2net, isnet-general-use, birefnet-massive)")
    args = parser.parse_args()

    root_dir = args.path
    model_name = args.model

    session = new_session(model_name)
    subs = [s for s in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, s))]

    for sub in tqdm(subs, desc="Processing", unit="folder"):
        sub_path = os.path.join(root_dir, sub)
        img_path = os.path.join(sub_path, "image.png")
        mask_path = os.path.join(sub_path, "mask.png")
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path)
                img.load()
                mask = remove(img, session=session, only_mask=True)
                mask.save(mask_path)
            except Exception as e:
                print(f"Skipping {img_path} due to error: {e}")
                continue

if __name__ == "__main__":
    main()
