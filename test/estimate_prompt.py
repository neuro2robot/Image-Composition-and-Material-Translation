import os
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Create prompt.txt for each folder")
    parser.add_argument("--path", required=True, help="Root directory containing subfolders")
    args = parser.parse_args()

    root_dir = args.path
    subs = [s for s in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, s))]

    for sub in tqdm(subs, desc="Creating prompts", unit="folder"):
        object_name = sub.rsplit("_", 1)[0].replace("_", " ")  # Take text before last underscore
        prompt_text = f"a {object_name}"
        prompt_path = os.path.join(root_dir, sub, "prompt.txt")
        try:
            with open(prompt_path, "w") as f:
                f.write(prompt_text)
        except Exception as e:
            print(f"⚠️ Could not write to {prompt_path}: {e}")

if __name__ == "__main__":
    main()
