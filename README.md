# Image Composition and Material Translation

This project has been inspired by [Diffusion Handles](https://github.com/adobe-research/DiffusionHandles) (Adobe Research).

## Data structure for `test/run_multi.sh`

`run_multi.sh` is run from inside `test/` and reads its inputs from `../data/test_new`
(i.e. `data/test_new` at the repository root). You only provide the images; every other file is generated.

```
DiffusionHandles/
├── data/
│   └── test_new/
│       ├── bg/                      # background images
│       │   ├── <bg_name>/
│       │   │   └── image.png        # <- you provide this
│       │   └── ...
│       ├── fg/                      # foreground objects
│       │   ├── <fg_name>/
│       │   │   └── image.png        # <- you provide this
│       │   └── ...
│       └── output/                  # created by the script
│           ├── <bg_name>-<fg_name>/ # one folder per bg/fg pair
│           └── results/             # final edited images
├── test/run_multi.sh
├── ml-depth-pro/                    # clone of apple/ml-depth-pro + run_multi.py from this repo
└── DiffHarmonization/
```

**Folder naming.** The last `_suffix` of a folder name is dropped and the remaining underscores become spaces to
form the label used in the prompt. For example `bg/wooden_table_1` and `fg/red_mug_2` give the prompt
`a red mug on a wooden table`. Always end folder names with a `_<number>` suffix.

**Generated per bg/fg folder:** `prompt.txt`, `depth.png`, `depth.exr`; `mask.png` for fg folders only.

**Generated per combination** (`output/<bg_name>-<fg_name>/`): `fg.png`, `fg_depth.png/.exr`, `fg_mask.png`,
`fg_prompt.txt`, `bg.png`, `bg_depth.png/.exr`, `bg_prompt.txt`, `composite.png`, `composite_refined.png`,
`composite_depth.png/.exr`, `composite_prompt.txt`, `transforms.json`, `sample.json`, `config.yaml`.

## Running

1. Put your images in `data/test_new/bg/<name>/image.png` and `data/test_new/fg/<name>/image.png`.
2. Create the conda environments `diffusionhandles` and `depth-pro`, and set the conda path at the top of `run_multi.sh`.
3. Copy `ml-depth-pro/run_multi.py` from this repo into your clone of ml-depth-pro and download its checkpoints.
4. Run:
   ```
   cd test
   bash run_multi.sh
   ```
