from os.path import join, exists, basename, splitext, dirname
from os import makedirs
import json
from collections import OrderedDict
import argparse
import tempfile

import torch
import numpy as np
from omegaconf import OmegaConf
#from diffhandles import DiffusionHandles
#from diffhandles.diffusion_handles import DiffusionHandles

import sys
import os

# Add parent directory of `test` (which is `DiffusionHandles/`) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from diffhandles.diffusion_handles import DiffusionHandles
# from diffhandles.depth_transform_utils import depth_transform_pipeline
from DiffHarmonization.attentionControlV2 import AttentionReplace

from remove_foreground import remove_foreground
from estimate_depth import estimate_depth
from generate_results_webpage import generate_results_webpage
from utils import crop_and_resize, load_image, load_depth, save_image
from estimate_composite_depth import get_composite_depth_3D

import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 3 = Suppress all messages, including errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore", category=FutureWarning)

def test_diffusion_handles(
        test_set_path:str, input_dir:str, output_dir:str,
        skip_existing: bool = False, cache_input_image_identity: bool = False,
        config_path: str = None, device: str = None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    img_res = 512

    # # TEMP!
    # # need to:
    # # export CUBLAS_WORKSPACE_CONFIG=:4096:8
    # torch.use_deterministic_algorithms(True)
    # torch.manual_seed(76523524)
    # import random
    # random.seed(634566435)
    # import numpy as np
    # np.random.seed(54396745)
    # # TEMP!

    # load the test set info
    with open(test_set_path, 'r') as f:
        dataset_names = json.load(f, object_pairs_hook=OrderedDict)

    preprocess_samples(input_dir=input_dir, dataset_names=dataset_names)

    diff_handles_config = OmegaConf.load(config_path) if config_path is not None else None
    diff_handles = DiffusionHandles(conf=diff_handles_config)
    diff_handles.to(device)

    # save config to output directory
    makedirs(output_dir, exist_ok=True)
    with open(join(output_dir, 'config.yaml'), 'w') as f:
        OmegaConf.save(config=diff_handles.conf, f=f)

    # generate a webpage for the results
    generate_results_webpage(
        test_set_path=args.test_set_path,
        website_path=join(args.output_dir, f'{splitext(basename(args.test_set_path))[0]}_summary.html'),
        relative_image_dir='.',
        show_denoising_steps=diff_handles.diffuser.conf.save_denoising_steps,
        num_timesteps=diff_handles.diffuser.conf.num_timesteps,
        num_optsteps=diff_handles.diffuser.conf.num_optsteps)

    # iterate over test set samples
    print(f"Editing {len(dataset_names)} images ...")
    for sample_idx, (sample_name, transform_names) in enumerate(dataset_names.items()):

        # load inputs for the sample
        transforms, prompt, img, fg_mask, depth, bg_depth, transform_exists = load_diffhandles_inputs(
            input_dir=input_dir, sample_name=sample_name, output_dir=output_dir, img_res=img_res, device=device, skip_existing=skip_existing)

        if transforms is None:
            continue

        print(f"[{sample_idx+1}/{len(dataset_names)}] Editing image {sample_name} with {len(transforms)} transforms ...")

        # save inputs for visualization to results directory
        save_diffhandles_inputs(
            transforms=transforms, prompt=prompt, img=img, fg_mask=fg_mask, depth=depth, bg_depth=bg_depth,
            input_dir=input_dir, sample_name=sample_name, output_dir=output_dir, img_res=img_res)

        # compute or load input image identity
        input_image_identity_path = join(tempfile.gettempdir(), 'diffhandles', splitext(basename(test_set_path))[0], sample_name, 'input_image_identity.npz')
        if cache_input_image_identity and exists(input_image_identity_path):
            # load input image identity
            input_image_identity = np.load(input_image_identity_path)
            null_text_emb = torch.from_numpy(input_image_identity['null_text_emb']).to(device=device)
            init_noise = torch.from_numpy(input_image_identity['init_noise']).to(device=device)
            activations = [torch.from_numpy(input_image_identity[f'activations{i+1}']).to(device=device) for i in range(3)]
            latent_image = torch.from_numpy(input_image_identity['latent_image']).to(device=device)
        else:
            # invert the input image to get initial noise and null text
            # that can be used to reconstruct the input image with the diffusion model
            null_text_emb, init_noise = diff_handles.invert_input_image(img, depth, prompt)

            # reconstruct the input image from the inverted null text and noise,
            # and save intermediate activations that will be used as guidance for the image edit
            null_text_emb, init_noise, activations, latent_image = diff_handles.generate_input_image(
                depth=depth, prompt=prompt, null_text_emb=null_text_emb, init_noise=init_noise)

            # The activations are intermediate decoder feature maps extracted from the diffusion U-Net 
            # during the denoising process. Specifically, they are the output hidden-state tensors (sample) 
            # produced after each of the three CrossAttnUpBlock2D modules in the U-Net's upsampling (decoder) 
            # path, before the final normalization and output convolution that predicts the noise residual. 
            # As the noisy latent is progressively refined at each diffusion timestep, these feature maps encode 
            # multi-scale information that combines the current latent representation, timestep embedding, text 
            # prompt conditioning, and depth conditioning. The first activation captures relatively coarse 
            # semantic and structural information, while the subsequent activations represent increasingly finer 
            # spatial details and object appearance as the decoder reconstructs the latent. During inversion, 
            # these activations are saved at every diffusion timestep, providing a trajectory of internal U-Net 
            # representations of the source image that can later be used during editing to preserve the original 
            # image's identity, geometry, and appearance by matching or constraining the corresponding decoder features.

            # Unlike latents, which are compact noisy representations of the image before being processed by the U-Net, 
            # activations are rich intermediate feature maps generated inside the U-Net after convolution, residual, 
            # skip-connection, and cross-attention operations. They encode high-level semantic information such as object 
            # structure, geometry, textures, and prompt-conditioned appearance at multiple scales, making them much more informative 
            # for preserving image identity during editing. Consequently, matching activations provides stronger semantic and 
            # perceptual consistency than matching latents alone, while still allowing the latent representation to change to accommodate 
            # the desired edit.

            if cache_input_image_identity:
                # save input image identity
                makedirs(dirname(input_image_identity_path), exist_ok=True)
                input_image_identity = {
                    'null_text_emb': null_text_emb.detach().cpu().numpy(),
                    'init_noise': init_noise.detach().cpu().numpy(),
                    'activations1': activations[0].detach().cpu().numpy(),
                    'activations2': activations[1].detach().cpu().numpy(),
                    'activations3': activations[2].detach().cpu().numpy(),
                    'latent_image': latent_image.detach().cpu().numpy()
                }
                np.savez(input_image_identity_path, **input_image_identity)

        # set the foreground object to get an updated background depth
        # bg_depth = diff_handles.set_foreground(depth=depth, fg_mask=fg_mask, bg_depth=bg_depth)
        # save_image((bg_depth/bg_depth.max())[0], join(output_dir, sample_name, 'bg_depth_2.png'))

        # save image reconstructed from inversion
        with torch.no_grad():
            latent_image = 1 / 0.18215 * latent_image.detach()
            recon_image = diff_handles.diffuser.vae.decode(latent_image)['sample']
            recon_image = (recon_image + 1) / 2
        save_image(recon_image.clamp(min=0, max=1)[0], join(output_dir, sample_name, 'recon.png'))

        # iterate over the transformations
        for transform_name in transform_names:

            if transform_name not in transforms:
                print(f'WARNING: Transform {transform_name} not found for image {sample_name}. Skipping.')
                continue

            if skip_existing and transform_exists[transform_name]:
                print(f'Skipping transform {transform_name} of image {sample_name}, since its output already exists.')
                continue

            transform = transforms[transform_name]

            # get transformation parameters
            translation = torch.tensor(transform['translation'], dtype=torch.float32) if 'translation' in transform else None
            rot_axis = torch.tensor(transform['rotation_axis'], dtype=torch.float32) if 'rotation_axis' in transform else None
            rot_angle = float(transform['rotation_angle']) if 'rotation_angle' in transform else None

            # transform the foreground object
            results = diff_handles.transform_foreground(
                depth=depth, prompt=prompt,
                fg_mask=fg_mask, bg_depth=bg_depth,
                null_text_emb=null_text_emb, init_noise=init_noise,
                activations=activations,
                rot_angle=rot_angle, rot_axis=rot_axis, translation=translation,
                use_input_depth_normalization=False)

            if diff_handles.conf.guided_diffuser.save_denoising_steps:
                edited_img, edited_disparity, denoising_steps = results
            else:
                edited_img, edited_disparity = results
                denoising_steps = None

            save_diffhandles_outputs(
                edited_img=edited_img, edited_disparity=edited_disparity, denoising_steps=denoising_steps,
                output_dir=output_dir, sample_name=sample_name, transform_name=transform_name)

    # save sample names to result directory
    with open(join(output_dir, basename(test_set_path)), 'w') as f:
        json.dump(dataset_names, f, indent=4)

def test_diffusion_handles_composite(
        test_set_path:str, input_dir:str, output_dir:str,
        skip_existing: bool = False, cache_input_image_identity: bool = False,
        config_path: str = None, device: str = None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    img_res = 512

    # load the test set info
    with open(test_set_path, 'r') as f:
        dataset_names = json.load(f, object_pairs_hook=OrderedDict)

    diff_handles_config = OmegaConf.load(config_path) if config_path is not None else None
    diff_handles = DiffusionHandles(conf=diff_handles_config)
    diff_handles.to(device)

    # save config to output directory
    makedirs(output_dir, exist_ok=True)
    with open(join(output_dir, 'config.yaml'), 'w') as f:
        OmegaConf.save(config=diff_handles.conf, f=f)

    # generate a webpage for the results
    generate_results_webpage(
        test_set_path=args.test_set_path,
        website_path=join(args.output_dir, f'{splitext(basename(args.test_set_path))[0]}_summary.html'),
        relative_image_dir='.',
        show_denoising_steps=diff_handles.diffuser.conf.save_denoising_steps,
        num_timesteps=diff_handles.diffuser.conf.num_timesteps,
        num_optsteps=diff_handles.diffuser.conf.num_optsteps)

    # iterate over test set samples
    print(f"Editing {len(dataset_names)} images ...")
    for sample_idx, (sample_name, transform_names) in enumerate(dataset_names.items()):

        # load inputs for the sample
        transforms, fg_prompt, bg_prompt, composite_prompt, harmonize_prompts, fg_img, bg_img, fg_mask, composite_mask, fg_depth, bg_depth, composite_depth, composite_inverted_depth, transform_exists = load_diffhandles_inputs_composite(
            input_dir=input_dir, sample_name=sample_name, output_dir=output_dir, img_res=img_res, device=device, skip_existing=skip_existing)

        if transforms is None:
            continue

        print(f"[{sample_idx+1}/{len(dataset_names)}] Editing image {sample_name} with {len(transforms)} transforms ...")

        # save inputs for visualization to results directory
        save_diffhandles_inputs_composite(
            fg_prompt, bg_prompt, composite_prompt, harmonize_prompts,
            fg_img, bg_img, fg_mask, composite_mask,
            fg_depth, bg_depth, composite_depth, 
            transforms, sample_name, output_dir
        )


        # The activations are intermediate decoder feature maps extracted from the diffusion U-Net 
        # during the denoising process. Specifically, they are the output hidden-state tensors (sample) 
        # produced after each of the three CrossAttnUpBlock2D modules in the U-Net's upsampling (decoder) 
        # path, before the final normalization and output convolution that predicts the noise residual. 
        # As the noisy latent is progressively refined at each diffusion timestep, these feature maps encode 
        # multi-scale information that combines the current latent representation, timestep embedding, text 
        # prompt conditioning, and depth conditioning. The first activation captures relatively coarse 
        # semantic and structural information, while the subsequent activations represent increasingly finer 
        # spatial details and object appearance as the decoder reconstructs the latent. During inversion, 
        # these activations are saved at every diffusion timestep, providing a trajectory of internal U-Net 
        # representations of the source image that can later be used during editing to preserve the original 
        # image's identity, geometry, and appearance by matching or constraining the corresponding decoder features.

        # Unlike latents, which are compact noisy representations of the image before being processed by the U-Net, 
        # activations are rich intermediate feature maps generated inside the U-Net after convolution, residual, 
        # skip-connection, and cross-attention operations. They encode high-level semantic information such as object 
        # structure, geometry, textures, and prompt-conditioned appearance at multiple scales, making them much more informative 
        # for preserving image identity during editing. Consequently, matching activations provides stronger semantic and 
        # perceptual consistency than matching latents alone, while still allowing the latent representation to change to accommodate 
        # the desired edit.

        # The latent-based losses directly constrain the current diffusion latent to match the foreground latent 
        # inside the transformed object mask and the background latent outside it, so they mainly preserve the 
        # low-level image state, spatial composition, and background consistency. In contrast, the activation-based 
        # losses compare U-Net feature vectors at corresponding original and transformed locations—using local patch 
        # averages for the foreground and local or global averages for the background—so they preserve higher-level 
        # semantic properties such as object identity, shape, texture, and contextual appearance even when the object 
        # moves. Thus, latent loss keeps the generated image numerically close to the desired foreground/background 
        # latent composition, while activation loss keeps the edited result perceptually and semantically consistent 
        # with the original image.


        # compute or load fg image identity
        fg_image_identity_path = join(tempfile.gettempdir(), 'diffhandles', splitext(basename(test_set_path))[0], sample_name, 'input_fg_image_identity.npz')
        
        condition_enter=False
        if cache_input_image_identity and exists(fg_image_identity_path):
            fg_image_identity = np.load(fg_image_identity_path)
            fg_null_text_emb = torch.from_numpy(fg_image_identity['null_text_emb']).to(device=device)
            fg_init_noise = torch.from_numpy(fg_image_identity['init_noise']).to(device=device)
            fg_activations = [torch.from_numpy(fg_image_identity[f'activations{i+1}']).to(device=device) for i in range(3)]
            fg_latent_image = torch.from_numpy(fg_image_identity['latent_image']).to(device=device)
            latents_list_len = int(fg_image_identity['latents_list_len'])  # you must have saved this earlier
            fg_latents_list = [
                torch.from_numpy(fg_image_identity[f'latents_list_{i}']).to(device=device)
                for i in range(latents_list_len)
            ]
        
            print("fg_init_noise shape:", fg_init_noise.shape)
            print("fg_latents_list shape:", fg_latents_list[0].shape)
            print("fg_null_text_emb shape:", fg_null_text_emb.shape)

        else:
            # invert the input image to get initial noise and null text
            # that can be used to reconstruct the input image with the diffusion model
            fg_null_text_emb, fg_init_noise = diff_handles.invert_input_image(fg_img, fg_depth, fg_prompt)

            # reconstruct the input image from the inverted null text and noise,
            # and save intermediate activations that will be used as guidance for the image edit
            fg_null_text_emb, fg_init_noise, fg_activations, fg_latent_image, fg_latents_list = diff_handles.generate_input_image(
                depth=fg_depth, prompt=fg_prompt, null_text_emb=fg_null_text_emb, init_noise=fg_init_noise)
            # print("Shape of foreground activations",len(fg_activations))
            if cache_input_image_identity:
                # save input image identity
                makedirs(dirname(fg_image_identity_path), exist_ok=True)
                input_image_identity = {
                    'null_text_emb': fg_null_text_emb.detach().cpu().numpy(),
                    'init_noise': fg_init_noise.detach().cpu().numpy(),
                    'activations1': fg_activations[0].detach().cpu().numpy(),
                    'activations2': fg_activations[1].detach().cpu().numpy(),
                    'activations3': fg_activations[2].detach().cpu().numpy(),
                    'latent_image': fg_latent_image.detach().cpu().numpy(),
                }
                for i, latent in enumerate(fg_latents_list):
                    input_image_identity[f'latents_list_{i}'] = latent.detach().cpu().numpy()
                input_image_identity['latents_list_len'] = len(fg_latents_list)

                np.savez(fg_image_identity_path, **input_image_identity)

        # compute or load bg image identity
        bg_image_identity_path = join(tempfile.gettempdir(), 'diffhandles', splitext(basename(test_set_path))[0], sample_name, 'input_bg_image_identity.npz')
        if cache_input_image_identity and exists(bg_image_identity_path):
            bg_image_identity = np.load(bg_image_identity_path)
            bg_null_text_emb = torch.from_numpy(bg_image_identity['null_text_emb']).to(device=device)
            bg_init_noise = torch.from_numpy(bg_image_identity['init_noise']).to(device=device)
            bg_activations = [torch.from_numpy(bg_image_identity[f'activations{i+1}']).to(device=device) for i in range(3)]
            bg_latent_image = torch.from_numpy(bg_image_identity['latent_image']).to(device=device)
            latents_list_len = int(bg_image_identity['latents_list_len'])  # you must have saved this earlier
            bg_latents_list = [
                torch.from_numpy(bg_image_identity[f'latents_list_{i}']).to(device=device)
                for i in range(latents_list_len)
            ]
        else:
            # invert the input image to get initial noise and null text
            # that can be used to reconstruct the input image with the diffusion model
            bg_null_text_emb, bg_init_noise = diff_handles.invert_input_image(bg_img, bg_depth, bg_prompt)

            # reconstruct the input image from the inverted null text and noise,
            # and save intermediate activations that will be used as guidance for the image edit
            bg_null_text_emb, bg_init_noise, bg_activations, bg_latent_image, bg_latents_list = diff_handles.generate_input_image(
                depth=bg_depth, prompt=bg_prompt, null_text_emb=bg_null_text_emb, init_noise=bg_init_noise)

            if cache_input_image_identity:
                # save input image identity
                makedirs(dirname(bg_image_identity_path), exist_ok=True)
                input_image_identity = {
                    'null_text_emb': bg_null_text_emb.detach().cpu().numpy(),
                    'init_noise': bg_init_noise.detach().cpu().numpy(),
                    'activations1': bg_activations[0].detach().cpu().numpy(),
                    'activations2': bg_activations[1].detach().cpu().numpy(),
                    'activations3': bg_activations[2].detach().cpu().numpy(),
                    'latent_image': bg_latent_image.detach().cpu().numpy()
                }
                for i, latent in enumerate(bg_latents_list):
                    input_image_identity[f'latents_list_{i}'] = latent.detach().cpu().numpy()
                input_image_identity['latents_list_len'] = len(bg_latents_list)

                np.savez(bg_image_identity_path, **input_image_identity)

        # set the foreground object to get an updated background depth
        # bg_depth = diff_handles.set_foreground(depth=depth, fg_mask=fg_mask, bg_depth=bg_depth)

        # save image reconstructed from inversion
        with torch.no_grad():
            fg_latent_image = 1 / 0.18215 * fg_latent_image.detach()
            fg_recon_image = diff_handles.diffuser.vae.decode(fg_latent_image)['sample']
            fg_recon_image = (fg_recon_image + 1) / 2
        save_image(fg_recon_image.clamp(min=0, max=1)[0], join(output_dir, sample_name, 'fg_recon.png'))

        with torch.no_grad():
            bg_latent_image = 1 / 0.18215 * bg_latent_image.detach()
            bg_recon_image = diff_handles.diffuser.vae.decode(bg_latent_image)['sample']
            bg_recon_image = (bg_recon_image + 1) / 2
        save_image(bg_recon_image.clamp(min=0, max=1)[0], join(output_dir, sample_name, 'bg_recon.png'))

        # iterate over the transformations
        for transform_name in transform_names:

            if transform_name not in transforms:
                print(f'WARNING: Transform {transform_name} not found for image {sample_name}. Skipping.')
                continue

            if skip_existing and transform_exists[transform_name]:
                print(f'Skipping transform {transform_name} of image {sample_name}, since its output already exists.')
                continue

            transform = transforms[transform_name]

            # get transformation parameters
            translation = torch.tensor(transform['translation'], dtype=torch.float32) if 'translation' in transform else None
            rot_axis = torch.tensor(transform['rotation_axis'], dtype=torch.float32) if 'rotation_axis' in transform else None
            rot_angle = float(transform['rotation_angle']) if 'rotation_angle' in transform else None

            # transform the foreground object
            results = diff_handles.transform_foreground_composite(
                fg_prompt=fg_prompt, bg_prompt=bg_prompt, composite_prompt=composite_prompt,
                fg_depth=fg_depth, bg_depth=bg_depth, composite_depth=composite_depth, composite_inverted_depth=composite_inverted_depth,
                fg_mask=fg_mask,
                composite_mask=composite_mask, 
                fg_latents=fg_latents_list, bg_latents=bg_latents_list, 
                fg_null_text_emb=fg_null_text_emb, bg_null_text_emb=bg_null_text_emb,
                fg_init_noise=fg_init_noise, bg_init_noise=bg_init_noise,
                fg_activations=fg_activations, bg_activations=bg_activations,
                rot_angle=rot_angle, rot_axis=rot_axis, translation=translation,
                use_input_depth_normalization=False)

            if diff_handles.conf.guided_diffuser.save_denoising_steps:
                edited_img, edited_disparity, denoising_steps = results
            else:
                edited_img, edited_disparity = results
                denoising_steps = None

            # Diff-Harmonization
            controller = AttentionReplace(
                prompts = harmonize_prompts,
                num_steps = diff_handles.conf.guided_diffuser.num_timesteps,
                tokenizer = diff_handles.diffuser.tokenizer,
                cross_replace_steps=1., 
                self_replace_steps=1.,
                device=device
            )

            save_diffhandles_outputs(
                edited_img=edited_img, edited_disparity=edited_disparity, denoising_steps=denoising_steps,
                output_dir=output_dir, sample_name=sample_name, transform_name=transform_name)

    # save sample names to result directory
    with open(join(output_dir, basename(test_set_path)), 'w') as f:
        json.dump(dataset_names, f, indent=4)

def preprocess_samples(input_dir, dataset_names):
    
    # check samples for completeness
    incomplete_samples = []
    foreground_removal_samples = []
    depth_estimation_samples = []
    for sample_name in dataset_names.keys():
        required_fnames = ['input.png', 'mask.png', 'prompt.txt', 'transforms.json']
        if any(not exists(join(input_dir, sample_name, fname)) for fname in required_fnames):
            print(f"Skipping sample {sample_name}, since it is missing one of the required input files ({required_fnames}).")
            incomplete_samples.append(sample_name)
            continue
        if not exists(join(input_dir, sample_name, 'bg_depth.exr')) and not exists(join(input_dir, sample_name, 'bg.png')):
            foreground_removal_samples.append(sample_name)
        if not exists(join(input_dir, sample_name, 'depth.exr')) or not exists(join(input_dir, sample_name, 'bg_depth.exr')):
            depth_estimation_samples.append(sample_name)

    # remove incomplete samples
    for sample_name in incomplete_samples:
        del dataset_names[sample_name]

    # estimate missing background images (with removed foreground object)
    if len(foreground_removal_samples) > 0:
        print(f"Estimating background images for {len(foreground_removal_samples)} samples ...")
        remove_foreground(
            input_image_paths=[join(input_dir, sample_name, 'input.png') for sample_name in foreground_removal_samples],
            foreground_mask_paths=[join(input_dir, sample_name, 'mask.png') for sample_name in foreground_removal_samples],
            output_paths=[join(input_dir, sample_name, 'bg.png') for sample_name in foreground_removal_samples])

    # estimate missing depths
    if len(depth_estimation_samples) > 0:
        print(f"Estimating depth and background depth for {len(depth_estimation_samples)} samples ...")
        estimate_depth(
            input_image_paths=[
                join(input_dir, sample_name, 'input.png') for sample_name in depth_estimation_samples]+[
                join(input_dir, sample_name, 'bg.png') for sample_name in depth_estimation_samples],
            output_paths=[
                join(input_dir, sample_name, 'depth.exr') for sample_name in depth_estimation_samples]+[
                join(input_dir, sample_name, 'bg_depth.exr') for sample_name in depth_estimation_samples]
            )

def load_diffhandles_inputs(input_dir, sample_name, output_dir, img_res, device, skip_existing):

    sample_dir = join(input_dir, sample_name)
    
    # load transforms
    with open(join(sample_dir, 'transforms.json'), 'r') as f:
        transforms = json.load(f, object_pairs_hook=OrderedDict)

    # if all outputs already exist, skip sample
    transform_exists = None
    if skip_existing:
        transform_exists = OrderedDict()
        for transform_name, transform in transforms.items():
            transform_path = join(output_dir, sample_name, f'{transform_name}.png')
            transform_exists[transform_name] = exists(transform_path)
        if all(transform_exists.values()):
            print(f'Skipping sample {sample_name}, since all outputs already exist.')
            return None, None, None, None, None, None, None

    # load prompt
    with open(join(sample_dir, 'prompt.txt'), 'r') as f:
        prompt = f.read().splitlines()
    prompt = [prompt for prompt in prompt if len(prompt) > 0]
    if len(prompt) == 0:
        print(f'WARNING: prompt for sample {sample_name} is empty. Skipping sample.')
        return None, None, None, None, None, None, None
    else:
        if len(prompt) > 1:
            print(f'WARNING: prompt for sample {sample_name} has multiple lines, only using the first line.')
        prompt = prompt[0]

    # load the input image
    img = load_image(join(sample_dir, 'input.png'))[None, ...]
    if img.shape[-2:] != (img_res, img_res):
        print(f"WARNING: Resizing and cropping image from {img.shape[-2]}x{img.shape[-1]} to {img_res}x{img_res}.")
        img = crop_and_resize(img=img, size=img_res)
    img = img.to(device)

    # load the foreground mask
    fg_mask = load_image(join(sample_dir, 'mask.png'))[None, ...]
    if fg_mask.shape[1] > 1:
        fg_mask = fg_mask.mean(dim=1, keepdim=True) # average channels
    fg_mask = crop_and_resize(img=fg_mask, size=img_res)
    fg_mask = (fg_mask>0.5).to(device=device, dtype=torch.float32)

    # load the input image depth
    depth = load_depth(join(sample_dir, 'depth.exr'))[None, ...]
    depth = crop_and_resize(img=depth, size=img_res)
    depth = depth.to(device=device, dtype=torch.float32)

    # load the background depth
    bg_depth = load_depth(join(sample_dir, 'bg_depth.exr'))[None, ...]
    bg_depth = crop_and_resize(img=bg_depth, size=img_res)
    bg_depth = bg_depth.to(device=device, dtype=torch.float32)

    return transforms, prompt, img, fg_mask, depth, bg_depth, transform_exists

def load_diffhandles_inputs_composite(input_dir, sample_name, output_dir, img_res, device, skip_existing):

    sample_dir = join(input_dir, sample_name)
    
    # load transforms
    with open(join(sample_dir, 'transforms.json'), 'r') as f:
        transforms = json.load(f, object_pairs_hook=OrderedDict)

    # if all outputs already exist, skip sample
    transform_exists = None
    if skip_existing:
        transform_exists = OrderedDict()
        for transform_name, transform in transforms.items():
            transform_path = join(output_dir, sample_name, f'{transform_name}.png')
            transform_exists[transform_name] = exists(transform_path)
        if all(transform_exists.values()):
            print(f'Skipping sample {sample_name}, since all outputs already exist.')
            return None, None, None, None, None, None, None

    # load prompts
    # fg
    with open(join(sample_dir, 'fg_prompt.txt'), 'r') as f:
        fg_prompt = f.read().splitlines()
    fg_prompt = [fg_prompt for fg_prompt in fg_prompt if len(fg_prompt) > 0]
    if len(fg_prompt) == 0:
        print(f'WARNING: fg prompt for sample {sample_name} is empty. Skipping sample.')
        return None, None, None, None, None, None, None
    else:
        if len(fg_prompt) > 1:
            print(f'WARNING: fg prompt for sample {sample_name} has multiple lines, only using the first line.')
        fg_prompt = fg_prompt[0]
    # bg
    with open(join(sample_dir, 'bg_prompt.txt'), 'r') as f:
        bg_prompt = f.read().splitlines()
    bg_prompt = [bg_prompt for bg_prompt in bg_prompt if len(bg_prompt) > 0]
    if len(bg_prompt) == 0:
        print(f'WARNING: bg prompt for sample {sample_name} is empty. Skipping sample.')
        return None, None, None, None, None, None, None
    else:
        if len(bg_prompt) > 1:
            print(f'WARNING: bg prompt for sample {sample_name} has multiple lines, only using the first line.')
        bg_prompt = bg_prompt[0]
    # composite
    with open(join(sample_dir, 'composite_prompt.txt'), 'r') as f:
        composite_prompt = f.read().splitlines()
    composite_prompt = [composite_prompt for composite_prompt in composite_prompt if len(composite_prompt) > 0]
    if len(composite_prompt) == 0:
        print(f'WARNING: composite prompt for sample {sample_name} is empty. Skipping sample.')
        return None, None, None, None, None, None, None
    else:
        if len(composite_prompt) > 1:
            print(f'WARNING: composite prompt for sample {sample_name} has multiple lines, only using the first line.')
        composite_prompt = composite_prompt[0]
    # fg harmonize
    with open(join(sample_dir, 'fg_harmonize_prompt.txt'), 'r') as f:
        fg_harmonize_prompt = f.read().splitlines()
    fg_harmonize_prompt = [fg_harmonize_prompt for fg_harmonize_prompt in fg_harmonize_prompt if len(fg_harmonize_prompt) > 0]
    if len(fg_harmonize_prompt) == 0:
        print(f'WARNING: fg harmonize prompt for sample {sample_name} is empty. Skipping sample.')
        return None, None, None, None, None, None, None
    else:
        if len(fg_harmonize_prompt) > 1:
            print(f'WARNING: fg harmonize prompt for sample {sample_name} has multiple lines, only using the first line.')
        fg_harmonize_prompt = fg_harmonize_prompt[0]
    # bg harmonize
    with open(join(sample_dir, 'bg_harmonize_prompt.txt'), 'r') as f:
        bg_harmonize_prompt = f.read().splitlines()
    bg_harmonize_prompt = [bg_harmonize_prompt for bg_harmonize_prompt in bg_harmonize_prompt if len(bg_harmonize_prompt) > 0]
    if len(bg_harmonize_prompt) == 0:
        print(f'WARNING: bg harmonize prompt for sample {sample_name} is empty. Skipping sample.')
        return None, None, None, None, None, None, None
    else:
        if len(bg_harmonize_prompt) > 1:
            print(f'WARNING: bg harmonize prompt for sample {sample_name} has multiple lines, only using the first line.')
        bg_harmonize_prompt = bg_harmonize_prompt[0]

    harmonize_prompts = [fg_harmonize_prompt, bg_harmonize_prompt]

    # load the images
    # fg
    fg_img = load_image(join(sample_dir, 'fg.png'))[None, ...]
    if fg_img.shape[-2:] != (img_res, img_res):
        print(f"WARNING: Resizing and cropping image from {fg_img.shape[-2]}x{fg_img.shape[-1]} to {img_res}x{img_res}.")
        fg_img = crop_and_resize(img=fg_img, size=img_res)
    fg_img = fg_img.to(device)
    # bg
    bg_img = load_image(join(sample_dir, 'bg.png'))[None, ...]
    if bg_img.shape[-2:] != (img_res, img_res):
        print(f"WARNING: Resizing and cropping image from {bg_img.shape[-2]}x{bg_img.shape[-1]} to {img_res}x{img_res}.")
        bg_img = crop_and_resize(img=bg_img, size=img_res)
    bg_img = bg_img.to(device)

    # load the masks
    # fg
    fg_mask = load_image(join(sample_dir, 'fg_mask.png'))[None, ...]
    if fg_mask.shape[1] > 1:
        fg_mask = fg_mask.mean(dim=1, keepdim=True) # average channels
    fg_mask = crop_and_resize(img=fg_mask, size=img_res)
    fg_mask = (fg_mask>0.5).to(device=device, dtype=torch.float32)
    # composite
    composite_mask = load_image(join(sample_dir, 'composite_mask.png'))[None, ...]
    if composite_mask.shape[1] > 1:
        composite_mask = composite_mask.mean(dim=1, keepdim=True) # average channels
    composite_mask = crop_and_resize(img=composite_mask, size=img_res)
    composite_mask = (composite_mask>0.5).to(device=device, dtype=torch.float32)

    # load depths
    # fg
    fg_depth = load_depth(join(sample_dir, 'fg_depth.exr'))[None, ...]
    fg_depth = crop_and_resize(img=fg_depth, size=img_res)
    fg_depth = fg_depth.to(device=device, dtype=torch.float32)
    # bg
    bg_depth = load_depth(join(sample_dir, 'bg_depth.exr'))[None, ...]
    bg_depth = crop_and_resize(img=bg_depth, size=img_res)
    bg_depth = bg_depth.to(device=device, dtype=torch.float32)

    # --- composite ---
    
    # fg_depth_intrinsics = np.load(join(sample_dir, 'fg_depth_intrinsics.npy'))
    # bg_depth_intrinsics = np.load(join(sample_dir, 'bg_depth_intrinsics.npy'))
    # print("FVBN")
    # correspondences, composite_depth = depth_transform_pipeline(fg_img, bg_img, fg_mask, fg_depth, bg_depth, img_res, fg_depth_intrinsics, bg_depth_intrinsics)
    # composite_depth = torch.from_numpy(composite_depth).unsqueeze(0).unsqueeze(0)

    # import sys
    # sys.exit()

    composite_depth = load_depth(join(sample_dir, 'composite_depth.png'))[None, ...]

    composite_depth = crop_and_resize(img=composite_depth, size=img_res)
    composite_depth = composite_depth.to(device=device, dtype=torch.float32)
    composite_inverted_depth = composite_depth.max() + composite_depth.min() - composite_depth

    # composite_inverted_depth = load_depth(join(sample_dir, 'composite_depth_inverted.png'))[None, ...]
    # composite_inverted_depth = crop_and_resize(img=composite_inverted_depth, size=img_res)
    # composite_inverted_depth = composite_inverted_depth.to(device=device, dtype=torch.float32)

    return transforms, fg_prompt, bg_prompt, composite_prompt, harmonize_prompts, fg_img, bg_img, fg_mask, composite_mask, fg_depth, bg_depth, composite_depth, composite_inverted_depth, transform_exists

def save_diffhandles_inputs(img, transforms, prompt, fg_mask, depth, bg_depth, input_dir, sample_name, output_dir, img_res):

    makedirs(join(output_dir, sample_name), exist_ok=True)
    
    # save transforms to output directory
    with open(join(output_dir, sample_name, 'transforms.json'), 'w') as f:
        json.dump(transforms, f, indent=4)

    # save prompt to output directory
    with open(join(output_dir, sample_name, 'prompt.txt'), 'w') as f:
        f.write(f'{prompt}\n')

    save_image(img[0], join(output_dir, sample_name, 'input.png'))
    save_image(fg_mask[0], join(output_dir, sample_name, 'fg_mask.png'))
    save_image((depth/depth.max())[0], join(output_dir, sample_name, 'depth.png'))
    save_image((bg_depth/bg_depth.max())[0], join(output_dir, sample_name, 'bg_depth.png'))
    if exists(join(input_dir, sample_name, 'bg.png')):
        bg_img = load_image(join(input_dir, sample_name, 'bg.png'))[None, ...]
        bg_img = crop_and_resize(img=bg_img, size=img_res)
        save_image(bg_img[0], join(output_dir, sample_name, 'bg.png'))

def save_diffhandles_inputs_composite(fg_prompt, bg_prompt, composite_prompt, harmonize_prompts, fg_img, bg_img, fg_mask, composite_mask, fg_depth, bg_depth, composite_depth, transforms, sample_name, output_dir):

    makedirs(join(output_dir, sample_name), exist_ok=True)
    
    # save transforms to output directory
    with open(join(output_dir, sample_name, 'transforms.json'), 'w') as f:
        json.dump(transforms, f, indent=4)

    # save prompts to output directory
    with open(join(output_dir, sample_name, 'fg_prompt.txt'), 'w') as f:
        f.write(f'{fg_prompt}\n')
    with open(join(output_dir, sample_name, 'bg_prompt.txt'), 'w') as f:
        f.write(f'{bg_prompt}\n')
    with open(join(output_dir, sample_name, 'composite_prompt.txt'), 'w') as f:
        f.write(f'{composite_prompt}\n')
    with open(join(output_dir, sample_name, 'fg_harmonize_prompt.txt'), 'w') as f:
        f.write(f'{harmonize_prompts[0]}\n')
    with open(join(output_dir, sample_name, 'bg_harmonize_prompt.txt'), 'w') as f:
        f.write(f'{harmonize_prompts[1]}\n')

    save_image(fg_img[0], join(output_dir, sample_name, 'fg.png'))
    save_image(bg_img[0], join(output_dir, sample_name, 'bg.png'))
    save_image(fg_mask[0], join(output_dir, sample_name, 'fg_mask.png'))
    save_image(composite_mask[0], join(output_dir, sample_name, 'composite_mask.png'))
    save_image((fg_depth/fg_depth.max())[0], join(output_dir, sample_name, 'fg_depth.png'))
    save_image((bg_depth/bg_depth.max())[0], join(output_dir, sample_name, 'bg_depth.png'))
    save_image((composite_depth/composite_depth.max())[0], join(output_dir, sample_name, 'composite_depth.png'))

def save_diffhandles_outputs(edited_img, edited_disparity, denoising_steps, output_dir, sample_name, transform_name):

    # save the edited depth
    save_image((edited_disparity/edited_disparity.max())[0], join(output_dir, sample_name, f'{transform_name}_disparity.png'))

    # save the edited image
    save_image(edited_img[0], join(output_dir, sample_name, f'{transform_name}.png'))

    if denoising_steps is not None:
        denoising_steps_dir = join(output_dir, sample_name, f'{transform_name}_denoising_steps')
        makedirs(denoising_steps_dir, exist_ok=True)
        for denoising_idx in range(len(denoising_steps['opt'])):
            for opt_idx in range(len(denoising_steps['opt'][denoising_idx])):
                denoising_steps['opt'][denoising_idx][opt_idx] = denoising_steps['opt'][denoising_idx][opt_idx]
                save_image(denoising_steps['opt'][denoising_idx][opt_idx][0], join(denoising_steps_dir, f'step_{denoising_idx}_opt_{opt_idx}.png', ))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_set_path', type=str, default='../custom_scripts/data/sample.json')
    parser.add_argument('--input_dir', type=str, default='../custom_scripts/data/')
    parser.add_argument('--output_dir', type=str, default='../custom_scripts/data/results/')
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--cache_input_image_identity', action='store_true')
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--mode', type=str, default=None)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    if args.mode == 'original':
        test_diffusion_handles(
            test_set_path=args.test_set_path,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            skip_existing=args.skip_existing,
            cache_input_image_identity=args.cache_input_image_identity,
            config_path=args.config_path,
            device=args.device)

    elif args.mode == 'custom':
        test_diffusion_handles_composite(
            test_set_path=args.test_set_path,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            skip_existing=args.skip_existing,
            cache_input_image_identity=args.cache_input_image_identity,
            config_path=args.config_path,
            device=args.device)

