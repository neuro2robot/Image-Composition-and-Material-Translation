import pathlib
import sys

import torch
import torch.nn.functional as F
import scipy.ndimage
from omegaconf import OmegaConf, DictConfig

from diffhandles.stable_null_inverter import StableNullInverter
from diffhandles.guided_stable_diffuser import GuidedStableDiffuser
from diffhandles.depth_transform import transform_depth, normalize_depth, get_correspondences
from diffhandles.utils import solve_laplacian_depth, unpack_correspondences

import imageio.v3 as imageio
def save_image(img: torch.Tensor, path: str):
    img = img.detach().cpu()
    img = img * 255.0
    img = img.permute(1, 2, 0)
    img = img.to(dtype=torch.uint8)
    if img.shape[-1] == 1:
        img = img[..., 0]
    imageio.imwrite(path, img.numpy())


class DiffusionHandles:

    def __init__(self, conf: DictConfig=None):

        if conf is None:
            conf = OmegaConf.load(f'{pathlib.Path(__file__).parent.resolve()}/config/default.yaml')
        
        self.conf = conf

        self.diffuser = GuidedStableDiffuser(conf=self.conf.guided_diffuser)
        self.inverter = StableNullInverter(self.diffuser)

        self.device = torch.device('cpu')

    def to(self, device: torch.device = None):

        self.diffuser.to(device=device)
        self.inverter.to(device=device)

        self.device = device

        return self

    def invert_input_image(self, img: torch.Tensor, depth: torch.Tensor, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Invert an input image to allow applying diffusion handles to that input image.
        Use the outputs of this function as inputs to generate_input_image to work with a reconstructed version of the given input image.

        Args:
            img: The input image.
            depth: Depth of the input image.

        Returns:
            null_text_emb: The null text of the inverted input image.
            init_noise: The initial noise of the inverted input image.
        """

        disparity = normalize_depth(1.0/depth)

        # invert image to get noise and null text that can be used to reproduce the image
        _, init_noise, null_text_emb = self.inverter.invert(
            target_img=img, depth=disparity, prompt=prompt, num_inner_steps=5, verbose=True)

        return null_text_emb, init_noise

    def generate_input_image(
            self, depth: torch.Tensor, prompt: str, null_text_emb: torch.Tensor = None, init_noise: torch.Tensor = None
            ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """
        Generates the input image from a prompt and a depth map,
        and stores intermdiate activations that descibe the identity of objects in the image.
        Optionally accepts a null text embedding and a noise tensor that can be used
        to reconstruct an inverted input image.

        Args:
            depth: Depth of the input image.
            prompt: Full prompt for the input image.
            null_text_emb: A null text embedding (may come from image inversion).
            init_noise: A starting noise (may come from image inversion).

        Returns:
            inverted_null_text: A null text embedding that can be used to reconstruct the input image
            init_noise: A starting noise that can be used to reconstruct the input image.
            activations: Activations from the first diffusion inference pass (from layers 1-3 of the decoder of the UNet as a list with 3 entries).
            latent_image: latent encoding of the input image
        """

        disparity = normalize_depth(1.0/depth)

        # perform first diffusion inference pass to get intermediate features
        with torch.no_grad():
            activations, latent_image, null_text_emb, init_noise, latents_list = self.diffuser.initial_inference(
                init_latents=init_noise, depth=disparity, uncond_embeddings=null_text_emb,
                prompt=prompt)

        return null_text_emb, init_noise, activations, latent_image, latents_list
    
    def set_foreground(self, depth: torch.Tensor, fg_mask: torch.Tensor, bg_depth: torch.Tensor) -> torch.Tensor:
        """
        Select the foreground object in the input image. 

        Args:
            depth: Depth of the input image.
            fg_mask: A mask of the foreground object.
            bg_depth: Depth of the background (with removed foreground object).

        Returns:
            bg_depth: An updated background depth that has been adjusted to better match the depth of the input image.
        """

        # update the background depth by copying the input depth, but infilling the hole in the input depth
        # (where the foreground object used to be) with the background depth
        bg_depth = solve_laplacian_depth(
            depth[0, 0].cpu().numpy(),
            bg_depth[0, 0].cpu().numpy(),
            scipy.ndimage.binary_dilation(fg_mask[0, 0].cpu().numpy(), iterations=15))
        bg_depth = torch.from_numpy(bg_depth).to(device=self.device)[None, None]

        return bg_depth

    def transform_foreground(
            self, depth: torch.Tensor, prompt: str,
            fg_mask:torch.Tensor, bg_depth: torch.Tensor,
            null_text_emb: torch.Tensor, init_noise: torch.Tensor, 
            activations: list[torch.Tensor],
            rot_angle: float = None, rot_axis: torch.Tensor = None, translation: torch.Tensor = None,
            fg_weight: float = None, bg_weight: float = None, use_input_depth_normalization=False):
        """
        Transform the foreground object. The following steps are performed:
        1) The depth of the foreground object is 3D-transformed, giving us a an updated depth map and corresondences between old and new 2D image coordinates.
        2) The edited image is generated guided by intermediate features that are warped with the correspondences from the 3D transformation.

        Args:
            depth: Depth of the input image.
            prompt: Full prompt for the input image.
            fg_mask: A mask of the foreground object.
            bg_depth: Depth of the background (with removed foreground object).
            inverted_null_text: The null text of the inverted input image.
            inverted_noise: The noise of the inverted input image.
            activations: Activations from the first diffusion inference pass (from layers 1-3 of the decoder of the UNet as a list with 3 entries).
            rot_angle: Rotation angle in degrees.
            rot_axis: Rotation axis.
            translation: Translation vector.
            use_input_depth_normalization: Use the same normalization factor and bias as the input depth for the edited depth, to make the edited depth as similar to the input depth as possible.
        
        Returns:
            output_img: The edited image.
        """
        
        # 3d-transform depth
        with torch.no_grad():
            edited_disparity, correspondences = transform_depth(
                depth=depth, bg_depth=bg_depth, fg_mask=fg_mask,
                intrinsics=self.diffuser.get_depth_intrinsics(device=depth.device),
                rot_angle=rot_angle, rot_axis=rot_axis, translation=translation,
                use_input_depth_normalization=use_input_depth_normalization,
                depth_transform_mode=self.conf.depth_transform_mode)
            
        print(edited_disparity)
            
        # edited_disparity = bg_depth
        # save_image((depth/depth.max())[0], 'depth.png')
        # save_image((bg_depth/bg_depth.max())[0], 'bg_depth.png')
        # save_image((edited_disparity/edited_disparity.max())[0], 'edited_disparity.png')

        # perform second diffusion inference pass guided by the 3d-transformed features
        with torch.no_grad():
            results = self.diffuser.guided_inference(
                latents=init_noise, depth=edited_disparity, uncond_embeddings=null_text_emb,
                prompt=prompt,
                activations_orig=activations,
                correspondences=correspondences,
                fg_weight=fg_weight, bg_weight=bg_weight,
                save_denoising_steps=self.conf.guided_diffuser.save_denoising_steps)

        if self.conf.guided_diffuser.save_denoising_steps:
            edited_img, denoising_steps = results
            return edited_img, edited_disparity, denoising_steps
        else:
            edited_img = results
            return edited_img, edited_disparity
        
    def transform_foreground_composite(
            self, 
            fg_prompt: str, bg_prompt: str, composite_prompt: str,
            fg_depth: torch.Tensor, bg_depth: torch.Tensor, composite_depth: torch.Tensor, composite_inverted_depth: torch.Tensor,
            fg_mask:torch.Tensor,
            composite_mask:torch.Tensor, 
            fg_latents: list[torch.Tensor], bg_latents: list[torch.Tensor],
            fg_null_text_emb: torch.Tensor, bg_null_text_emb: torch.Tensor, 
            fg_init_noise: torch.Tensor, bg_init_noise: torch.Tensor, 
            fg_activations: list[torch.Tensor], bg_activations: list[torch.Tensor],
            rot_angle: float = None, rot_axis: torch.Tensor = None, translation: torch.Tensor = None,
            fg_weight: float = None, bg_weight: float = None, use_input_depth_normalization=False):
        
        # 3d-transform depth
        with torch.no_grad():
            # correspondences = torch.load("/diffusion/DiffusionHandles/ml-depth-pro/data_process/correspondences.pt")
            correspondences, rendered_depth = transform_depth(
                depth=composite_depth, bg_depth=bg_depth, fg_mask=composite_mask,
                intrinsics=self.diffuser.get_depth_intrinsics(device=composite_depth.device),
                rot_angle=rot_angle, rot_axis=rot_axis, translation=translation,
                use_input_depth_normalization=use_input_depth_normalization,
                depth_transform_mode=self.conf.depth_transform_mode)
                
            # original_x, original_y, transformed_x, transformed_y = unpack_correspondences(correspondences)
            # src_depth = 1.0 / composite_depth[0, 0].detach().cpu().numpy()
            # tgt_depth = rendered_depth  # recommended if you returned it
            # visualize_correspondences_to_file(
            #     depth_src=src_depth,
            #     depth_tgt=tgt_depth,
            #     original_x=original_x,
            #     original_y=original_y,
            #     transformed_x=transformed_x,
            #     transformed_y=transformed_y,
            #     save_path="./depth_correspondences.png",
            #     max_lines=20
            # )
            
            # import sys
            # sys.exit(0)

            
            # get_correspondences(
            #     depth_new=composite_depth, depth_orig=fg_depth, fg_mask=fg_mask, mask_new=composite_mask,
            #     intrinsics=self.diffuser.get_depth_intrinsics(device=composite_depth.device))        
        
        edited_disparity = composite_inverted_depth

        print("FG Mask Shape:", fg_mask.shape)
        # Assume `x` is shape [1, 1, 512, 512]
        fg_mask = F.interpolate(fg_mask, size=(64, 64), mode='bilinear', align_corners=False)  # [1, 1, 64, 64]
        fg_mask = fg_mask.expand(-1, 4, -1, -1)  # [1, 4, 64, 64]
        print("FG Mask Shape:", fg_mask.shape)

        print("Composite Mask Shape:", composite_mask.shape)
        # Assume `x` is shape [1, 1, 512, 512]
        composite_mask = F.interpolate(composite_mask, size=(64, 64), mode='bilinear', align_corners=False)  # [1, 1, 64, 64]
        composite_mask = composite_mask.expand(-1, 4, -1, -1)  # [1, 4, 64, 64]
        print("Composite Mask Shape:", composite_mask.shape)

        with torch.no_grad():
            results = self.diffuser.guided_inference_composite(
                bg_latents_init=bg_init_noise,
                fg_latents=fg_latents, bg_latents=bg_latents,
                depth=edited_disparity, fg_mask_latents=fg_mask, composite_mask_latents=composite_mask,
                fg_uncond_embeddings=fg_null_text_emb, bg_uncond_embeddings=bg_null_text_emb,
                fg_prompt=fg_prompt, bg_prompt=bg_prompt, composite_prompt=composite_prompt,
                fg_activations=fg_activations, bg_activations=bg_activations,
                correspondences=correspondences,
                fg_weight=fg_weight, bg_weight=bg_weight,
                save_denoising_steps=self.conf.guided_diffuser.save_denoising_steps)

        if self.conf.guided_diffuser.save_denoising_steps:
            edited_img, denoising_steps = results
            return edited_img, edited_disparity, denoising_steps
        else:
            edited_img = results
            return edited_img, edited_disparity

import os
import cv2
import numpy as np

def visualize_correspondences_to_file(
    depth_src,
    depth_tgt,
    original_x, original_y,
    transformed_x, transformed_y,
    save_path,
    max_lines=3000
):
    """
    Saves a side-by-side correspondence visualization to disk.
    """

    # Ensure numpy + flatten
    ox = np.asarray(original_x).squeeze()
    oy = np.asarray(original_y).squeeze()
    tx = np.asarray(transformed_x).squeeze()
    ty = np.asarray(transformed_y).squeeze()

    H, W = depth_src.shape

    def norm_depth(d):
        d = d.copy()
        d[np.isinf(d)] = 0
        d = (d - d.min()) / (d.max() - d.min() + 1e-6)
        return (d * 255).astype(np.uint8)

    src_img = cv2.cvtColor(norm_depth(depth_src), cv2.COLOR_GRAY2BGR)
    tgt_img = cv2.cvtColor(norm_depth(depth_tgt), cv2.COLOR_GRAY2BGR)

    canvas = np.hstack([src_img, tgt_img])

    # Subsample for clarity
    N = len(ox)
    idxs = np.random.choice(N, min(max_lines, N), replace=False)

    for i in idxs:
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        p1 = (int(ox[i]), int(oy[i]))
        p2 = (int(tx[i] + W), int(ty[i]))  # shift right image

        cv2.line(canvas, p1, p2, color, 1)
        cv2.circle(canvas, p1, 2, color, -1)
        cv2.circle(canvas, p2, 2, color, -1)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, canvas)
