import os
from typing import List
import numpy as np

import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from .utils import is_torch2_available, get_generator

if is_torch2_available():
    from .attention_processor import (
        AttnProcessor2_0 as AttnProcessor,
    )
    from .attention_processor import (
        CNAttnProcessor2_0 as CNAttnProcessor,
    )
    from .attention_processor import (
        IPAttnProcessor2_0 as IPAttnProcessor,
    )
else:
    from .attention_processor import AttnProcessor, CNAttnProcessor, IPAttnProcessor
from .resampler import Resampler

import torch
import torch.nn.functional as F
import open_clip
from torchvision import transforms
from PIL import Image

import torch

def get_noise_from_forward(x0, t, scheduler, return_noisy_latents=True):
    """
    Compute the noise used in the forward diffusion process and optionally return x_t.

    Args:
        x0 (torch.Tensor): Original clean latent (e.g., from VAE).
        t (torch.Tensor): Timestep(s) to use. Shape: (batch_size,)
        scheduler (SchedulerMixin): A scheduler instance from diffusers (e.g., DDIMScheduler).
        return_noisy_latents (bool): If True, returns the noisy latent x_t.

    Returns:
        noise (torch.Tensor): The actual noise added.
        x_t (torch.Tensor): The noisy latent at timestep t (if requested).
        recovered_noise (torch.Tensor): Reconstructed noise from x0 and x_t for sanity check.
    """
    device = x0.device
    batch_size = x0.shape[0]
    
    # Generate noise
    noise = torch.randn_like(x0)

    # Get alpha bar (cumulative product of alphas)
    alphas_cumprod = scheduler.alphas_cumprod.to(device)  # shape: (num_timesteps,)
    sqrt_alpha = alphas_cumprod[t].reshape(batch_size, 1, 1, 1).sqrt()
    sqrt_one_minus_alpha = (1 - alphas_cumprod[t]).reshape(batch_size, 1, 1, 1).sqrt()

    # Create x_t using forward diffusion
    x_t = sqrt_alpha * x0 + sqrt_one_minus_alpha * noise

    # Recover noise from x_t and x0 (sanity check)
    recovered_noise = (x_t - sqrt_alpha * x0) / sqrt_one_minus_alpha

    if return_noisy_latents:
        return noise, x_t, recovered_noise
    else:
        return noise, recovered_noise




def update_token_embeddings(
    image_path: str,
    new_token_name: str = "<token>",
    base_token_phrase: str = "rusted, transparent",
    placeholder_token: str = "X",
    prompt_template: str = "a photo of a {}",
    model_id: str = 'hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
    num_steps: int = 500,
    lr: float = 5e-4,
    save_path: str = "zest_learned_token.pt",
    device: str = "cuda"
):
    # Load CLIP model and tokenizer
    clip_model, _, preprocess = open_clip.create_model_and_transforms(model_id)
    clip_model = clip_model.to(device).eval().half()
    tokenizer = open_clip.get_tokenizer(model_id)

    # Load and preprocess the image
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device).half()
    with torch.no_grad():
        image_embed = clip_model.encode_image(image)
        image_embed = F.normalize(image_embed, dim=-1)

    # Initialize new token embedding from base phrase
    base_token_ids = tokenizer(base_token_phrase)
    base_token_ids = torch.tensor(base_token_ids).to(device)
    embedding_layer = clip_model.token_embedding

    with torch.no_grad():
        init_embedding = embedding_layer(base_token_ids).mean(dim=0, keepdim=True).clone().detach()

    new_embedding = init_embedding.to(device).half()
    new_embedding.requires_grad = True
    optimizer = torch.optim.Adam([new_embedding], lr=lr)

    # Tokenize prompt and find placeholder index
    prompt_text = prompt_template.format(placeholder_token)
    tokenized = tokenizer(prompt_text)
    tokenized = torch.tensor(tokenized).unsqueeze(0).to(device)
    placeholder_id = tokenizer(placeholder_token)[0]
    replace_index = (tokenized == placeholder_id).nonzero(as_tuple=False)[0, 1]

    # Training loop
    for step in range(num_steps + 1):
        with torch.no_grad():
            token_embeds = embedding_layer(tokenized)

        token_embeds[:, replace_index, :] = new_embedding

        text_embed = clip_model.encode_text(embeddings=token_embeds)
        text_embed = F.normalize(text_embed, dim=-1)


        




        loss = 1 - torch.cosine_similarity(text_embed, image_embed).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Token:")
        if step % 50 == 0:
            print(f"[Step {step:03d}] Loss: {loss.item():.4f}")

    #torch.save(new_embedding.detach().cpu(), save_path)
    #print(f" Token '{new_token_name}' trained and saved to: {save_path}")
    return new_embedding.detach()


class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class MLPProjModel(torch.nn.Module):
    """SD model with image prompt"""
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024):
        super().__init__()
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            torch.nn.GELU(),
            torch.nn.Linear(clip_embeddings_dim, cross_attention_dim),
            torch.nn.LayerNorm(cross_attention_dim)
        )
        
    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class IPAdapter:
    def __init__(self, sd_pipe, image_encoder_path, ip_ckpt, device, num_tokens=4):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens

        self.pipe = sd_pipe.to(self.device)
        self.set_ip_adapter()

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )
        self.clip_image_processor = CLIPImageProcessor()
        # image proj model
        self.image_proj_model = self.init_proj()

        self.load_ip_adapter()

    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    def set_ip_adapter(self):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = IPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=torch.float16)
        unet.set_attn_processor(attn_procs)
        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))
            else:
                self.pipe.controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))

    def load_ip_adapter(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"])

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if clip_image_embeds is None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image = clip_image.to(self.device, dtype=torch.float16)
            clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
            uncond_clip_image_embeds = self.image_encoder(
                torch.zeros_like(clip_image), output_hidden_states=True
            ).hidden_states[-2]
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
            uncond_clip_image_embeds = torch.zeros_like(clip_image_embeds)

        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds


    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

    def generate(
        self,
        pil_image=None,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=9,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)

        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = clip_image_embeds.size(0)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=clip_image_embeds
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images


class IPAdapterXL(IPAdapter):
    """SDXL"""

    def generate(
        self,
        pil_image=None,
        clip_image_embeds=None, 
        prompt=None, 
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)

        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = clip_image_embeds.shape[0]

        if prompt is None: # Default prompts
            prompt = "best quality, high quality" 
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image=pil_image,clip_image_embeds=clip_image_embeds) # CLIP embeddings of M and Zero tensor
        bs_embed, seq_len, hidden_dim = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.expand(bs_embed, num_samples, seq_len, hidden_dim).reshape(bs_embed * num_samples, seq_len, hidden_dim).contiguous()
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.expand(bs_embed, num_samples, seq_len, hidden_dim).reshape(bs_embed * num_samples, seq_len, hidden_dim).contiguous()

        '''
        self.features_all = []
        self.features_current = {}

        def get_feature_hook(name):
            def hook(module, input, output):
                self.features_current[name] = output
            return hook
        

        self.hooks = []
        #self.hooks.append(self.pipe.unet.conv_in.register_forward_hook(get_feature_hook("conv_in")))
        for idx, block in enumerate(self.pipe.unet.down_blocks):
            self.hooks.append(block.register_forward_hook(get_feature_hook(f"down_{idx}")))
        self.hooks.append(self.pipe.unet.mid_block.register_forward_hook(get_feature_hook("mid")))
        #for idx, block in enumerate(self.pipe.unet.up_blocks):
        #   self.hooks.append(block.register_forward_hook(get_feature_hook(f"up_{idx}")))
        #self.hooks.append(self.pipe.unet.conv_norm_out.register_forward_hook(get_feature_hook("conv_norm_out")))
        #self.hooks.append(self.pipe.unet.conv_out.register_forward_hook(get_feature_hook("conv_out")))

        # Monkey-patch UNet.forward
        orig_forward = self.pipe.unet.forward
        def hooked_forward(unet_self, *args, **kwargs):
            self.features_current.clear()
            output = orig_forward(*args, **kwargs)
            self.features_all.append({k: v for k, v in self.features_current.items()})
            return output
        self.pipe.unet.forward = hooked_forward.__get__(self.pipe.unet, type(self.pipe.unet))
        '''
        
        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1) 
            # This refers to the decoupled cross attention
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)
            # Concat negative prompt (default) with image prompt as we need to inject the uncoditional (null) image

        self.generator = get_generator(seed, self.device)
        
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=self.generator,
            **kwargs,
        ).images
        
        '''
        # Optionally print features summary
        if len(self.features_all) > 0:
            feat_dict = self.features_all[0]
            for layer_name, feat in feat_dict.items():
                #print(f"Layer: {layer_name} | Shape: {torch.tensor(feat).to('cpu')} | dtype: {feat.dtype}")
                if isinstance(feat, tuple):
                    for i, part in enumerate(feat):
                        if isinstance(part, torch.Tensor):
                            print(f"Layer: {layer_name}[{i}] | Shape: {tuple(part.shape)} | dtype: {part.dtype}")
                else:
                    print(f"Layer: {layer_name} | Shape: {tuple(feat.shape)} | dtype: {feat.dtype}")

        else:
            print("No features collected. Make sure UNet forward ran.")

        # Clean up hooks if you want
        for h in self.hooks:
            h.remove()
        self.hooks = []
        del self.hooks
        del prompt_embeds
        del negative_prompt_embeds
        del pooled_prompt_embeds
        del negative_pooled_prompt_embeds
        del image_prompt_embeds
        del uncond_image_prompt_embeds
        torch.cuda.empty_cache()
        '''
        return images
    
    def get_scale_shift_from_clip_embedding(self, clip_embed,
                                            CATEGORY_THRESHOLDS=[23.0, 23.7778, 24.5556, 25.3333, 26.1111, 26.8889, 27.6667, 28.4444, 29.2222, 30.0],
                                            CATEGORY_SCALE_SHIFT = {0: (1.3,0),
                                                                    1: (1.3, -0.1),
                                                                    2: (1.2, 0),
                                                                    3: (1.2, 0),
                                                                    4: (1.1, 0),
                                                                    5: (1.0, 0),
                                                                    6: (0.9, -0.2),
                                                                    7: (0.9, -0.2),
                                                                    8: (0.8, -0.2),
                                                                    9: (0.8, -0.2),
                                                                    10: (0.7, -0.2)}):
        """
        Given a CLIP image embedding, return (scale, shift) per sample based on the norm of the Key vector.
        Works by finding the attention module and its .to_v projection.
        """

        unet = self.pipe.unet

        # Step 1: Get attention processor name → attention module path
        for name, processor in unet.attn_processors.items():
            if "attn2" in name:  # only cross-attn blocks have .to_v
                # Get parent attention module
                attn_module = unet
                for key in name.replace(".processor", "").split("."):
                    if key.isdigit():
                        attn_module = attn_module[int(key)]
                    else:
                        attn_module = getattr(attn_module, key)

                # Check if it has .to_v and .heads
                if hasattr(attn_module, "to_v") and hasattr(attn_module, "heads"):
                    with torch.no_grad():
                        to_v = attn_module.to_v
                        heads = attn_module.heads
                        print("Clip embed shape ",clip_embed.shape)
                        K_proj = to_v(clip_embed)  # [B, heads * dim_head]
                        print("K_proj shape",K_proj.shape)
                        #K_proj = K_proj.view(B, heads * D_head)
                        #B, D_total = K_proj.shape
                        #D_head = D_total // heads
                        #K_proj = K_proj.view(B, heads, D_head)


                        norms = torch.norm(K_proj, dim=-1)
                        norms = torch.norm(norms,dim=-1) #.m   ean(dim=-1)  # [B]
                    
                        #norms = torch.norm(K_proj, p=2, dim=-1).squeeze()

                        print("norms ",norms.shape)

                        scale_shift = []
                        for norm_val in norms:
                            category = len(CATEGORY_THRESHOLDS)
                            for i, th in enumerate(CATEGORY_THRESHOLDS):
                                print("norm_val ",i,norm_val)
                                if norm_val.item() < th:
                                    
                                    category = i
                                    break
                            scale_shift.append(CATEGORY_SCALE_SHIFT[category])
                        return scale_shift

        raise RuntimeError("No attention module with 'to_v' found in UNet.")



class IPAdapterPlus(IPAdapter):
    """IP-Adapter with fine-grained features"""

    def init_proj(self):
        image_proj_model = Resampler(
            dim=self.pipe.unet.config.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if clip_image_embeds is None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image = clip_image.to(self.device, dtype=torch.float16)
            clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
            uncond_clip_image_embeds = self.image_encoder(
                torch.zeros_like(clip_image), output_hidden_states=True
            ).hidden_states[-2]
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
            uncond_clip_image_embeds = torch.zeros_like(clip_image_embeds)

        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds



class IPAdapterFull(IPAdapterPlus):
    """IP-Adapter with full features"""

    def init_proj(self):
        image_proj_model = MLPProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.hidden_size,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model


class IPAdapterPlusXL(IPAdapter):
    """SDXL"""

    def init_proj(self):
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if clip_image_embeds is None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image = clip_image.to(self.device, dtype=torch.float16)
            clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
            uncond_clip_image_embeds = self.image_encoder(
                torch.zeros_like(clip_image), output_hidden_states=True
            ).hidden_states[-2]
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
            uncond_clip_image_embeds = torch.zeros_like(clip_image_embeds)

        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds


    def generate(
        self,
        pil_image=None,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)

        #num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = clip_image_embeds.shape[0]

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=generator,
            #callback_on_step_end=update_token_embeddings,
            #callback_on_step_end_tensor_inputs=["noise_pred", "prompt_embeds"],
            **kwargs,
        ).images

        return images

