"""Z-Image inference adapter. Modified from LINs-lab/APEX."""

import torch
from diffusers import ZImagePipeline, ZImageTransformer2DModel

from ..common import ImageModel, load_pipeline
from .transformer_z_image import ZImageTransformer2DModelWrapper


class GenTransformer(torch.nn.Module):
    def __init__(self, transformer, vae_scale_factor, aux_time_embed):
        super().__init__()
        self.transformer = transformer
        self.in_channels = transformer.config.in_channels
        self.vae_scale_factor = vae_scale_factor
        self.aux_time_embed = aux_time_embed

    def forward(self, x_t, t, c, tt=None):
        # Map sampler times to checkpoint times; keep zero as the target sentinel.
        kwargs = dict(
            x=list(x_t.unsqueeze(2).unbind(0)),
            t=t.sign() * (1 - t.abs()),
            cap_feats=c,
            patch_size=2,
            f_patch_size=1,
        )
        if self.aux_time_embed:
            if tt is None:
                raise ValueError("An auxiliary-time checkpoint requires target timesteps.")
            kwargs["target_timestep"] = tt.sign() * (1 - tt.abs())
        if not self.aux_time_embed:
            kwargs["return_dict"] = False
        output = self.transformer(**kwargs)[0]
        return -torch.stack(output).squeeze(2)


class ZImage(ImageModel):
    def __init__(self, model_id, *, dtype=torch.bfloat16, device="cuda", **kwargs):
        self.pipeline, self.aux_time_embed = load_pipeline(
            ZImagePipeline,
            ZImageTransformer2DModel,
            ZImageTransformer2DModelWrapper,
            model_id,
            dtype=dtype,
            device=device,
            **kwargs,
        )
        self.device, self.dtype = torch.device(device), dtype
        self.transformer = GenTransformer(
            self.pipeline.transformer, self.pipeline.vae_scale_factor, self.aux_time_embed
        ).eval()

    def encode_prompt(self, prompts):
        embeddings, _ = self.pipeline.encode_prompt(
            prompt=prompts,
            negative_prompt=None,
            do_classifier_free_guidance=False,
            device=self.device,
            max_sequence_length=1024,
        )
        return [embedding.to(self.dtype) for embedding in embeddings]

    def latents_to_pixels(self, latents):
        vae = self.pipeline.vae
        latents = latents.to(vae.dtype) / vae.config.scaling_factor + vae.config.shift_factor
        return vae.decode(latents, return_dict=False)[0]
