"""Qwen-Image inference adapter. Modified from LINs-lab/APEX."""

import torch
from diffusers import QwenImagePipeline, QwenImageTransformer2DModel

from ..common import ImageModel, load_pipeline
from .transformer_qwenimage import QwenImageTransformer2DModelWrapper


class GenTransformer(torch.nn.Module):
    def __init__(self, transformer, vae_scale_factor, aux_time_embed):
        super().__init__()
        self.transformer = transformer
        self.in_channels = transformer.config.in_channels // 4
        self.vae_scale_factor = vae_scale_factor
        self.aux_time_embed = aux_time_embed

    def forward(self, x_t, t, c, tt=None):
        batch, channels, height, width = x_t.shape
        packed = QwenImagePipeline._pack_latents(x_t, batch, channels, height, width)
        embeddings, mask = c
        lengths = mask.int().sum(dim=1).tolist()
        embeddings, mask = embeddings[:, : max(lengths)], mask[:, : max(lengths)]
        image_shapes = [[(1, height // 2, width // 2)]] * batch
        key_mask = torch.cat(
            (mask.bool(), torch.ones(batch, packed.shape[1], device=mask.device, dtype=torch.bool)),
            dim=1,
        )
        attention_mask = torch.zeros_like(key_mask, dtype=packed.dtype).masked_fill(
            ~key_mask, float("-inf")
        )[:, None, None, :]
        kwargs = dict(
            hidden_states=packed,
            timestep=t,
            guidance=None,
            encoder_hidden_states=embeddings,
            encoder_hidden_states_mask=mask,
            img_shapes=image_shapes,
            attention_kwargs={"attention_mask": attention_mask},
            return_dict=False,
        )
        if self.aux_time_embed:
            if tt is None:
                raise ValueError("An auxiliary-time checkpoint requires target timesteps.")
            kwargs["target_timestep"] = tt
        prediction = self.transformer(**kwargs)[0]
        return QwenImagePipeline._unpack_latents(
            prediction,
            height=height * self.vae_scale_factor,
            width=width * self.vae_scale_factor,
            vae_scale_factor=self.vae_scale_factor,
        ).squeeze(2)


class QwenImage(ImageModel):
    def __init__(self, model_id, *, dtype=torch.bfloat16, device="cuda", **kwargs):
        self.pipeline, self.aux_time_embed = load_pipeline(
            QwenImagePipeline,
            QwenImageTransformer2DModel,
            QwenImageTransformer2DModelWrapper,
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
        embeddings, mask = self.pipeline.encode_prompt(
            prompt=prompts,
            device=self.device,
            num_images_per_prompt=1,
            max_sequence_length=1024,
        )
        return embeddings.to(self.dtype), mask

    def latents_to_pixels(self, latents):
        vae = self.pipeline.vae
        value = latents.to(vae.dtype).unsqueeze(2)
        mean = torch.as_tensor(
            vae.config.latents_mean, device=value.device, dtype=value.dtype
        ).view(1, -1, 1, 1, 1)
        std = torch.as_tensor(vae.config.latents_std, device=value.device, dtype=value.dtype).view(
            1, -1, 1, 1, 1
        )
        return vae.decode(value * std + mean, return_dict=False)[0][:, :, 0]
