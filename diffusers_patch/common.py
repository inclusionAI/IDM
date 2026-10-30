"""Shared checkpoint loading and image sampling for IDM inference."""

import torch


def load_lora(pipeline, path, *, weight_name=None, scale=1.0, local_files_only=False):
    pipeline.load_lora_weights(
        path,
        adapter_name="idm",
        local_files_only=local_files_only,
        use_safetensors=True,
        weight_name=weight_name or "pytorch_lora_weights.safetensors",
    )
    pipeline.fuse_lora(lora_scale=scale)
    pipeline.unload_lora_weights()


def load_pipeline(
    pipeline_cls,
    transformer_cls,
    wrapper_cls,
    model_id,
    *,
    transformer_path=None,
    lora_path=None,
    lora_weight_name=None,
    lora_scale=1.0,
    aux_time_embed="auto",
    dtype=torch.bfloat16,
    device="cuda",
    local_files_only=False,
):
    location = transformer_path or model_id
    source = {"local_files_only": local_files_only}
    if transformer_path is None:
        source["subfolder"] = "transformer"
    config = transformer_cls.load_config(location, **source)
    checkpoint_class = config.get("_class_name")
    if checkpoint_class and checkpoint_class not in (
        transformer_cls.__name__,
        wrapper_cls.__name__,
    ):
        raise ValueError(
            f"Checkpoint architecture {checkpoint_class} does not match the selected backbone "
            f"{transformer_cls.__name__}. Check --backbone and the weight path."
        )
    if aux_time_embed not in ("auto", "on", "off"):
        raise ValueError("aux_time_embed must be auto, on, or off.")
    auxiliary = (
        config.get("_class_name") == wrapper_cls.__name__
        if aux_time_embed == "auto"
        else aux_time_embed == "on"
    )
    cls = wrapper_cls if auxiliary else transformer_cls
    transformer, info = cls.from_pretrained(
        location,
        **source,
        torch_dtype=dtype,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        output_loading_info=True,
    )
    errors = {
        k: info[k]
        for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        if info.get(k)
    }
    if errors:
        raise ValueError(
            f"Checkpoint does not match {cls.__name__}: {errors}. "
            "Check the backbone and --aux-time-embed setting; partially initialized models cannot be used."
        )
    pipeline = pipeline_cls.from_pretrained(
        model_id,
        transformer=transformer,
        torch_dtype=dtype,
        use_safetensors=True,
        local_files_only=local_files_only,
    )
    if lora_path:
        load_lora(
            pipeline,
            lora_path,
            weight_name=lora_weight_name,
            scale=lora_scale,
            local_files_only=local_files_only,
        )
    pipeline.to(device)
    for module in (pipeline.transformer, pipeline.text_encoder, pipeline.vae):
        module.requires_grad_(False).eval()
    return pipeline, auxiliary


class ImageModel:
    @torch.inference_mode()
    def sample(self, prompts, *, seed=42, height=1024, width=1024, sampler):
        if isinstance(prompts, str) or not prompts or any(not p.strip() for p in prompts):
            raise ValueError("prompts must be a nonempty list of nonempty strings.")
        multiple = self.pipeline.vae_scale_factor * 2
        if height <= 0 or width <= 0 or height % multiple or width % multiple:
            raise ValueError(f"Image height and width must be positive multiples of {multiple}.")
        conditions = self.encode_prompt(prompts)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(
            len(prompts),
            self.transformer.in_channels,
            height // self.pipeline.vae_scale_factor,
            width // self.pipeline.vae_scale_factor,
            generator=generator,
            dtype=self.dtype,
            device="cpu",
        ).to(self.device)
        latents = sampler(noise, self.transformer, generator=generator, c=conditions)
        return self.latents_to_pixels(latents)
