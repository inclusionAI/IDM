"""IDM inference CLI. Adapted from LINs-lab/APEX."""

import argparse
import json
import math
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate images with an IDM checkpoint.")
    parser.add_argument(
        "--model", required=True, help="Diffusers pipeline directory or Hugging Face repo ID."
    )
    parser.add_argument("--backbone", required=True, choices=("z-image", "qwen-image"))
    parser.add_argument(
        "--transformer",
        help="Optional transformer checkpoint directory or repo ID (config.json at root).",
    )
    parser.add_argument("--lora", help="Optional Diffusers-format LoRA directory or repo ID.")
    parser.add_argument("--lora-weight-name", help="LoRA .safetensors filename.")
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--aux-time-embed", choices=("auto", "on", "off"), default="auto")
    prompts = parser.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompt", help="One prompt, including quoted text to render.")
    prompts.add_argument(
        "--prompt-file", type=Path, help="UTF-8 file, one prompt per nonempty line."
    )
    parser.add_argument("--steps", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--timesteps",
        nargs="+",
        type=float,
        help="Descending noise levels, one per NFE; default: 1 (or 1, 0.5).",
    )
    parser.add_argument(
        "--stochastic-ratio",
        type=float,
        default=0.0,
        help="Noise refresh between steps, from 0 to 1.",
    )
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-images",
        type=int,
        default=1,
        help="Images per prompt; increment the seed for each output.",
    )
    parser.add_argument("--device", default="cuda", help="cuda, cuda:N, or cpu.")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only local files and the existing Hugging Face cache.",
    )
    args = parser.parse_args(argv)
    if min(args.height, args.width, args.num_images) < 1:
        parser.error("height, width, and num-images must be positive")
    if args.height % 16 or args.width % 16:
        parser.error("height and width must be multiples of 16")
    if not 0 <= args.seed < 2**63:
        parser.error("seed must be in [0, 2**63)")
    if not 0 <= args.stochastic_ratio <= 1:
        parser.error("stochastic-ratio must be in [0, 1]")
    if not math.isfinite(args.lora_scale) or args.lora_scale < 0:
        parser.error("lora-scale must be finite and nonnegative")
    if args.lora_weight_name and not args.lora:
        parser.error("lora-weight-name requires --lora")
    if args.timesteps is not None:
        ts = args.timesteps
        if len(ts) != args.steps or not all(0 < t <= 1 for t in ts):
            parser.error("timesteps must contain one value in (0, 1] per step")
        if any(a <= b for a, b in zip(ts, ts[1:])):
            parser.error("timesteps must be strictly decreasing")
    try:
        texts = (
            args.prompt_file.read_text(encoding="utf-8").splitlines()
            if args.prompt_file
            else [args.prompt]
        )
    except OSError as exc:
        parser.error(str(exc))
    args.prompts = [p.strip() for p in texts if p.strip()]
    if not args.prompts:
        parser.error("provide at least one nonempty prompt")
    if args.seed + len(args.prompts) * args.num_images > 2**63:
        parser.error("the requested seed range exceeds 2**63")
    return args


def main(argv=None):
    args = parse_args(argv)
    # Validate arguments before importing or loading any models.
    import torch
    from PIL import Image

    from unified_sampler import UnifiedSampler

    device = torch.device(args.device)
    if device.type not in ("cuda", "cpu"):
        raise SystemExit("Supported devices: cuda, cuda:N, cpu.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is unavailable. Run on a CUDA machine, or use --device cpu --dtype float32 for a small compatibility check."
        )
    if args.backbone == "z-image":
        from diffusers_patch.z_image.modeling_z_image import ZImage as Model
    else:
        from diffusers_patch.qwen_image.modeling_qwen_image import QwenImage as Model
    model = Model(
        args.model,
        transformer_path=args.transformer,
        lora_path=args.lora,
        lora_weight_name=args.lora_weight_name,
        lora_scale=args.lora_scale,
        aux_time_embed=args.aux_time_embed,
        dtype=getattr(torch, args.dtype),
        device=device,
        local_files_only=args.local_files_only,
    )
    sampler = UnifiedSampler(args.steps, args.timesteps, args.stochastic_ratio)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = 0
    for prompt in args.prompts:
        for _ in range(args.num_images):
            seed = args.seed + index
            stem = f"idm_{args.backbone}_{index:04d}_seed{seed}"
            image_path = args.output_dir / f"{stem}.png"
            metadata_path = args.output_dir / f"{stem}.json"
            if image_path.exists() or metadata_path.exists():
                raise FileExistsError(
                    f"Output already exists: {stem}. Choose another --output-dir or --seed."
                )
            pixels = model.sample(
                [prompt], seed=seed, height=args.height, width=args.width, sampler=sampler
            )
            array = (
                ((pixels[0].float().clamp(-1, 1) + 1) * 127.5)
                .round()
                .byte()
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            Image.fromarray(array).save(image_path)
            metadata = {
                "prompt": prompt,
                "seed": seed,
                "backbone": args.backbone,
                "model": args.model,
                "transformer": args.transformer,
                "lora": args.lora,
                "lora_weight_name": args.lora_weight_name,
                "lora_scale": args.lora_scale,
                "aux_time_embed": model.aux_time_embed,
                "nfe": args.steps,
                "timesteps": sampler.timesteps,
                "stochastic_ratio": args.stochastic_ratio,
                "height": args.height,
                "width": args.width,
                "dtype": args.dtype,
            }
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(image_path)
            index += 1


if __name__ == "__main__":
    main()
