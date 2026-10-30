"""CPU checks of sampling, model interfaces, and checkpoint compatibility."""

from pathlib import Path

import pytest
import torch
from diffusers import QwenImageTransformer2DModel, ZImageTransformer2DModel

from diffusers_patch.common import load_lora, load_pipeline
from diffusers_patch.qwen_image.modeling_qwen_image import GenTransformer as QwenAdapter
from diffusers_patch.qwen_image.transformer_qwenimage import QwenImageTransformer2DModelWrapper
from diffusers_patch.z_image.modeling_z_image import GenTransformer as ZAdapter
from diffusers_patch.z_image.transformer_z_image import ZImageTransformer2DModelWrapper
from inference import parse_args
from unified_sampler import UnifiedSampler


def tiny_model(backbone, auxiliary):
    if backbone == "qwen":
        cls = QwenImageTransformer2DModelWrapper if auxiliary else QwenImageTransformer2DModel
        return cls(
            in_channels=16,
            out_channels=4,
            num_layers=1,
            attention_head_dim=16,
            num_attention_heads=2,
            joint_attention_dim=8,
            axes_dims_rope=(4, 6, 6),
        )
    cls = ZImageTransformer2DModelWrapper if auxiliary else ZImageTransformer2DModel
    return cls(
        in_channels=4,
        dim=32,
        n_layers=1,
        n_refiner_layers=1,
        n_heads=4,
        n_kv_heads=4,
        cap_feat_dim=8,
        axes_dims=[2, 2, 4],
        axes_lens=[128, 32, 32],
    )


@pytest.mark.parametrize("steps", [1, 2])
def test_euler_endpoint_and_nfe(steps):
    calls = []
    noise = torch.ones(2, 4, 4, 4)

    def velocity(x, t, tt):
        assert not torch.is_grad_enabled()
        assert t.shape == tt.shape == (2,)
        assert not tt.any()
        calls.append(t[0].item())
        return torch.full_like(x, 3.0)

    out = UnifiedSampler(steps)(noise, velocity)
    torch.testing.assert_close(out, torch.full_like(noise, -2.0))
    assert len(calls) == steps


def test_two_step_euler_uses_updated_sample():
    # dx/dt = x: two steps of -0.5 multiply x by (1-0.5)^2.
    noise = torch.ones(1, 4, 4, 4)
    out = UnifiedSampler(2)(noise, lambda x, **_: x)
    torch.testing.assert_close(out, noise * 0.25)


def test_stochastic_sampling_uses_supplied_seed():
    sampler = UnifiedSampler(2, stochastic_ratio=0.7)
    noise = torch.ones(1, 4, 4, 4)

    def run(seed):
        return sampler(noise, lambda x, **_: x * 0.1, generator=torch.Generator().manual_seed(seed))

    torch.manual_seed(99)
    first = run(42)
    torch.manual_seed(1)
    torch.testing.assert_close(first, run(42))
    assert not torch.equal(first, run(43))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"steps": 0},
        {"steps": 3},
        {"timesteps": [1]},
        {"timesteps": [0.5, 1]},
        {"timesteps": [1, 0]},
        {"timesteps": [1, float("nan")]},
        {"stochastic_ratio": 1.1},
    ],
)
def test_invalid_schedules(kwargs):
    with pytest.raises(ValueError):
        UnifiedSampler(**kwargs)


@pytest.mark.parametrize("backbone", ["z", "qwen"])
@pytest.mark.parametrize("auxiliary", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_real_transformer_forward(backbone, auxiliary, dtype):
    torch.manual_seed(42)
    model = tiny_model(backbone, auxiliary).to(dtype).eval()
    adapter = (ZAdapter if backbone == "z" else QwenAdapter)(model, 8, auxiliary)
    noise = torch.randn(2, 4, 4, 4, dtype=dtype)
    features = [torch.randn(3, 8, dtype=dtype), torch.randn(5, 8, dtype=dtype)]
    if backbone == "qwen":
        features = [
            torch.nn.utils.rnn.pad_sequence(features, batch_first=True),
            torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool),
        ]
    out = UnifiedSampler(2)(noise, adapter, c=features)
    assert out.shape == noise.shape
    assert out.dtype == dtype
    assert torch.isfinite(out).all()


class LocalPipeline:
    """Minimal pipeline around a real saved transformer; no hub access."""

    @classmethod
    def from_pretrained(cls, model_id, *, transformer, **kwargs):
        obj = cls()
        obj.transformer = transformer
        obj.text_encoder = torch.nn.Linear(1, 1)
        obj.vae = torch.nn.Linear(1, 1)
        return obj

    def to(self, device):
        for module in (self.transformer, self.text_encoder, self.vae):
            module.to(device)
        return self


@pytest.mark.parametrize("backbone", ["z", "qwen"])
@pytest.mark.parametrize("auxiliary", [False, True])
def test_checkpoint_round_trip(tmp_path, backbone, auxiliary):
    original = tiny_model(backbone, auxiliary)
    original.save_pretrained(tmp_path / "transformer")
    base, wrapped = (
        (ZImageTransformer2DModel, ZImageTransformer2DModelWrapper)
        if backbone == "z"
        else (QwenImageTransformer2DModel, QwenImageTransformer2DModelWrapper)
    )
    pipeline, detected = load_pipeline(
        LocalPipeline,
        base,
        wrapped,
        str(tmp_path),
        device="cpu",
        dtype=torch.float32,
        local_files_only=True,
    )
    assert detected == auxiliary
    assert not pipeline.transformer.training
    assert all(not p.requires_grad for p in pipeline.transformer.parameters())
    for name, param in original.state_dict().items():
        torch.testing.assert_close(pipeline.transformer.state_dict()[name], param)
    # A wrong override must not silently ignore or initialize auxiliary weights.
    with pytest.raises((ValueError, RuntimeError)):
        load_pipeline(
            LocalPipeline,
            base,
            wrapped,
            str(tmp_path),
            device="cpu",
            dtype=torch.float32,
            local_files_only=True,
            aux_time_embed="off" if auxiliary else "on",
        )


def test_cli_prompt_file_and_validation(tmp_path):
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("菜单\n\nA book cover.\n", encoding="utf-8")
    args = parse_args(["--model", "local", "--backbone", "z-image", "--prompt-file", str(prompts)])
    assert args.prompts == ["菜单", "A book cover."]
    for extra in (
        ["--width", "1023"],
        ["--steps", "3"],
        ["--timesteps", "1", "0"],
        ["--num-images", "0"],
    ):
        with pytest.raises(SystemExit):
            parse_args(["--model", "local", "--backbone", "z-image", "--prompt", "test", *extra])


def test_readme_assets_exist():
    import re

    root = Path(__file__).resolve().parents[1]
    for asset in re.findall(r'src="(assets/[^"]+)"', (root / "README.md").read_text()):
        assert (root / asset).is_file()


@pytest.mark.parametrize("backbone", ["z", "qwen"])
def test_diffusers_lora_load_and_merge(tmp_path, backbone):
    from diffusers import QwenImagePipeline, ZImagePipeline
    from peft import LoraConfig, get_peft_model_state_dict

    torch.manual_seed(4)
    model = tiny_model(backbone, False)
    original_state = {k: v.clone() for k, v in model.state_dict().items()}
    config = LoraConfig(r=2, lora_alpha=2, target_modules=["to_q"])
    model.add_adapter(config)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                parameter.fill_(0.05)
    module_name, adapted = next((n, m) for n, m in model.named_modules() if n.endswith("to_q"))
    input_tensor = torch.randn(2, adapted.in_features)
    expected = adapted(input_tensor).detach()
    pipeline_cls = ZImagePipeline if backbone == "z" else QwenImagePipeline
    pipeline_cls.save_lora_weights(
        tmp_path, transformer_lora_layers=get_peft_model_state_dict(model)
    )
    restored = tiny_model(backbone, False)
    restored.load_state_dict(original_state)
    pipe = pipeline_cls(
        transformer=restored, vae=None, text_encoder=None, tokenizer=None, scheduler=None
    )
    load_lora(pipe, str(tmp_path), local_files_only=True)
    result = pipe.transformer.get_submodule(module_name)(input_tensor)
    torch.testing.assert_close(result, expected)
    assert not any("lora_" in n for n, _ in pipe.transformer.named_parameters())


def test_rejects_wrong_backbone_before_loading(tmp_path):
    tiny_model("z", False).save_pretrained(tmp_path / "transformer")
    with pytest.raises(ValueError, match="selected backbone"):
        load_pipeline(
            LocalPipeline,
            QwenImageTransformer2DModel,
            QwenImageTransformer2DModelWrapper,
            str(tmp_path),
            device="cpu",
            local_files_only=True,
        )


def test_cli_saves_images_metadata_and_does_not_overwrite(tmp_path, monkeypatch):
    import json

    import diffusers_patch.z_image.modeling_z_image as z_module
    import inference

    class DummyModel:
        aux_time_embed = False

        def __init__(self, *args, **kwargs):
            pass

        def sample(self, prompts, **kwargs):
            return torch.zeros(1, 3, kwargs["height"], kwargs["width"])

    monkeypatch.setattr(z_module, "ZImage", DummyModel)
    args = [
        "--model",
        "local",
        "--backbone",
        "z-image",
        "--prompt",
        "今日咖啡",
        "--device",
        "cpu",
        "--height",
        "16",
        "--width",
        "16",
        "--num-images",
        "2",
        "--output-dir",
        str(tmp_path),
    ]
    inference.main(args)
    assert len(list(tmp_path.glob("*.png"))) == 2
    metadata = [json.loads(p.read_text()) for p in sorted(tmp_path.glob("*.json"))]
    assert [m["seed"] for m in metadata] == [42, 43]
    assert all(m["prompt"] == "今日咖啡" and m["nfe"] == 2 for m in metadata)
    with pytest.raises(FileExistsError):
        inference.main(args)


@pytest.mark.parametrize("backbone", ["z", "qwen"])
def test_auxiliary_wrapper_matches_base_when_time_difference_is_zero(backbone):
    # With no target-time difference, the added embedding has zero coefficient.
    base = tiny_model(backbone, False).eval()
    wrapper = tiny_model(backbone, True).eval()
    missing, unexpected = wrapper.load_state_dict(base.state_dict(), strict=False)
    assert missing and not unexpected
    adapter_cls = ZAdapter if backbone == "z" else QwenAdapter
    original = adapter_cls(base, 8, False)
    adapted = adapter_cls(wrapper, 8, True)
    noise = torch.randn(2, 4, 4, 4)
    times = torch.tensor([0.5, 0.5])
    features = [torch.randn(3, 8), torch.randn(5, 8)]
    if backbone == "qwen":
        features = [
            torch.nn.utils.rnn.pad_sequence(features, batch_first=True),
            torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool),
        ]
    with torch.inference_mode():
        torch.testing.assert_close(
            adapted(noise, times, features, tt=times),
            original(noise, times, features),
            atol=1e-5,
            rtol=1e-5,
        )
