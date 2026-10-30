"""First-order inference sampler, adapted from APEX/UCGM.

The flow uses t=1 for noise and t=0 for data. Each step evaluates the
conditional velocity once; no negative-prompt or fake-score branch is used.
"""

import torch


class UnifiedSampler:
    def __init__(self, steps=2, timesteps=None, stochastic_ratio=0.0):
        if steps not in (1, 2):
            raise ValueError("IDM inference supports one or two network evaluations.")
        self.timesteps = (
            list(timesteps) if timesteps is not None else [1.0 - i / steps for i in range(steps)]
        )
        if len(self.timesteps) != steps or not all(0 < t <= 1 for t in self.timesteps):
            raise ValueError("Provide one timestep in (0, 1] per network evaluation.")
        if any(a <= b for a, b in zip(self.timesteps, self.timesteps[1:])):
            raise ValueError("Timesteps must be strictly decreasing.")
        if not 0 <= stochastic_ratio <= 1:
            raise ValueError("stochastic_ratio must be in [0, 1].")
        self.stochastic_ratio = stochastic_ratio

    @torch.inference_mode()
    def __call__(self, noise, model, *, generator=None, **model_kwargs):
        dtype = noise.dtype
        current = noise.float()
        for t, next_t in zip(self.timesteps, self.timesteps[1:] + [0.0]):
            time = torch.full((len(noise),), t, dtype=dtype, device=noise.device)
            velocity = model(
                current.to(dtype), t=time, tt=torch.zeros_like(time), **model_kwargs
            ).float()
            if self.stochastic_ratio == 0 or next_t == 0:
                current = current + (next_t - t) * velocity
            else:
                endpoint = current - t * velocity
                predicted_noise = current + (1 - t) * velocity
                fresh_noise = torch.randn(
                    current.shape, generator=generator, device="cpu", dtype=torch.float32
                ).to(current.device)
                ratio = self.stochastic_ratio
                current = (1 - next_t) * endpoint + next_t * (
                    (1 - ratio) ** 0.5 * predicted_noise + ratio**0.5 * fresh_noise
                )
        return current.to(dtype)
