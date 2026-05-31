import torch
from torch import nn, Tensor


class DiffusionSchedule(nn.Module):
    def __init__(
        self,
        *,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ):
        super().__init__()

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.timesteps = timesteps

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer(
            "sqrt_one_minus_alpha_bars",
            torch.sqrt(1.0 - alpha_bars),
        )

    def extract(self, values: Tensor, t: Tensor, x: Tensor) -> Tensor:
        out = values.gather(0, t)
        while out.ndim < x.ndim:
            out = out.unsqueeze(-1)
        return out

    def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        sqrt_ab = self.extract(self.sqrt_alpha_bars, t, x0)
        sqrt_1mab = self.extract(
            self.sqrt_one_minus_alpha_bars,
            t,
            x0,
        )
        return sqrt_ab * x0 + sqrt_1mab * noise

    def predict_x0_from_eps(
        self,
        xt: Tensor,
        t: Tensor,
        eps_pred: Tensor,
    ) -> Tensor:
        sqrt_ab = self.extract(self.sqrt_alpha_bars, t, xt)
        sqrt_1mab = self.extract(
            self.sqrt_one_minus_alpha_bars,
            t,
            xt,
        )
        return (xt - sqrt_1mab * eps_pred) / sqrt_ab.clamp_min(1e-8)
