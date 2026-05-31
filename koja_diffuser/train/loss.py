from einops import rearrange
import torch
from torch import Tensor
from jaxtyping import Float, Int, Bool
from typing import Optional
from koja_diffuser.train.debug import Debug
from koja_diffuser.tokenizer.special import SpecialToken


class MMDLoss:
    @staticmethod
    def flatten_latent(z: Float[Tensor, "B L S"]) -> Float[Tensor, "B LS"]:
        return rearrange(z, "b l s -> b (l s)")

    @staticmethod
    def pairwise_sq_dist(x: Tensor, y: Tensor) -> Tensor:
        x_norm = (x**2).sum(dim=1, keepdim=True)
        y_norm = (y**2).sum(dim=1, keepdim=True).transpose(0, 1)
        dist = x_norm + y_norm - 2.0 * (x @ y.transpose(0, 1))
        return dist.clamp_min(0.0)

    @staticmethod
    @torch.no_grad()
    def estimate_bandwidth(
        x: Float[Tensor, "B LS"], y: Float[Tensor, "B LS"], eps=1e-4
    ) -> Tensor:
        z = torch.cat([x, y], dim=0)
        d2 = MMDLoss.pairwise_sq_dist(z, z)

        mask = d2 > 0
        if mask.any():
            sigma = d2[mask].median().sqrt()  # median heuristic
            return sigma.clamp_min(eps)
        return z.new_tensor(1.0)

    @staticmethod
    def off_diagonal_mean(k: Tensor) -> Tensor:
        n = k.size(0)
        if n <= 1:
            return k.mean()

        mask = ~torch.eye(n, dtype=torch.bool, device=k.device)
        return k[mask].mean()

    @staticmethod
    def mmd_rbf_loss(
        z_fake: Float[Tensor, "B L S"], z_real: Float[Tensor, "B L S"]
    ) -> Tensor:
        x = MMDLoss.flatten_latent(z_fake.float())
        y = MMDLoss.flatten_latent(z_real.detach().float())

        sigma = MMDLoss.estimate_bandwidth(x.detach(), y.detach())
        sigmas = [sigma * 0.5, sigma, sigma * 2.0]

        d_xx = MMDLoss.pairwise_sq_dist(x, x)
        d_yy = MMDLoss.pairwise_sq_dist(y, y)
        d_xy = MMDLoss.pairwise_sq_dist(x, y)

        k_xx = torch.zeros_like(d_xx)
        k_yy = torch.zeros_like(d_yy)
        k_xy = torch.zeros_like(d_xy)

        for s in sigmas:
            denom = 2.0 * (s**2)
            k_xx += torch.exp(-d_xx / denom)
            k_yy += torch.exp(-d_yy / denom)
            k_xy += torch.exp(-d_xy / denom)

        k_xx /= len(sigmas)
        k_yy /= len(sigmas)
        k_xy /= len(sigmas)

        return (
            MMDLoss.off_diagonal_mean(k_xx)
            + MMDLoss.off_diagonal_mean(k_yy)
            - 2.0 * k_xy.mean()
        )

    @staticmethod
    def direct_domain_loss(
        *,
        z_ja_hat: Tensor,
        z_ko_hat: Tensor,
        z_ja: Tensor,
        z_ko: Tensor,
        d: Optional[Debug] = None,
    ):
        loss_domain_ja = MMDLoss.mmd_rbf_loss(z_ja_hat, z_ja)
        loss_domain_ko = MMDLoss.mmd_rbf_loss(z_ko_hat, z_ko)

        total = loss_domain_ja + loss_domain_ko

        if d is not None:
            d.loss.domain_ja(loss_domain_ja)
            d.loss.domain_ko(loss_domain_ko)
            d.loss.domain_total(total)

        return total


def repeat_penalty_loss(
    logits: Tensor,
    *,
    temperature=1.0,
    exclude_token_ids: tuple[int, ...] = (SpecialToken.eos,),
) -> Tensor:
    probs = (logits.float() / temperature).softmax(dim=-1)

    if exclude_token_ids:
        probs = probs.clone()
        for token_id in exclude_token_ids:
            probs[..., token_id] = 0.0
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    p_prev = probs[:, :-1, :]
    p_next = probs[:, 1:, :]

    repeat_prob = (p_prev * p_next).sum(dim=-1)

    return repeat_prob.mean()


class CenterOneSepLoss:
    def __init__(self, logits: Float[Tensor, "B L S"], temperature=1.0):
        self.logits = logits
        self.probs = (self.logits.float() / temperature).softmax(dim=-1)
        self.device = logits.device

    def get_len(self) -> Int[Tensor, "B"]:
        # EOS 제외 길이
        pred_tokens = self.probs.argmax(dim=-1)
        is_eos = pred_tokens == SpecialToken.eos
        has_eos = is_eos.any(dim=-1)
        eos_indices = is_eos.int().argmax(dim=-1)
        seq_len = self.probs.size(1)
        return torch.where(
            has_eos, eos_indices, torch.tensor(seq_len, device=self.device)
        )

    def mirror_tensor(
        self, probs_len: Int[Tensor, "B"]
    ) -> tuple[Float[Tensor, "B L"], Bool[Tensor, "B L"]]:
        max_len = self.logits.size(1)

        pos = rearrange(torch.arange(max_len, device=self.device), "l -> 1 l")

        mirror = torch.minimum(pos, probs_len[:, None] - 1 - pos)

        peak = (probs_len[:, None] - 1) // 2
        denom = peak.clamp_min(1)

        out = mirror.float() / denom.float()

        mask = pos < probs_len[:, None]

        out = out.masked_fill(~mask, 0.0)

        return out, mask

    def loss(self):
        probs_len = self.get_len()
        sep_probs = self.probs[..., SpecialToken.sep]
        mirror_tensor, mask = self.mirror_tensor(probs_len)
        pad_mass = sep_probs.masked_fill(~mask, 0.0)

        center_loss = (pad_mass * (1.0 - mirror_tensor)).sum(dim=-1).mean()

        # Count Loss
        expected_pad_count = pad_mass.sum(dim=-1)
        target_count = (probs_len > 0).float()

        count_loss = ((expected_pad_count - target_count) ** 2).mean()

        # Peak Loss
        max_pad_prob = pad_mass.max(dim=-1).values
        peak_loss = ((max_pad_prob - target_count) ** 2).mean()

        return center_loss, count_loss, peak_loss
