import torch
from torch import Tensor
from koja_diffuser.model import DiffusionTranslate
from torch.utils.checkpoint import checkpoint


def bridge_forward(
    *,
    bridge: DiffusionTranslate,
    x: Tensor,
    guide: Tensor,
    t: Tensor,
    guide_encoded: Tensor | None = None,
    use_checkpoint: bool,
) -> Tensor:
    t_embed = t.float().unsqueeze(-1)

    if use_checkpoint and torch.is_grad_enabled():
        return checkpoint(
            lambda x_, guide_, t_, guide_encoded_: bridge(
                x_, guide_, t_, guide_encoded_
            ),
            x,
            guide,
            t_embed,
            guide_encoded,
            use_reentrant=False,
            preserve_rng_state=False,  # bridge dropout 사용시 True, 미사용시 False
        )

    return bridge(x, guide, t_embed, guide_encoded)
