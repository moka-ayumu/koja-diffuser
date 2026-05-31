from koja_diffuser.model import DiffusionTranslate
from koja_diffuser.runtime.schedule import DiffusionSchedule
from koja_diffuser.runtime.bridge_utils import bridge_forward
import torch
from torch import Tensor
import dataclasses
from koja_diffuser.util import Emitter, noop_emitter, tensor_norm


@dataclasses.dataclass(frozen=True)
class DdimConfig:
    start_timestep: int = 999
    num_steps: int = 6
    use_checkpoint: bool = False
    return_trace: bool = False


def ddim_step(
    *,
    bridge: DiffusionTranslate,
    schedule: DiffusionSchedule,
    x: Tensor,
    guide: Tensor,
    t: Tensor,
    t_next: Tensor | None,
    guide_encoded: Tensor | None = None,
    use_checkpoint=False,
):
    eps_pred = bridge_forward(
        bridge=bridge,
        x=x,
        guide=guide,
        t=t,
        guide_encoded=guide_encoded,
        use_checkpoint=use_checkpoint,
    )

    x0_pred = schedule.predict_x0_from_eps(
        x,
        t,
        eps_pred,
    )

    # 마지막 step이면 x0를 직접 반환
    if t_next is None:
        return x0_pred

    alpha_bar_next = schedule.extract(
        schedule.alpha_bars,
        t_next,
        x,
    )

    x_next = alpha_bar_next.sqrt() * x0_pred + (1.0 - alpha_bar_next).sqrt() * eps_pred

    return x_next


async def ddim_sample_bridge(
    *,
    bridge: DiffusionTranslate,
    schedule: DiffusionSchedule,
    guide: Tensor,
    config: DdimConfig,
    generator: torch.Generator | None = None,
    emit: Emitter = noop_emitter,
):
    batch_size = guide.size(0)
    model_size = guide.size(-1)
    device = guide.device
    target_latent_size = bridge.target_latent_size
    steps = config.num_steps

    x = torch.randn(
        batch_size, target_latent_size, model_size, device=device, generator=generator
    )

    grad_step_ids = torch.linspace(
        config.start_timestep,
        0,
        steps=config.num_steps,
        device=device,
    ).long()

    guide_encoded = bridge.encode_guide(guide)

    await emit(
        "ddim.init_noise",
        {
            "step": -1,
            "guide_encoded": tensor_norm(guide_encoded),
            "x": tensor_norm(x),
        },
    )

    for i in range(steps):
        t = grad_step_ids[i].expand(batch_size)

        if i == steps - 1:
            t_next = None
        else:
            t_next = grad_step_ids[i + 1].expand(batch_size)

        x = ddim_step(
            bridge=bridge,
            schedule=schedule,
            x=x,
            guide=guide,
            t=t,
            t_next=t_next,
            guide_encoded=guide_encoded,
            use_checkpoint=config.use_checkpoint,
        )

        if not (i == steps - 1 and grad_step_ids[i].item() == 0):
            await emit(
                "ddim.step",
                {
                    "step": i,
                    "x": tensor_norm(x),
                },
            )

    return x
