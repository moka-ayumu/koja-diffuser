import dataclasses
from typing import Literal


@dataclasses.dataclass(frozen=True)
class KoreanConfig:
    lr: float = 3e-4
    steps: int = 1000
    warmup_ratio: float = 0.15
    latent_size: int = 16
    mask_prob: float = 0.2
    max_len: int = 10


@dataclasses.dataclass(frozen=True)
class JapaneseConfig:
    lr: float = 3e-4
    steps: int = 1000
    warmup_ratio: float = 0.15  # 0.15
    latent_size: int = 64  # 24
    mask_prob: float = 0.2
    max_len: int = 20


type Config = KoreanConfig | JapaneseConfig


def get_config(lang: Literal["ko", "ja"], config_dict: dict = {}):
    if lang == "ko":
        return KoreanConfig(**config_dict)
    else:
        return JapaneseConfig(**config_dict)


@dataclasses.dataclass(frozen=True)
class Stage2Config:
    lr: float = 5e-4
    steps: int = 300
    weight_decay: float = 1e-2
    warmup_ratio: float = 0.1
    grad_clip_norm: float | None = 1.0

    diffusion_timesteps: int = 1000

    # cycle latent loss
    cycle_start_epoch: int = 5
    cycle_ramp_epochs: int = 10
    max_lambda_cycle_latent: float = 0.1

    # token cycle loss
    token_cycle_start_epoch: int = 20
    token_ko_scale: float = 0.04
    token_ja_scale: float = 0.1

    # direct domain MMD loss
    domain_kj_scale: float = 0.08
    domain_jk_scale: float = 0.09

    # prior
    prior_kj_scale: float = 1.0
    prior_jk_scale: float = 1.0

    # Cylce Start
    cycle_start_timesteps: list[tuple[int, int]] = dataclasses.field(
        default_factory=lambda: [
            (0, 500),
        ]
    )

    repeat_penalty_ja_scale: float = 0.05
    repeat_penalty_ko_scale: float = 0.02

    sep_center_scale: float = 0.1
    sep_count_scale: float = 0.1
    sep_peak_scale: float = 0.05

    use_gradient_checkpointing: bool = False
    use_amp: bool = True
