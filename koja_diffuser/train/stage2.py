from koja_diffuser.model import Encoder, Decoder, DiffusionTranslate
from koja_diffuser.tokenizer.special import SpecialToken
from koja_diffuser.tokenizer.ko import KoreanTokenizer
from koja_diffuser.tokenizer.ja import JapaneseTokenizer
from koja_diffuser.train.dataset import get_stage2_dataloader
from koja_diffuser.train.debug import Debug, to_float
from koja_diffuser.runtime.schedule import DiffusionSchedule
from koja_diffuser.config import get_config, Stage2Config
from koja_diffuser.train.loss import MMDLoss, repeat_penalty_loss, CenterOneSepLoss
from koja_diffuser.runtime.bridge_utils import bridge_forward
from koja_diffuser.runtime.ddim import ddim_sample_bridge, DdimConfig
import torch
from torch import Tensor
import torch.optim as optim
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup
import tqdm
from typing import Literal, Optional
from dataclasses import asdict
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d = Debug("name-generate", "stage2")


class LangModel:
    def __init__(self, lang: Literal["ko", "ja"] = "ko"):
        ckpt = torch.load(f"./dist/stage1_{lang}/1000.pt", map_location=device)
        config = get_config(lang, ckpt["config"])

        if lang == "ko":
            tokenizer = KoreanTokenizer("./dist/ko_token.parquet")
        else:
            tokenizer = JapaneseTokenizer("./dist/ja_token.parquet")
        vocab_size = len(tokenizer)
        encoder = Encoder(
            vocab_size=vocab_size,
            latent_size=config.latent_size,
            max_len=config.max_len,
        ).to(device)
        encoder.load_state_dict(ckpt["encoder"])
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad_(False)

        decoder = Decoder(vocab_size=vocab_size, max_len=config.max_len).to(device)
        decoder.load_state_dict(ckpt["decoder"])
        decoder.eval()
        for param in decoder.parameters():
            param.requires_grad_(False)

        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.config = config

    def encode_names(self, names: list[str]):
        return torch.tensor(
            [
                self.tokenizer.encode(
                    name,
                    add_eos=True,
                    max_len=self.config.max_len,
                )
                for name in names
            ],
            dtype=torch.long,
        ).to(device=device, non_blocking=True)


class LambdaCycleLatent:
    def __init__(self, config: Stage2Config):
        self.config = config
        self.current_epoch = 0
        self.v = 0.0

    def epoch(self, i: Optional[int] = None):
        if i is not None:
            self.current_epoch = i
        else:
            self.current_epoch += 1

        if self.current_epoch < self.config.cycle_start_epoch:
            self.v = 0.0
            return

        progress = (self.current_epoch - self.config.cycle_start_epoch + 1) / max(
            self.config.cycle_ramp_epochs, 1
        )

        progress = min(max(progress, 0.0), 1.0)
        self.v = self.config.max_lambda_cycle_latent * progress

    def __call__(self) -> float:
        return self.v


class LambdaCycleToken:
    def __init__(self, config: Stage2Config):
        self.config = config
        self.current_epoch = 0
        self.v = (0.0, 0.0)

    def epoch(self, i: Optional[int] = None):
        if i is not None:
            self.current_epoch = i
        else:
            self.current_epoch += 1

        if self.current_epoch < self.config.token_cycle_start_epoch:
            self.v = (0.0, 0.0)
        else:
            self.v = (self.config.token_ko_scale, self.config.token_ja_scale)

    def __call__(self) -> tuple[float, float]:
        return self.v


class StartTimestep:
    def __init__(self, config: Stage2Config):
        self.timesteps = config.cycle_start_timesteps
        self.current_epoch = 0
        self.timesteps_current_index = 0

        assert len(self.timesteps) > 0
        assert self.timesteps[0][0] == 0
        assert self.timesteps[-1][0] <= config.steps

        for (prev_epoch, prev_t), (next_epoch, next_t) in zip(
            self.timesteps,
            self.timesteps[1:],
        ):
            assert prev_epoch < next_epoch
            assert 0 <= prev_t < config.diffusion_timesteps
            assert 0 <= next_t < config.diffusion_timesteps

        self.v = self.timesteps[0][1]

    def epoch(self, i: Optional[int] = None):
        if i is not None:
            self.current_epoch = i
        else:
            self.current_epoch += 1

        while self.timesteps_current_index + 1 < len(self.timesteps):
            next_epoch, next_timestep = self.timesteps[self.timesteps_current_index + 1]

            if self.current_epoch < next_epoch:
                break

            self.timesteps_current_index += 1
            self.v = next_timestep

    def __call__(self) -> int:
        return self.v


def cycle_steps_from_timestep(t: int) -> int:
    return math.ceil(t / 500) + 1


class CycleSteps:
    def __init__(self, config: Stage2Config):
        self.config = config
        self.current_epoch = 0
        self.add = 0

    def epoch(self, i: Optional[int] = None):
        if i is not None:
            self.current_epoch = i
        else:
            self.current_epoch += 1

        self.add = self.current_epoch // 30

    def __call__(self, timestep: int) -> int:
        return min(6, cycle_steps_from_timestep(timestep) + self.add)


class CycleSchedule:
    def __init__(self, config: Stage2Config):
        self._lambda_latent = LambdaCycleLatent(config)
        self._lambda_cycle_token = LambdaCycleToken(config)
        self._start_timestep = StartTimestep(config)
        self._cycle_steps = CycleSteps(config)

    def epoch(self, i: Optional[int] = None):
        # ! debug가 위에 있어야 현재 epoch에 맞는 commit
        d.cycle_schedule.lambda_latent(self.lambda_latent)
        d.cycle_schedule.num_steps(self.num_steps)
        d.cycle_schedule.token_ko_scale(self.token_ko_scale)
        d.cycle_schedule.token_ja_scale(self.token_ja_scale)
        d.cycle_schedule.start_timestep(self.start_timestep)

        self._lambda_latent.epoch(i)
        self._lambda_cycle_token.epoch(i)
        self._start_timestep.epoch(i)
        self._cycle_steps.epoch(i)

    @property
    def lambda_latent(self) -> float:
        return self._lambda_latent()

    @property
    def num_steps(self) -> int:
        return self._cycle_steps(self.start_timestep)

    @property
    def token_ko_scale(self) -> float:
        return self._lambda_cycle_token()[0]

    @property
    def token_ja_scale(self) -> float:
        return self._lambda_cycle_token()[1]

    @property
    def start_timestep(self) -> float:
        return self._start_timestep()


def ddim_sample_cycle(
    *,
    bridge: DiffusionTranslate,
    schedule: DiffusionSchedule,
    guide: Tensor,
    config: Stage2Config,
    cycle_schedule: CycleSchedule,
):
    return ddim_sample_bridge(
        bridge=bridge,
        schedule=schedule,
        guide=guide,
        config=DdimConfig(
            num_steps=cycle_schedule.num_steps,
            start_timestep=cycle_schedule.start_timestep,
            use_checkpoint=config.use_gradient_checkpointing,
        ),
    )


def prior_loss(
    *,
    bridge: DiffusionTranslate,
    schedule: DiffusionSchedule,
    z_unpaired: Tensor,
    z_real: Tensor,
    config: Stage2Config,
):
    batch_size = z_real.size(0)
    device = z_real.device

    t = torch.randint(
        0, schedule.timesteps, (batch_size,), device=device, dtype=torch.long
    )

    noise = torch.randn_like(z_real)
    z_t = schedule.q_sample(z_real, t, noise)

    eps_pred = bridge_forward(
        bridge=bridge,
        x=z_t,
        guide=z_unpaired,
        t=t,
        use_checkpoint=config.use_gradient_checkpointing,
    )

    return F.mse_loss(eps_pred, noise)


def cycle_latent_loss_ko(
    *,
    bridge_kj: DiffusionTranslate,
    bridge_jk: DiffusionTranslate,
    schedule: DiffusionSchedule,
    z_ko: Tensor,
    ko_latent_size: int,
    ja_latent_size: int,
    config: Stage2Config,
    cycle_schedule: CycleSchedule,
):
    # ko -> ja -> ko
    z_ja_hat = ddim_sample_cycle(
        bridge=bridge_kj,
        schedule=schedule,
        guide=z_ko,
        config=config,
        cycle_schedule=cycle_schedule,
    )
    z_ko_rec = ddim_sample_cycle(
        bridge=bridge_jk,
        schedule=schedule,
        guide=z_ja_hat,
        config=config,
        cycle_schedule=cycle_schedule,
    )
    loss_cycle_ko = F.mse_loss(z_ko_rec, z_ko)
    d.loss.cycle_ko(to_float(loss_cycle_ko))

    return {
        "loss": loss_cycle_ko,
        "z_ko_rec": z_ko_rec,
        "z_ja_hat": z_ja_hat,
    }


def cycle_latent_loss_ja(
    *,
    bridge_kj: DiffusionTranslate,
    bridge_jk: DiffusionTranslate,
    schedule: DiffusionSchedule,
    z_ja: Tensor,
    ko_latent_size: int,
    ja_latent_size: int,
    config: Stage2Config,
    cycle_schedule: CycleSchedule,
):
    # ja -> ko -> ja
    z_ko_hat = ddim_sample_cycle(
        bridge=bridge_jk,
        schedule=schedule,
        guide=z_ja,
        config=config,
        cycle_schedule=cycle_schedule,
    )
    z_ja_rec = ddim_sample_cycle(
        bridge=bridge_kj,
        schedule=schedule,
        guide=z_ko_hat,
        config=config,
        cycle_schedule=cycle_schedule,
    )
    loss_cycle_ja = F.mse_loss(z_ja_rec, z_ja)
    d.loss.cycle_ja(to_float(loss_cycle_ja))

    return {
        "loss": loss_cycle_ja,
        "z_ja_rec": z_ja_rec,
        "z_ko_hat": z_ko_hat,
    }


def cycle_token_loss_ko(
    *,
    ko_dec: Decoder,
    z_ko_rec: Tensor,
    ko_name: Tensor,
):
    ko_logits = ko_dec(z_ko_rec)
    loss_token_ko = F.cross_entropy(
        ko_logits.transpose(1, 2),
        ko_name,
        ignore_index=SpecialToken.pad,
    )
    d.loss.cycle_token_ko(to_float(loss_token_ko))
    return loss_token_ko


def cycle_token_loss_ja(
    *,
    ja_dec: Decoder,
    z_ja_rec: Tensor,
    ja_name: Tensor,
):
    ja_logits = ja_dec(z_ja_rec)
    loss_token_ja = F.cross_entropy(
        ja_logits.transpose(1, 2),
        ja_name,
        ignore_index=SpecialToken.pad,
    )
    d.loss.cycle_token_ja(to_float(loss_token_ja))

    return loss_token_ja


class Stage2:
    bridge_kj: DiffusionTranslate
    bridge_jk: DiffusionTranslate
    optimizer: optim.AdamW
    step = 0
    config: Stage2Config
    ko: LangModel
    ja: LangModel

    def __init__(self):
        self.ko = LangModel("ko")
        self.ja = LangModel("ja")
        self.config = Stage2Config()
        self.dataloader = get_stage2_dataloader(500)

        d.log_params(
            {
                **asdict(self.config),
                "dataset_size": len(self.dataloader.dataset),
                "ko_vocab_size": self.ko.vocab_size,
                "ja_vocab_size": self.ja.vocab_size,
            }
        )
        print("ko_vocab_size: ", self.ko.vocab_size)
        print("ja_vocab_size: ", self.ja.vocab_size)
        print("dataset_size: ", len(self.dataloader.dataset))

        self.bridge_kj = DiffusionTranslate(
            source_latent_size=self.ko.config.latent_size,
            target_latent_size=self.ja.config.latent_size,
        ).to(device)
        self.bridge_jk = DiffusionTranslate(
            source_latent_size=self.ja.config.latent_size,
            target_latent_size=self.ko.config.latent_size,
        ).to(device)
        self.bridge_kj.train()
        self.bridge_jk.train()

        self.optimizer = optim.AdamW(
            list(self.bridge_kj.parameters()) + list(self.bridge_jk.parameters()),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        self.diff_schedule = DiffusionSchedule(
            timesteps=self.config.diffusion_timesteps
        ).to(device)

        num_training_steps = len(self.dataloader) * self.config.steps
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup_steps, num_training_steps
        )

        self.cs = CycleSchedule(self.config)

        self.bridge_params = list(self.bridge_kj.parameters()) + list(
            self.bridge_jk.parameters()
        )

    def autocast_context(self):
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=self.config.use_amp and torch.cuda.is_available(),
        )

    def encode_ages(self, ages: Tensor):
        return (
            ages.to(dtype=torch.long)
            .clamp(min=0, max=9)
            .to(device=device, non_blocking=True)
        )

    def get_data_from_batch(
        self,
        batch,
    ):
        ko_name = self.ko.encode_names(batch["ko"]["name"])
        ko_age = self.encode_ages(batch["ko"]["age"])

        ja_name = self.ja.encode_names(batch["ja"]["name"])
        ja_age = self.encode_ages(batch["ja"]["age"])

        return ko_name, ko_age, ja_name, ja_age

    def prior_loss(self, *, z_ko: Tensor, z_ja: Tensor):
        loss_prior_kj = prior_loss(
            bridge=self.bridge_kj,
            schedule=self.diff_schedule,
            z_unpaired=z_ko,
            z_real=z_ja,
            config=self.config,
        )
        loss_prior_jk = prior_loss(
            bridge=self.bridge_jk,
            schedule=self.diff_schedule,
            z_unpaired=z_ja,
            z_real=z_ko,
            config=self.config,
        )
        loss_prior = (
            loss_prior_kj * self.config.prior_kj_scale
            + loss_prior_jk * self.config.prior_jk_scale
        )

        d.loss.prior_kj(to_float(loss_prior_kj))
        d.loss.prior_jk(to_float(loss_prior_jk))
        d.loss.prior(to_float(loss_prior))

        return loss_prior

    def cycle_loss_ko(self, *, ko_name: Tensor, z_ko: Tensor, z_ja: Tensor):
        loss_value = 0.0
        cycle_value = 0.0
        token_value = 0.0

        if self.cs.lambda_latent > 0.0:
            with self.autocast_context():
                cycle = cycle_latent_loss_ko(
                    bridge_kj=self.bridge_kj,
                    bridge_jk=self.bridge_jk,
                    schedule=self.diff_schedule,
                    z_ko=z_ko,
                    ko_latent_size=self.ko.config.latent_size,
                    ja_latent_size=self.ja.config.latent_size,
                    config=self.config,
                    cycle_schedule=self.cs,
                )
                loss = self.cs.lambda_latent * cycle["loss"].float()
                cycle_value = to_float(cycle["loss"])

            # MMD Loss는 exp 사용으로 autocast bf16를 하면 불안정할 수 있음
            loss_domain_ja = MMDLoss.mmd_rbf_loss(cycle["z_ja_hat"], z_ja)
            loss = loss + self.config.domain_kj_scale * loss_domain_ja

            d.loss.domain_ko2ja(loss_domain_ja)
            d.loss.cycle_latent_ko(cycle_value)

            with self.autocast_context():
                logits = self.ja.decoder(cycle["z_ja_hat"])
                loss_repeat_ja = repeat_penalty_loss(logits)
                d.loss.repeat_ja(loss_repeat_ja)
                loss = (
                    loss + self.config.repeat_penalty_ja_scale * loss_repeat_ja.float()
                )

                # center_pad
                center_loss, count_loss, peak_loss = CenterOneSepLoss(logits).loss()
                d.loss.center_ja(center_loss)
                d.loss.count_ja(count_loss)
                d.loss.peak_ja(peak_loss)
                loss = (
                    loss
                    + self.config.sep_center_scale * center_loss
                    + self.config.sep_count_scale * count_loss
                    + self.config.sep_peak_scale * peak_loss
                )

                if self.cs.token_ko_scale > 0.0:
                    loss_token_ko = cycle_token_loss_ko(
                        ko_dec=self.ko.decoder,
                        z_ko_rec=cycle["z_ko_rec"],
                        ko_name=ko_name,
                    )
                    token_value = to_float(loss_token_ko)
                    loss = loss + self.cs.token_ko_scale * loss_token_ko.float()
                    d.loss.token_ko(loss_token_ko)

                loss_value = to_float(loss)
            loss.backward()
            del loss
            del cycle

        return (loss_value, cycle_value, token_value)

    def cycle_loss_ja(self, *, ja_name: Tensor, z_ko: Tensor, z_ja: Tensor):
        loss_value = 0.0
        cycle_value = 0.0
        token_value = 0.0

        if self.cs.lambda_latent > 0.0:
            with self.autocast_context():
                cycle = cycle_latent_loss_ja(
                    bridge_kj=self.bridge_kj,
                    bridge_jk=self.bridge_jk,
                    schedule=self.diff_schedule,
                    z_ja=z_ja,
                    ko_latent_size=self.ko.config.latent_size,
                    ja_latent_size=self.ja.config.latent_size,
                    config=self.config,
                    cycle_schedule=self.cs,
                )
                loss = self.cs.lambda_latent * cycle["loss"].float()
                cycle_value = to_float(cycle["loss"])

            # MMD Loss는 exp 사용으로 autocast bf16를 하면 불안정할 수 있음
            loss_domain_ko = MMDLoss.mmd_rbf_loss(cycle["z_ko_hat"], z_ko)
            loss = loss + self.config.domain_jk_scale * loss_domain_ko

            d.loss.domain_ja2ko(loss_domain_ko)
            d.loss.cycle_latent_ja(cycle_value)

            with self.autocast_context():
                loss_repeat_ko = repeat_penalty_loss(self.ko.decoder(cycle["z_ko_hat"]))
                d.loss.repeat_ko(loss_repeat_ko)
                loss = (
                    loss + self.config.repeat_penalty_ko_scale * loss_repeat_ko.float()
                )

                if self.cs.token_ja_scale > 0.0:
                    loss_token_ja = cycle_token_loss_ja(
                        ja_dec=self.ja.decoder,
                        z_ja_rec=cycle["z_ja_rec"],
                        ja_name=ja_name,
                    )
                    token_value = to_float(loss_token_ja)
                    loss = loss + self.cs.token_ja_scale * loss_token_ja.float()
                    d.loss.token_ja(loss_token_ja)

                loss_value = to_float(loss)
            loss.backward()
            del loss
            del cycle

        return (loss_value, cycle_value, token_value)

    def get_enable_cycle(self, epoch_idx: int, i: int):
        do_ko_cycle = i % 2 == 0
        do_ja_cycle = not do_ko_cycle
        if epoch_idx % 2 == 1:
            do_ko_cycle = not do_ko_cycle
            do_ja_cycle = not do_ja_cycle
        return (do_ko_cycle, do_ja_cycle)

    def train(self):
        pbar_total = tqdm.tqdm(total=self.config.steps, desc="Steps", position=0)

        for step in range(self.config.steps):
            self.step = step
            pbar_batch = tqdm.tqdm(
                total=len(self.dataloader),
                desc=f"Batch {step + 1}",
                position=1,
                leave=False,
            )

            for i, batch in enumerate(self.dataloader):
                ko_name, ko_age, ja_name, ja_age = self.get_data_from_batch(batch)
                assert torch.equal(ko_age, ja_age)

                self.optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    z_ko = self.ko.encoder(ko_name, ko_age)
                    z_ja = self.ja.encoder(ja_name, ja_age)

                total_loss_value = 0.0
                total_cycle_value = 0.0
                total_token_value = 0.0

                with self.autocast_context():
                    loss_prior = self.prior_loss(z_ko=z_ko, z_ja=z_ja)
                total_loss_value += to_float(loss_prior)
                loss_prior.backward()
                del loss_prior

                (do_ko_cycle, do_ja_cycle) = self.get_enable_cycle(step, i)
                (ko_loss, ko_cycle, ko_token) = (
                    self.cycle_loss_ko(z_ko=z_ko, z_ja=z_ja, ko_name=ko_name)
                    if do_ko_cycle
                    else (0.0, 0.0, 0.0)
                )
                (ja_loss, ja_cycle, ja_token) = (
                    self.cycle_loss_ja(z_ko=z_ko, z_ja=z_ja, ja_name=ja_name)
                    if do_ja_cycle
                    else (0.0, 0.0, 0.0)
                )
                total_loss_value += ko_loss + ja_loss
                cycle_scale = 2.0  # because of alternate cycle direction
                total_cycle_value += (ko_cycle + ja_cycle) * cycle_scale
                total_token_value += ko_token + ja_token

                d.loss.total(total_loss_value)
                d.loss.total_cycle(total_cycle_value)
                d.loss.total_token(total_token_value)

                # ----

                current_lr = self.optimizer.param_groups[0]["lr"]
                d.lr(to_float(current_lr))

                if self.config.grad_clip_norm is not None:
                    grad_norm_bridge = torch.nn.utils.clip_grad_norm_(
                        self.bridge_params,
                        max_norm=self.config.grad_clip_norm,
                    )
                    d.norm.grad_bridge(to_float(grad_norm_bridge))

                self.optimizer.step()
                self.scheduler.step()
                pbar_batch.update(1)
            self.cs.epoch()
            d.commit(step)
            pbar_total.update(1)
            if step > 0 and step % 100 == 0:
                self.save()
        self.step = self.config.steps
        self.save()

    def unwrap_compile(self, model: DiffusionTranslate) -> DiffusionTranslate:
        return model._orig_mod if hasattr(model, "_orig_mod") else model

    def save(self):
        bridge_kj = self.unwrap_compile(self.bridge_kj)
        bridge_jk = self.unwrap_compile(self.bridge_jk)

        torch.save(
            {
                "bridge_kj": bridge_kj.state_dict(),
                "bridge_jk": bridge_jk.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": self.step,
                "config": asdict(self.config),
            },
            f"dist/stage2/{self.step}.pt",
        )


def stage2():
    a = Stage2()
    a.train()
