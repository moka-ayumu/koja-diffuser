from koja_diffuser.runtime.model_loader import load_model
from koja_diffuser.runtime.schedule import DiffusionSchedule
from koja_diffuser.runtime.ddim import ddim_sample_bridge, DdimConfig
import torch
from koja_diffuser.util import Emitter, noop_emitter, tensor_norm
from typing import Literal


class Inference:
    def __init__(
        self,
        *,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        self.model = load_model(device=device)
        self.schedule = DiffusionSchedule(
            timesteps=self.model.config.diffusion_timesteps
        ).to(device=device)
        self.device = device

    def resolve_seed(self, seed: int | None) -> int:
        if seed is None or seed < 0:
            return int(
                torch.randint(
                    low=0,
                    high=2**31 - 1,
                    size=(1,),
                    device="cpu",
                ).item()
            )

        return int(seed)

    def make_generator(self, seed: int | None) -> tuple[torch.Generator, int]:
        used_seed = self.resolve_seed(seed)

        generator = torch.Generator(device=self.device)
        generator.manual_seed(used_seed)

        return generator, used_seed

    @torch.inference_mode()
    async def ko_to_ja(
        self,
        names: list[str],
        ages: list[int],
        *,
        seed: int | None = None,
        num_steps: int = 6,
        start_timestep: int = 500,
        sampling_mode: Literal["greedy", "sample"] = "sample",
        temperature: float = 0.8,
        top_k: int = 20,
        top_p: float = 0.9,
        emit: Emitter = noop_emitter,
    ):
        generator, seed = self.make_generator(seed)
        await emit("info", {"type": "ko_to_ja", "seed": seed})

        guide, names_ids = self.model.ko.encode(names, ages)
        await emit(
            "encoded",
            {"guide": tensor_norm(guide), "names_ids": tensor_norm(names_ids)},
        )

        z_ja_hat = await ddim_sample_bridge(
            bridge=self.model.bridge_kj,
            generator=generator,
            schedule=self.schedule,
            guide=guide,
            config=DdimConfig(start_timestep, num_steps),
            emit=emit,
        )

        decoded, decoded_ids = self.model.ja.decode(
            z_ja_hat,
            sampling_mode=sampling_mode,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=self.make_generator(seed)[0],
        )

        await emit(
            "decoded", {"result": decoded, "names_ids": tensor_norm(decoded_ids)}
        )

        return decoded

    @torch.inference_mode()
    async def ja_to_ko(
        self,
        names: list[str],
        ages: list[int],
        *,
        seed: int | None = None,
        num_steps: int = 6,
        start_timestep: int = 500,
        sampling_mode: Literal["greedy", "sample"] = "sample",
        temperature: float = 0.8,
        top_k: int = 20,
        top_p: float = 0.9,
        emit: Emitter = noop_emitter,
    ):
        generator, seed = self.make_generator(seed)
        await emit("info", {"type": "ja_to_ko", "seed": seed})

        guide, names_ids = self.model.ja.encode(names, ages)
        await emit(
            "encoded",
            {"guide": tensor_norm(guide), "names_ids": tensor_norm(names_ids)},
        )

        z_ko_hat = await ddim_sample_bridge(
            bridge=self.model.bridge_jk,
            generator=generator,
            schedule=self.schedule,
            guide=guide,
            config=DdimConfig(start_timestep, num_steps),
            emit=emit,
        )

        decoded, decoded_ids = self.model.ko.decode(
            z_ko_hat,
            sampling_mode=sampling_mode,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=self.make_generator(seed)[0],
        )

        await emit(
            "decoded", {"result": decoded, "names_ids": tensor_norm(decoded_ids)}
        )

        return decoded
