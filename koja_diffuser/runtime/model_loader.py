import torch
from koja_diffuser.util import tensor_to_bytes
from koja_diffuser.model import Encoder, Decoder, DiffusionTranslate
from koja_diffuser.tokenizer.ko import KoreanTokenizer
from koja_diffuser.tokenizer.ja import JapaneseTokenizer
from koja_diffuser.config import Stage2Config, Config, get_config
from koja_diffuser.runtime.sample import sample_logits
from typing import Literal
from torch import Tensor
from pathlib import Path
from huggingface_hub import hf_hub_download
import dataclasses


def load_lang(lang: Literal["ko", "ja"], data, *, device: torch.device):
    config = get_config(lang, data["config"])
    tokenizer = (KoreanTokenizer if lang == "ko" else JapaneseTokenizer)(
        tensor_to_bytes(data["parquet"])
    )
    vocab_size = len(tokenizer)
    encoder = Encoder(
        vocab_size=vocab_size,
        latent_size=config.latent_size,
        max_len=config.max_len,
    ).to(device)
    encoder.load_state_dict(data["encoder"])
    encoder.eval()

    decoder = Decoder(vocab_size=vocab_size, max_len=config.max_len).to(device)
    decoder.load_state_dict(data["decoder"])
    decoder.eval()

    return LangModels(tokenizer, encoder, decoder, device, config)


@dataclasses.dataclass(frozen=True)
class LangModels:
    tokenizer: KoreanTokenizer | JapaneseTokenizer
    encoder: Encoder
    decoder: Decoder
    device: torch.device
    config: Config

    def encode(self, names: list[str], ages: list[int]):
        encoded_ids = [
            self.tokenizer.encode(
                name,
                add_eos=True,
                max_len=self.config.max_len,
            )
            for name in names
        ]

        names = torch.tensor(
            encoded_ids,
            dtype=torch.long,
        ).to(device=self.device, non_blocking=True)

        ages = torch.tensor(ages, dtype=torch.long, device=self.device).clamp(
            min=0, max=9
        )

        return self.encoder(names, ages), encoded_ids

    def decode(
        self,
        names: Tensor,
        *,
        sampling_mode: Literal["greedy", "sample"] = "sample",
        temperature: float = 0.8,
        top_k: int = 20,
        top_p: float = 0.9,
        generator: torch.Generator | None = None,
    ):
        if sampling_mode == "greedy":
            logits = self.decoder(names).argmax(dim=-1)
        else:
            logits = sample_logits(
                self.decoder(names), temperature, top_k, top_p, generator
            )
        decoded: list[list[int]] = logits.cpu().tolist()
        res: list[str] = []
        for ids in decoded:
            res.append(self.tokenizer.decode(ids))
        return res, decoded


@dataclasses.dataclass(frozen=True)
class Models:
    ko: LangModels
    ja: LangModels
    bridge_kj: DiffusionTranslate
    bridge_jk: DiffusionTranslate
    config: Stage2Config


def load_model(*, path="./dist/full.pt", device: torch.device):
    if not Path(path).is_file():
        path = hf_hub_download(
            repo_id="MOKA-AYUMU/KoJaNameDiffuser",
            filename="full.pt",
            revision="main",
        )

    ckpt = torch.load(path, map_location=device)

    ko = load_lang("ko", ckpt["ko"], device=device)
    ja = load_lang("ja", ckpt["ja"], device=device)

    bridge_kj = DiffusionTranslate(
        source_latent_size=ckpt["ko"]["config"]["latent_size"],
        target_latent_size=ckpt["ja"]["config"]["latent_size"],
    ).to(device=device)
    bridge_kj.load_state_dict(ckpt["bridge_kj"])
    bridge_kj.eval()

    bridge_jk = DiffusionTranslate(
        source_latent_size=ckpt["ja"]["config"]["latent_size"],
        target_latent_size=ckpt["ko"]["config"]["latent_size"],
    ).to(device=device)
    bridge_jk.load_state_dict(ckpt["bridge_jk"])
    bridge_jk.eval()

    return Models(ko, ja, bridge_kj, bridge_jk, Stage2Config(**ckpt["config"]))
