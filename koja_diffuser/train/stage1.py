from koja_diffuser.model import Encoder, Decoder
from koja_diffuser.tokenizer.ko import KoreanTokenizer
from koja_diffuser.tokenizer.ja import JapaneseTokenizer
from koja_diffuser.tokenizer.special import SpecialToken
from koja_diffuser.train.dataset import get_dataloader
from koja_diffuser.train.debug import Debug, to_float
from koja_diffuser.config import get_config, Config
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup
import tqdm
from typing import Literal
from dataclasses import asdict


def apply_noise(ids: Tensor, *, mask_prob=0.2):
    noisy_ids = ids.clone().detach()
    mask = torch.rand(ids.shape, device=ids.device) < mask_prob
    mask &= (
        (ids != SpecialToken.pad)
        & (ids != SpecialToken.eos)
        & (ids != SpecialToken.sep)
    )
    noisy_ids[mask] = SpecialToken.mask
    return noisy_ids


def eos_margin_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eos_id: int,
    pad_id: int,
    margin: float = 1.0,
) -> torch.Tensor:
    B, T, V = logits.shape

    eos_mask = targets == eos_id
    assert eos_mask.any(dim=1).all()

    true_eos_pos = (targets == eos_id).float().argmax(dim=1)  # [B]

    batch_idx = torch.arange(B, device=logits.device)

    eos_slot_logits = logits[batch_idx, true_eos_pos, :]  # [B, V]

    min_value = torch.finfo(eos_slot_logits.dtype).min

    eos_logit = eos_slot_logits[:, eos_id]  # [B]

    non_eos_logits = eos_slot_logits.clone()
    non_eos_logits[:, eos_id] = min_value
    non_eos_logits[:, pad_id] = min_value

    max_non_eos_logit = non_eos_logits.max(dim=-1).values  # [B]

    loss = F.relu(margin - eos_logit + max_non_eos_logit)

    return loss.mean()


def eos_pad_weighted_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eos_id: int,
    pad_id: int,
    sep_id: int,
    eos_weight: float = 2.0,
    pad_weight: float = 0.05,
    sep_weight: float = 0.1,
    eos_pos_loss_weight: float = 0.1,
    eos_margin_loss_weight: float = 0.05,
    margin: float = 1.0,
) -> torch.Tensor:
    B, T, V = logits.shape

    loss = F.cross_entropy(
        logits.reshape(-1, V),
        targets.reshape(-1),
        reduction="none",
    ).view(B, T)

    weights = torch.ones_like(loss)

    weights[targets == eos_id] = eos_weight
    weights[targets == pad_id] = pad_weight
    weights[targets == sep_id] = sep_weight

    token_loss = (loss * weights).sum() / weights.sum().clamp_min(1.0)

    # 2. EOS position loss
    eos_mask = targets == eos_id  # [B, T]

    true_eos_pos = eos_mask.float().argmax(dim=1)  # [B]

    eos_pos_logits = logits[:, :, eos_id]  # [B, T]

    eos_pos_loss = F.cross_entropy(
        eos_pos_logits,
        true_eos_pos,
    )

    margin_loss = eos_margin_loss(logits, targets, eos_id, pad_id, margin)

    d.loss.token(to_float(token_loss))
    d.loss.eos_pos(to_float(eos_pos_loss))
    d.loss.margin_loss(to_float(margin_loss))

    return (
        token_loss
        + eos_pos_loss_weight * eos_pos_loss
        + margin_loss * eos_margin_loss_weight
    )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lang: Literal["ko", "ja"] = "ko"

d = Debug("name-generate", f"stage1_{lang}")
print("lang: ", lang)
print("device: ", device)


def save(
    encoder: Encoder,
    decoder: Decoder,
    optimizer: optim.AdamW,
    step: int,
    config: Config,
):
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": asdict(config),
        },
        f"dist/stage1_{lang}/{step}.pt",
    )


def stage1():
    config = get_config(lang)
    dataloader = get_dataloader(lang, config.max_len, 1500)

    if lang == "ko":
        tokenizer = KoreanTokenizer("./dist/ko_token.parquet")
    else:
        tokenizer = JapaneseTokenizer("./dist/ja_token.parquet")
    vocab_size = len(tokenizer)

    d.log_params(
        {
            **asdict(config),
            "dataset_size": len(dataloader.dataset),
            "vocab_size": vocab_size,
        }
    )
    print("vocab_size: ", vocab_size)
    print("dataset_size: ", len(dataloader.dataset))
    encoder = Encoder(
        vocab_size=vocab_size, latent_size=config.latent_size, max_len=config.max_len
    ).to(device)
    decoder = Decoder(vocab_size=vocab_size, max_len=config.max_len).to(device)

    encoder.train()
    decoder.train()

    optimizer = optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()), lr=config.lr
    )

    num_training_steps = len(dataloader) * config.steps
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )

    criterion = nn.CrossEntropyLoss(ignore_index=SpecialToken.pad).to(device)

    pbar_total = tqdm.tqdm(total=config.steps, desc="Steps", position=0)

    for step in range(config.steps):
        pbar_batch = tqdm.tqdm(
            total=len(dataloader), desc=f"Batch {step + 1}", position=1, leave=False
        )
        for batch in dataloader:
            name = torch.tensor(
                [
                    tokenizer.encode(name, add_eos=True, max_len=config.max_len)
                    for name in batch["name"]
                ],
                device=device,
            )
            age: Tensor = (
                batch["age"]
                .detach()
                .clone()
                .to(device=device, dtype=torch.int)
                .clamp(min=0, max=9)
            )

            noisy_name = apply_noise(name, mask_prob=config.mask_prob)
            optimizer.zero_grad()

            encoded = encoder(noisy_name, age)
            logits = decoder(encoded)

            loss = criterion(logits.view(-1, vocab_size), name.view(-1))
            current_lr = optimizer.param_groups[0]["lr"]

            d.loss.total(to_float(loss))
            d.lr(to_float(current_lr))

            loss.backward()

            grad_norm_enc = torch.nn.utils.clip_grad_norm_(
                encoder.parameters(), max_norm=1.0
            )
            grad_norm_dec = torch.nn.utils.clip_grad_norm_(
                decoder.parameters(), max_norm=1.0
            )
            d.norm.grad_enc(to_float(grad_norm_enc))
            d.norm.grad_dec(to_float(grad_norm_dec))

            optimizer.step()
            scheduler.step()
            pbar_batch.update(1)
        d.commit(step)
        pbar_total.update(1)
        if step > 0 and step % 100 == 0:
            save(encoder, decoder, optimizer, step, config)
    save(encoder, decoder, optimizer, config.steps, config)
    d.end()
