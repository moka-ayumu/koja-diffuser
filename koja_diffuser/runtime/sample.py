import torch
from torch import Tensor
from einops import rearrange
from transformers import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


def sample_logits(
    logits: Tensor,
    temperature: float = 0.8,
    top_k: int = 20,
    top_p: float = 0.9,
    generator: torch.Generator | None = None,
) -> Tensor:
    """
    logits: (B, T, V)
    return: (B, T)
    """
    B, T, _ = logits.shape

    flat_logits = rearrange(logits, "b t v -> (b t) v")

    processors = LogitsProcessorList(
        [
            TemperatureLogitsWarper(temperature=temperature),
            TopKLogitsWarper(top_k=top_k),
            TopPLogitsWarper(top_p=top_p),
        ]
    )

    dummy_input_ids = torch.zeros(
        B * T,
        1,
        dtype=torch.long,
        device=logits.device,
    )

    flat_logits = processors(dummy_input_ids, flat_logits)

    probs = torch.softmax(flat_logits, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1, generator=generator)

    return rearrange(sampled, "(b t) 1 -> b t", b=B, t=T)
