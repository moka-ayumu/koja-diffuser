import numpy as np
import torch
from io import BytesIO
from typing import Any, Callable, Awaitable


def file_to_tensor(path: str) -> torch.Tensor:
    return torch.from_numpy(np.fromfile(path, dtype=np.uint8).copy())


def tensor_to_bytes(tensor: torch.Tensor):
    return BytesIO(tensor.cpu().numpy().tobytes())


type Emitter = Callable[[str, Any], Awaitable[None]]


async def noop_emitter(name: str, value: Any) -> None:
    pass


def round_nested(value, precision):
    if isinstance(value, list):
        return [round_nested(v, precision) for v in value]
    return round(float(value), precision)


def tensor_norm(t, precision=3):
    # Tensor 또는 list 모두 허용
    if isinstance(t, torch.Tensor):
        x = t.detach().cpu().float()
    elif isinstance(t, list):
        x = torch.tensor(t, dtype=torch.float32)
    else:
        raise TypeError()

    x = torch.nan_to_num(x)

    x_min = x.min()
    x_max = x.max()

    normalized = (x - x_min) / (x_max - x_min + 1e-8)

    return round_nested(normalized.tolist(), precision)
