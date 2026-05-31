import dataclasses


@dataclasses.dataclass
class SpecialToken:
    unk = 0
    pad = 1
    sep = 2
    eos = 3
    mask = 4
    # next start
    next_id = 10
