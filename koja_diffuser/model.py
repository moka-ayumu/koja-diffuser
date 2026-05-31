import torch
from torch import nn, Tensor
from jaxtyping import Float, Int
from einops import repeat
from einops.layers.torch import Rearrange
from koja_diffuser.tokenizer.special import SpecialToken
import math

MODEL_SIZE = 128
GENERATION_COUNT = 10

default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MoudleDevice(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("_device_ref", torch.empty(0), persistent=False)

    @property
    def device(self) -> torch.device:
        return self._device_ref.device


class Encoder(MoudleDevice):
    def __init__(self, *, vocab_size: int, latent_size: int, max_len: int):
        super().__init__()

        self.layer = nn.TransformerEncoderLayer(
            d_model=MODEL_SIZE,
            nhead=8,
            dim_feedforward=MODEL_SIZE * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(self.layer, num_layers=6)
        self.token_emb = nn.Embedding(vocab_size, MODEL_SIZE)
        self.pos_emb = nn.Embedding(
            max_len, MODEL_SIZE
        )  # todo POSITION == "fixed" 일때만 사용
        self.age_emb = nn.Embedding(GENERATION_COUNT, MODEL_SIZE)

        # Latent pooling
        self.latent_query = nn.Parameter(
            torch.randn(latent_size, MODEL_SIZE, device=self.device)
        )
        self.latent_attention = nn.MultiheadAttention(
            embed_dim=MODEL_SIZE, num_heads=8, dropout=0.1, batch_first=True
        )

    def forward(
        self,
        ids: Int[Tensor, "BATCH_SIZE LEN"],
        age: Int[Tensor, "BATCH_SIZE"],
    ) -> Float[Tensor, "BATCH_SIZE LATENT_LEN MODEL_SIZE"]:
        batch_size = ids.size(0)
        id_size = ids.size(-1)

        embeded_token: Float[Tensor, "BATCH_SIZE LEN MODEL_SIZE"] = self.token_emb(ids)
        embeded_age: Float[Tensor, "BATCH_SIZE MODEL_SIZE"] = self.age_emb(age)
        embeded_age: Float[Tensor, "BATCH_SIZE LEN MODEL_SIZE"] = repeat(
            embeded_age, "bs ms -> bs len ms", len=id_size
        )

        pos_ids: Float[Tensor, "BATCH_SIZE LEN"] = repeat(
            torch.arange(id_size, device=self.device), "s -> b s", b=batch_size
        )

        embeded_pos: Float[Tensor, "BATCH_SIZE LEN MODEL_SIZE"] = self.pos_emb(pos_ids)
        sum_embeded: Float[Tensor, "BATCH_SIZE LEN MODEL_SIZE"] = (
            embeded_token + embeded_age + embeded_pos
        )

        padding_mask = ids == SpecialToken.pad
        encoded: Float[Tensor, "BATCH_SIZE LEN MODEL_SIZE"] = self.encoder(
            sum_embeded, src_key_padding_mask=padding_mask
        )

        latent_query: Float[Tensor, "BATCH_SIZE LATENT_LEN MODEL_SIZE"] = repeat(
            self.latent_query, "latent size -> batch latent size", batch=batch_size
        )
        out, _ = self.latent_attention(
            latent_query, encoded, encoded, key_padding_mask=padding_mask
        )

        return out

    def __call__(
        self,
        ids: Int[Tensor, "BATCH_SIZE LEN"],
        age: Int[Tensor, "BATCH_SIZE"],
    ) -> Float[Tensor, "BATCH_SIZE LATENT_LEN MODEL_SIZE"]:
        return super().__call__(ids, age)


class Decoder(MoudleDevice):
    def __init__(self, *, vocab_size: int, max_len: int):
        super().__init__()

        self.layer = nn.TransformerDecoderLayer(
            d_model=MODEL_SIZE, nhead=8, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(self.layer, num_layers=6)
        self.query = nn.Parameter(torch.randn(max_len, MODEL_SIZE, device=self.device))
        self.out = nn.Linear(MODEL_SIZE, vocab_size)

    def forward(self, n: Float[Tensor, "BATCH_SIZE LATENT_LEN MODEL_SIZE"]):
        batch_size = n.size(0)
        query: Float[Tensor, "BATCH_SIZE MAX_LEN MODEL_SIZE"] = repeat(
            self.query, "len size -> batch len size", batch=batch_size
        )

        decoded: Float[Tensor, "BATCH_SIZE MAX_LEN MODEL_SIZE"] = self.decoder(query, n)

        logits: Float[Tensor, "BATCH_SIZE MAX_LEN VOCAB_SIZE"] = self.out(decoded)

        return logits

    def __call__(
        self, n: Float[Tensor, "BATCH_SIZE LATENT_LEN MODEL_SIZE"]
    ) -> Float[Tensor, "BATCH_SIZE MAX_LEN VOCAB_SIZE"]:
        return super().__call__(n)


class DiffusionTranslate(nn.Module):
    def __init__(self, *, source_latent_size: int, target_latent_size: int):
        super().__init__()

        self.source_latent_size = source_latent_size
        self.target_latent_size = target_latent_size

        self.noise_proj = nn.Sequential(
            nn.LayerNorm(MODEL_SIZE), nn.Linear(MODEL_SIZE, MODEL_SIZE)
        )
        self.guide_proj = nn.Sequential(
            nn.LayerNorm(MODEL_SIZE), nn.Linear(MODEL_SIZE, MODEL_SIZE)
        )

        self.source_pos_emb = nn.Embedding(source_latent_size, MODEL_SIZE)
        self.target_pos_emb = nn.Embedding(target_latent_size, MODEL_SIZE)

        self.time_mlp = nn.Sequential(
            nn.Linear(MODEL_SIZE, MODEL_SIZE * 4),
            nn.SiLU(),
            nn.Linear(MODEL_SIZE * 4, MODEL_SIZE * 2),
            Rearrange("b d -> b 1 d"),
        )

        self.layer = nn.TransformerDecoderLayer(
            d_model=MODEL_SIZE,
            nhead=8,
            dim_feedforward=MODEL_SIZE * 4,
            batch_first=True,
            norm_first=True,  # ? 학습 안정성을 위해 Pre-Norm
            dropout=0.0,
        )

        # tgt: 노이즈 latent, memory: 가이드 latent(encoded)
        self.transformer = nn.TransformerDecoder(self.layer, num_layers=6)

        self.out_norm = nn.LayerNorm(MODEL_SIZE)
        self.output_proj = nn.Linear(MODEL_SIZE, MODEL_SIZE)

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def pos_encoding(
        self, timestep: Float[Tensor, "BATCH_SIZE 1"], dim: int
    ) -> Float[Tensor, "BATCH_SIZE DIM"]:
        timestep = timestep.float()

        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb_tensor = torch.exp(torch.arange(half_dim, device=timestep.device) * -emb)

        scaled_t: Float[Tensor, "BATCH_SIZE {half_dim}"] = (
            timestep * emb_tensor[None, :]
        )

        return torch.cat((scaled_t.sin(), scaled_t.cos()), dim=-1)

    def encode_guide(
        self,
        guide: Float[Tensor, "BATCH_SIZE SRC_LATENT_LEN MODEL_SIZE"],
    ) -> Float[Tensor, "BATCH_SIZE SRC_LATENT_LEN MODEL_SIZE"]:
        _, src_len, _ = guide.shape

        if src_len != self.source_latent_size:
            raise ValueError(
                f"Expected source latent size {self.source_latent_size}, got {src_len}"
            )

        src_pos_ids = torch.arange(src_len, device=guide.device)
        src_pos = self.source_pos_emb(src_pos_ids)[None, :, :]
        return self.guide_proj(guide) + src_pos

    def forward(
        self,
        noise: Float[Tensor, "BATCH_SIZE TARGET_LATENT_LEN MODEL_SIZE"],
        guide: Float[Tensor, "BATCH_SIZE SRC_LATENT_LEN MODEL_SIZE"],
        timestep: Float[Tensor, "BATCH_SIZE 1"],
        guide_encoded: Float[Tensor, "BATCH_SIZE SRC_LATENT_LEN MODEL_SIZE"]
        | None = None,
    ) -> Float[Tensor, "BATCH_SIZE TARGET_LATENT_LEN MODEL_SIZE"]:
        _, tgt_len, _ = noise.shape

        if tgt_len != self.target_latent_size:
            raise ValueError(
                f"Expected target latent size {self.target_latent_size}, got {tgt_len}"
            )

        if guide_encoded is None:
            guide_encoded = self.encode_guide(guide)

        tgt_pos_ids = torch.arange(tgt_len, device=noise.device)
        tgt_pos: Float[Tensor, "1 TGT_LATENT MODEL_SIZE"] = self.target_pos_emb(
            tgt_pos_ids
        )[None, :, :]

        noise: Float[Tensor, "BATCH_SIZE TARGET_LATENT_LEN MODEL_SIZE"] = (
            self.noise_proj(noise) + tgt_pos
        )

        time_emb: Float[Tensor, "BATCH_SIZE 1 TWO_MODEL_SIZE"] = self.time_mlp(
            self.pos_encoding(timestep, MODEL_SIZE)
        )
        time_scale, time_shift = time_emb.chunk(2, dim=-1)

        noise = noise * (1 + time_scale) + time_shift

        hidden: Float[Tensor, "BATCH_SIZE TARGET_LATENT_LEN MODEL_SIZE"] = (
            self.transformer(noise, guide_encoded)
        )

        out = self.out_norm(hidden)
        out = self.output_proj(out)

        return out

    def __call__(
        self,
        noise: Float[Tensor, "BATCH_SIZE TARGET_LATENT_LEN MODEL_SIZE"],
        guide: Float[Tensor, "BATCH_SIZE SRC_LATENT_LEN MODEL_SIZE"],
        timestep: Float[Tensor, "BATCH_SIZE 1"],
        guide_encoded: Float[Tensor, "BATCH_SIZE SRC_LATENT_LEN MODEL_SIZE"]
        | None = None,
    ) -> Float[Tensor, "BATCH_SIZE TARGET_LATENT_LEN MODEL_SIZE"]:
        return super().__call__(noise, guide, timestep, guide_encoded)
