import torch
from koja_diffuser.util import file_to_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ko_parquet = file_to_tensor("./dist/ko_token.parquet")
ja_parquet = file_to_tensor("./dist/ja_token.parquet")

ko_ckpt = torch.load("./dist/stage1_ko/1000.pt", map_location=device)
ja_ckpt = torch.load("./dist/stage1_ja/1000.pt", map_location=device)
stage2_ckpt = torch.load("./dist/stage2/300.pt", map_location=device)


torch.save(
    {
        "bridge_kj": stage2_ckpt["bridge_kj"],
        "bridge_jk": stage2_ckpt["bridge_jk"],
        "config": stage2_ckpt["config"],
        "ko": {
            "parquet": ko_parquet,
            "encoder": ko_ckpt["encoder"],
            "decoder": ko_ckpt["decoder"],
            "config": ko_ckpt["config"],
        },
        "ja": {
            "parquet": ja_parquet,
            "encoder": ja_ckpt["encoder"],
            "decoder": ja_ckpt["decoder"],
            "config": ja_ckpt["config"],
        },
    },
    "dist/full.pt",
)
