import re
import pandas as pd
from typing import Optional
from koja_diffuser.tokenizer.special import SpecialToken
from io import BytesIO


class JapaneseTokenizer:
    series: pd.Series
    token_pattern: re.Pattern[str]

    def __init__(self, filepath: Optional[str | BytesIO] = None):
        self.token_pattern = re.compile(
            r".[ぁぃぅぇぉゃゅょっゎァィゥェォャュョッヮヵヶ]?"
        )
        if filepath is None:
            self.series = pd.Series()
        else:
            self.series = pd.read_parquet(filepath).squeeze()

    def train(self, frame: list[str], output="./dist/ja_token.parquet"):
        tokens_dict: dict[str, int] = {}
        next_id = SpecialToken.next_id
        for text in frame:
            tokens = self.tokenize(text)
            for token in tokens:
                if token not in tokens_dict:
                    tokens_dict[token] = next_id
                    next_id += 1
        df = pd.Series(tokens_dict).to_frame()
        df.to_parquet(output)

    def tokenize(self, text: str) -> list[str]:
        return self.token_pattern.findall(text)

    def id_to_token(self, i: int) -> str:
        if i == SpecialToken.sep:
            return " "
        result = self.series[self.series == i].index
        key = result[0] if not result.empty else ""
        return key

    def encode(self, text: str, *, add_eos=False, max_len=0) -> list[int]:
        tokens = self.tokenize(text)
        encoded = []

        for token in tokens:
            if token == " ":
                token_id = SpecialToken.sep
            else:
                token_id = int(self.series.get(token, SpecialToken.unk))
            encoded.append(token_id)

        if add_eos:
            if max_len > 0 and len(encoded) >= max_len:
                encoded = encoded[: max_len - 1]
            encoded.append(SpecialToken.eos)

        if max_len > 0:
            if len(encoded) > max_len:
                encoded = encoded[:max_len]
                raise ("impossible")
            else:
                pad_len = max_len - len(encoded)
                encoded.extend([SpecialToken.pad] * pad_len)

        return encoded

    def decode(self, ids: list[int]) -> str:
        tokens = []
        for t in ids:
            if t == SpecialToken.eos:
                break
            tokens.append(t)
        return "".join([self.id_to_token(i) for i in tokens])

    def __len__(self):
        return self.series.max() + 1
