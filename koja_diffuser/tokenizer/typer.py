import typer
from koja_diffuser.tokenizer.ko import KoreanTokenizer
from koja_diffuser.tokenizer.ja import JapaneseTokenizer
import lancedb
from typing import Literal

tokenizer_app = typer.Typer()


@tokenizer_app.command()
def build(lang: Literal["ja", "ko"]):
    if lang == "ja":
        tokenizer = JapaneseTokenizer()
        db = lancedb.connect("./koja_diffuser/data/generated")
        table = db.open_table("data")
        text_list: list[str] = (
            table.search().select(["hiragana"]).to_pandas()["hiragana"].tolist()
        )
        tokenizer.train(text_list, "./dist/ja_token.parquet")
    else:
        tokenizer = KoreanTokenizer()
        table = db.open_table("data_korea")
        text_list: list[str] = (
            table.search().select(["name"]).to_pandas()["name"].tolist()
        )
        tokenizer.train(text_list, "./dist/ko_token.parquet")


@tokenizer_app.command()
def length():
    tokenizer = JapaneseTokenizer("./dist/ja_token.parquet")
    print("ja len: ", len(tokenizer))
    tokenizer = KoreanTokenizer("./dist/ko_token.parquet")
    print("ko len: ", len(tokenizer))
