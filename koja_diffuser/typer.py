import typer
from koja_diffuser.tokenizer.typer import tokenizer_app
from koja_diffuser.train.typer import train_app
from koja_diffuser.runtime.inference import Inference

app = typer.Typer()

app.add_typer(tokenizer_app, name="tokenizer")
app.add_typer(train_app, name="train")

inference = typer.Typer()


@inference.command()
def k2j(name: str):
    inf = Inference()
    res = inf.ko_to_ja([name], [1])
    print(res)


@inference.command()
def j2k(name: str):
    inf = Inference()
    res = inf.ja_to_ko([name], [1])
    print(res)


app.add_typer(inference, name="inference")
