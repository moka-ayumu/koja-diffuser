from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from koja_diffuser.runtime.inference import Inference
import asyncio
from contextlib import suppress
from typing import Any, Awaitable, AsyncGenerator, Literal
from pydantic import BaseModel, Field
import json


app = FastAPI()
inference = Inference(device="cpu")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def make_sse(data: Any, event: str | None = None) -> str:
    payload = json.dumps(data, ensure_ascii=False)

    lines = []

    if event is not None:
        lines.append(f"event: {event}")

    for line in payload.splitlines():
        lines.append(f"data: {line}")

    return "\n".join(lines) + "\n\n"


class Emitter:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def emit(self, event: str, data: Any) -> None:
        await self.queue.put(make_sse(data, event=event))

    async def error(self, data: str):
        await self.emit(
            "error",
            {
                "message": str(data),
            },
        )

    async def _run(self, awaitable: Awaitable[Any]) -> None:
        try:
            await awaitable

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            await self.error(exc)

        finally:
            await self.queue.put(None)

    async def stream(
        self,
        awaitable: Awaitable[Any],
        *,
        request=None,
    ) -> AsyncGenerator[str, None]:
        task = asyncio.create_task(self._run(awaitable))

        try:
            while True:
                if request is not None and await request.is_disconnected():
                    task.cancel()
                    break

                try:
                    item = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=0.5,
                    )
                except asyncio.TimeoutError:
                    continue

                if item is None:
                    break

                yield item

        finally:
            task.cancel()

            with suppress(asyncio.CancelledError):
                await task


async def generate():
    for i in range(5):
        yield f"data: chunk {i}\n\n"
        await asyncio.sleep(1)


class NameRequest(BaseModel):
    name: str = Field(min_length=1)
    age: int = Field(ge=0, le=9, strict=True)
    seed: int | None = Field(default=None, strict=True)
    sampling_mode: Literal["greedy", "sample"] = Field(default="sample")
    temperature: float = Field(default=0.8, ge=0.1, le=2.0, strict=True)
    top_k: int = Field(default=20, ge=1, le=100, strict=True)
    top_p: float = Field(default=0.9, ge=0.1, le=1.0, strict=True)


def detect_script(text: str) -> str:
    chars = [ch for ch in text if not ch.isspace() and ch.isalpha()]

    if not chars:
        return "unknown"

    if all(
        "\uac00" <= ch <= "\ud7a3"
        or "\u1100" <= ch <= "\u11ff"
        or "\u3130" <= ch <= "\u318f"
        for ch in chars
    ):
        return "hangul"

    if all("\u3040" <= ch <= "\u309f" for ch in chars):
        return "hiragana"

    return "mixed_or_other"


@app.post("/stream")
async def stream(request: Request, body: NameRequest):
    emitter = Emitter()

    generate = (
        inference.ja_to_ko
        if detect_script(body.name) == "hiragana"
        else inference.ko_to_ja
    )

    return StreamingResponse(
        emitter.stream(
            generate(
                [body.name],
                [body.age],
                seed=body.seed,
                sampling_mode=body.sampling_mode,
                temperature=body.temperature,
                top_k=body.top_k,
                top_p=body.top_p,
                emit=emitter.emit,
            ),
            request=request,
        ),
        media_type="text/event-stream",
    )
