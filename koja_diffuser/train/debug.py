import mlflow
from typing import Protocol, Callable, Union
from contextlib import contextmanager
import torch
from torch import Tensor


def to_float(x):
    if torch.is_tensor(x):
        return float(x.detach().item())
    return float(x)


type AddType = float | Tensor


class LogCall(Protocol):
    def __call__(self, value: Union[AddType, Callable[[], AddType]]) -> None: ...
    def __enter__(self) -> "LogCall": ...
    def __exit__(self, exc_type, exc_val, exc_tb) -> None: ...
    def __getattr__(self, name: str) -> "LogCall": ...


class Debug:
    logs: dict[str, list[AddType | Callable[[], AddType]]] = {}
    _current_prefix = ""
    _is_pre = False

    def __init__(
        self,
        experiment: str,
        run: str,
    ):
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment(experiment)
        if mlflow.active_run() is not None:
            mlflow.end_run()
        mlflow.start_run(run_name=run)

    @contextmanager
    def _scope(self, name: str, is_pre: bool = False):
        old_prefix = self._current_prefix
        old_is_pre = self._is_pre

        new_path = f"{self._current_prefix}.{name}" if self._current_prefix else name
        self._current_prefix = new_path
        self._is_pre = is_pre
        try:
            yield self
        finally:
            self._current_prefix = old_prefix
            self._is_pre = old_is_pre

    def __getattr__(self, name: str) -> LogCall:
        if name == "pre":
            return self._PreProxy(self)
        return self._DynamicLogger(self, name)

    class _PreProxy:
        def __init__(self, outer, path=""):
            self.outer = outer
            self.path = path

        def __getattr__(self, name):
            new_path = f"{self.path}.{name}" if self.path else name
            return Debug._PreProxy(self.outer, new_path)

        def __call__(self, value: Union[float, Callable[[], float]]):
            self.outer.preAdd(self.path, value)

        def __enter__(self):
            self._cm = self.outer._scope(self.path, is_pre=True)
            return self._cm.__enter__()

        def __exit__(self, exc_type, exc_val, exc_tb):
            return self._cm.__exit__(exc_type, exc_val, exc_tb)

    class _DynamicLogger:
        def __init__(self, parent, name):
            self.parent = parent
            self.name = name
            self._cm = None

        def __getattr__(self, name: str) -> "Debug._DynamicLogger":
            new_name = f"{self.name}.{name}"
            return Debug._DynamicLogger(self.parent, new_name)

        def __call__(self, value: Union[float, Callable[[], float]]):
            prefix = self.parent._current_prefix
            full_key = f"{prefix}.{self.name}" if prefix else self.name

            if self.parent._is_pre or callable(value):
                self.parent.preAdd(full_key, value)
            else:
                self.parent.add(full_key, value)

        def __enter__(self):
            self._cm = self.parent._scope(self.name, is_pre=self.parent._is_pre)
            return self._cm.__enter__()

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self._cm:
                return self._cm.__exit__(exc_type, exc_val, exc_tb)

    def add(
        self,
        key: str,
        value: AddType,
    ):
        if key not in self.logs:
            self.logs[key] = []
        self.logs[key].append(to_float(value))

    def preAdd(
        self,
        key: str,
        fn: Callable[[], AddType],
    ):
        if key not in self.logs:
            self.logs[key] = []
        self.logs[key].append(fn)

    def process(self):
        for key in self.logs:
            self.logs[key] = [
                to_float(item() if callable(item) else item) for item in self.logs[key]
            ]

    def commit(self, step: int):
        self.process()

        metrics: dict[str, float] = {}
        for key, values in self.logs.items():
            if not values:
                continue
            total_sum = sum(values)
            metrics[key.replace(".", "/")] = total_sum / len(values)

        if step > 0 and step % 10 == 0:
            mlflow.log_metrics(metrics, step)
        self.logs = {}

    def log_params(self, params: dict):
        mlflow.log_params(params)

    def end(self):
        mlflow.end_run()
