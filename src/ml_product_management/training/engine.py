import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from ml_product_management.training.config import TrainConfig
from ml_product_management.training.metrics import Metrics, from_confusion

Loader = DataLoader[tuple[Tensor, int]]


@dataclass(frozen=True)
class Evaluation:
    metrics: Metrics
    confusion: npt.NDArray[np.int64]


@dataclass(frozen=True)
class FitResult:
    best_epoch: int
    val: Evaluation
    test: Evaluation
    training_seconds: float


def seed_everything(seed: int, cpu_threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cpu_threads)
    torch.use_deterministic_algorithms(True)


def train_epoch(model: nn.Module, loader: Loader, optimizer: torch.optim.Optimizer) -> None:
    model.train()
    for images, labels in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = cast(Tensor, model(images))
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()  # type: ignore[no-untyped-call]  # PyTorch omits this return annotation.
        optimizer.step()


@torch.no_grad()
def evaluate(model: nn.Module, loader: Loader, num_classes: int) -> Evaluation:
    model.eval()
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    loss_sum = 0.0
    for images, labels in loader:
        logits = cast(Tensor, model(images))
        loss_sum += float(nn.functional.cross_entropy(logits, labels, reduction="sum").item())
        predictions = logits.argmax(dim=1)
        confusion += torch.bincount(
            labels * num_classes + predictions, minlength=num_classes**2
        ).reshape(num_classes, num_classes)
    matrix: npt.NDArray[np.int64] = confusion.numpy()
    return Evaluation(from_confusion(loss_sum, matrix), matrix)


def fit(
    model: nn.Module,
    train_loader: Loader,
    val_loader: Loader,
    test_loader: Loader,
    config: TrainConfig,
    checkpoint: Path,
    metadata: dict[str, Any],
    on_epoch: Callable[[int, Metrics], None],
) -> FitResult:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    best_loss = float("inf")
    best_epoch = 0
    start = time.perf_counter()
    num_classes = len(metadata["classes"])
    for epoch in range(1, config.epochs + 1):
        train_epoch(model, train_loader, optimizer)
        val = evaluate(model, val_loader, num_classes)
        if not np.isfinite(val.metrics.loss):
            raise ValueError("val loss не является конечным числом")
        if val.metrics.loss < best_loss:
            best_loss = val.metrics.loss
            best_epoch = epoch
            torch.save(
                {**metadata, "state_dict": model.state_dict(), "best_epoch": epoch}, checkpoint
            )
        on_epoch(epoch, val.metrics)
    duration = time.perf_counter() - start
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["state_dict"])
    return FitResult(
        best_epoch=best_epoch,
        val=evaluate(model, val_loader, num_classes),
        test=evaluate(model, test_loader, num_classes),
        training_seconds=duration,
    )
