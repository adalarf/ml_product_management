import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Metrics:
    loss: float
    accuracy: float
    macro_f1: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def from_confusion(loss_sum: float, matrix: npt.NDArray[np.int64]) -> Metrics:
    total = int(matrix.sum())
    if total == 0:
        raise ValueError("Нельзя оценить пустую выборку")
    diagonal = np.diag(matrix).astype(np.float64)
    denominators = matrix.sum(axis=0) + matrix.sum(axis=1)
    f1 = np.divide(
        2 * diagonal,
        denominators,
        out=np.zeros_like(diagonal),
        where=denominators != 0,
    )
    return Metrics(loss_sum / total, float(diagonal.sum() / total), float(f1.mean()))


def save_confusion(path: Path, matrix: npt.NDArray[np.int64], classes: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["actual/predicted", *classes])
        for label, row in zip(classes, matrix, strict=True):
            writer.writerow([label, *row.tolist()])
