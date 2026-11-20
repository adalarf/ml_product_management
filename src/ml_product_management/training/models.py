from torch import nn

from ml_product_management.config import PipelineError
from ml_product_management.training.config import ModelConfig


def build_model(config: ModelConfig, num_classes: int) -> nn.Sequential:
    if config.type == "linear":
        if config.params:
            raise PipelineError("Линейная модель не принимает model.params")
        return nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, num_classes))
    if config.type != "simple_cnn":
        raise PipelineError(f"Неизвестная модель: {config.type}")
    if set(config.params) - {"channels", "hidden"}:
        raise PipelineError("Неизвестный параметр CNN")
    channels = config.params.get("channels", [32, 64])
    hidden = config.params.get("hidden", 128)
    if (
        not isinstance(channels, list)
        or len(channels) != 2
        or any(not isinstance(value, int) or value <= 0 for value in channels)
        or not isinstance(hidden, int)
        or hidden <= 0
    ):
        raise PipelineError("CNN требует два положительных channels и положительный hidden")
    return nn.Sequential(
        nn.Conv2d(3, channels[0], kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(channels[1] * 8 * 8, hidden),
        nn.ReLU(),
        nn.Linear(hidden, num_classes),
    )
