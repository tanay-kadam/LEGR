"""Isolated, non-destructive model research extensions for LEGR.

Nothing in this package mutates the campaign datasets or the legacy model
implementations.  Existing encoders are imported and composed as backbones.
"""

from .config import ExperimentConfig, LossConfig, ModelConfig, TrainConfig
from .losses import CompositeRetrievalLoss

__all__ = [
    "CompositeRetrievalLoss",
    "ExperimentConfig",
    "LEGRResearchModel",
    "LossConfig",
    "ModelConfig",
    "TrainConfig",
]


def __getattr__(name):
    if name == "LEGRResearchModel":
        from .model import LEGRResearchModel
        return LEGRResearchModel
    raise AttributeError(name)
