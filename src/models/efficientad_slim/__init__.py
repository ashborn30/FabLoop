"""Frozen Anomalib baseline and the experimental Slim-0.5 candidate.

See SOURCE.json for the unchanged copied implementation's version, hash, and
fixed 384 / 768 / 384 feature-channel contract. Candidate changes are isolated in
slim_model.py; the existing baseline exports retain their original meaning.
"""

from .slim_model import SlimAutoEncoder, SlimEfficientAdModel, SlimStudent
from .torch_model import EfficientAdModel, EfficientAdModelSize

__all__ = [
    "EfficientAdModel",
    "EfficientAdModelSize",
    "SlimStudent",
    "SlimAutoEncoder",
    "SlimEfficientAdModel",
]
