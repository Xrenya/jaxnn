from dataclasses import dataclass, field
from typing import List
from enum import Enum


@dataclass
class LoadResult:
    missing_keys: List[str] = field(default_factory=list)
    unexpected_keys: List[str] = field(default_factory=list)


@dataclass
class FeatureInfo:
    """Metadata for one feature stage"""

    index: int
    name: str
    num_chs: int
    reduction: int  # total stride / spatial reduction factor

    def __repr__(self):
        return (
            f"FeatureInfo(index={self.index}, name='{self.name}', "
            f"num_chs={self.num_chs}, reduction={self.reduction})"
        )


class CheckpointFormat(Enum):
    """Supported checkpoint formats"""

    ORBAX = "orbax"
    NUMPY = "numpy"
    PICKLE = "pickle"
    SAFETENSORS = "safetensors"
    AUTO = "auto"
