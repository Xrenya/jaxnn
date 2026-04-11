from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Any, Deque, Dict, Tuple, Optional, Union


__all__ = ["PretrainedCfg", "filter_pretrained_cfg", "DefaultCfg"]


@dataclass
class PretrainedCfg:
    # Weight source locations
    url: Optional[Union[str, Tuple[str, str]]] = None
    file: Optional[str] = None
    state_dict: Optional[Dict[str, Any]] = None
    hf_hub_id: Optional[str] = None
    hf_hub_filename: Optional[str] = None
    hf_hub_revision: Optional[str] = None

    source: Optional[str] = None
    architecture: Optional[str] = None
    tag: Optional[str] = None
    custom_load: bool = False

    # Input / data config
    rngs: int = 42
    input_size: Tuple[int, int, int] = (224, 224, 3)
    test_input_size: Optional[Tuple[int, int, int]] = None
    min_input_size: Optional[Tuple[int, int, int]] = None
    fixed_input_size: bool = False
    interpolation: str = "bicubic"
    crop_pct: float = 0.875
    test_crop_pct: Optional[float] = None
    crop_mode: str = "center"
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)

    # Head / classifier config and meta-data
    num_classes: int = 1000
    label_offset: Optional[int] = None
    label_names: Optional[Tuple[str]] = None
    label_desctiption: Optional[Dict[str, str]] = None

    # Model attributes
    pool_size: Optional[Tuple[int, ...]] = None
    test_pool_size: Optional[Tuple[int, ...]] = None
    first_conv: Optional[str] = None
    classifier: Optional[str] = None

    license: Optional[str] = None
    description: Optional[str] = None
    origin_url: Optional[str] = None
    paper_name: Optional[str] = None
    paper_ids: Optional[Union[str, Tuple[str]]] = None
    notes: Optional[Tuple[str]] = None

    @property
    def has_weights(self):
        return self.url or self.file or self.hf_hub_id

    def to_dict(
        self,
        remove_source: bool = False,
        remove_null: bool = True,
    ):
        return filter_pretrained_cfg(
            asdict(self),
            remove_source=remove_source,
            remove_null=remove_null,
        )


def filter_pretrained_cfg(
    cfg: Dict[str, Any],
    remove_source=False,
    remove_null=True,
) -> Dict[str, Any]:
    filtered_cfg = {}
    # These keys must NEVER be removed by remove_null, even if None,
    # because downstream code checks for their presence
    keep_null = {
        "pool_size",
        "first_conv",
        "classifier",
        "label_offset",
    }
    # Source fields that carry weight locations - only remove if
    # remove_source is explicitly True
    source_keys = {
        "url",
        "file",
        "state_dict",
        "hf_hub_id",
        "hf_hub_filename",
        "hf_hub_revision",
        "source",
    }
    for k, v in cfg.items():
        if remove_source and k in source_keys:
            continue
        if remove_null and v is None and k not in keep_null:
            continue
        filtered_cfg[k] = v
    return filtered_cfg


@dataclass
class DefaultCfg:
    tags: Deque[str] = field(default_factory=deque)
    cfgs: Dict[str, PretrainedCfg] = field(default_factory=dict)
    is_pretrained: bool = False

    @property
    def default(self):
        return self.cfgs[self.tags[0]]

    @property
    def default_with_tag(self):
        tag = self.tags[0]
        return tag, self.cfgs[tag]
