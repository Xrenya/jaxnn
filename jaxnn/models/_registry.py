""" Model Registry
Hacked together by / Copyright 2020 Ross Wightman
"""

import fnmatch
import re
import sys
import warnings
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import replace
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Sequence, Union, Tuple


__all__ = ["is_model", "get_arch_name", "split_model_name_tag"]

_model_entrypoints: Dict[str, Callable[..., Any]] = {}  # mapping of model names to architecture entrypoint fns

def is_model(model_name: str) -> bool:
    """Verify model existance"""
    arch_name = get_arch_name(model_name=model_name)
    return arch_name in _model_entrypoints

def split_model_name_tag(model_name: str, no_tag: str = '') -> Dict[str, str]:
    model_name, *tag_list = model_name.split(".", 1)
    tag = tag_list[0] if tag_list else no_tag
    return {
        "model_name": model_name,
        "tag": tag,
    }


def get_arch_name(model_name: str) -> str:
    return split_model_name_tag(model_name).get("name", None)