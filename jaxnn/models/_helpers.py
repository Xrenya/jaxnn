"""Model creation / weight loading / state_dict helpers

Hacked together by / Copyright 2026 Rinat Shaymukhametov
"""

import functools
import logging
from typing import Any, Callable, Dict, Optional, Union, List, Tuple, Type
from pathlib import Path

from flax import nnx
from flax.typing import Dtype, PromoteDtypeFn
from flax.nnx.nn import dtypes as flax_dtypes
import jax.numpy as jnp

from jaxnn.models._builder import load_orbax_state_dict, _apply_flat_state_dict

_logger = logging.getLogger(__name__)


def load_checkpoint(
    model: nnx.Module,
    checkpoint_path: Union[str, Path],
    strict: bool = True,
    remap: Optional[Union[Dict[str, str], List[Tuple[str, str]]]] = None,
    filter_fn: Optional[Callable[[str, Any], bool]] = None,
    verbose: bool = True,
) -> nnx.Module:
    """Load checkpoint weights into a model from a local path.

    Args:
        model: The model to load weights into
        checkpoint_path: Path to checkpoint directory (Orbax format)
        strict: If True, require all model params to be in checkpoint
        remap: Dictionary or list of (old_name, new_name) tuples for remapping parameter names
        filter_fn: Function(param_name, param_value) -> bool to filter parameters
        verbose: Print loading summary

    Returns:
        Model with loaded weights
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict, _ = load_orbax_state_dict(checkpoint_path)

    # Apply key remapping
    if remap:
        remap_dict = dict(remap) if isinstance(remap, list) else remap
        state_dict = {remap_dict.get(k, k): v for k, v in state_dict.items()}

    # Apply filter function
    if filter_fn is not None:
        state_dict = {k: v for k, v in state_dict.items() if filter_fn(k, v)}

    result = _apply_flat_state_dict(model, state_dict, strict=strict)

    if verbose:
        if result.missing_keys:
            _logger.info("Missing keys: %s", ", ".join(result.missing_keys))
        if result.unexpected_keys:
            _logger.warning("Unexpected keys: %s", ", ".join(result.unexpected_keys))

    return model
