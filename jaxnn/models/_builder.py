import dataclasses
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union
import json

import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp

from jaxnn.models._pretrained import PretrainedCfg
from jaxnn.models._registry import get_pretrained_cfg
from jaxnn.models._hub import load_state_path_from_hf
from jaxnn.models._types import LoadResult
from jaxnn.models._features import FeatureGetterNet

_logger = logging.getLogger(__name__)

# Global variables for rarely used pretrained checkpoint download progress and hash check.
# Use set_pretrained_download_progress / set_pretrained_check_hash functions to toggle.
_DOWNLOAD_PROGRESS = False
_CHECK_HASH = False

__all__ = [
    "set_pretrained_download_progress",
    "set_pretrained_check_hash",
    "load_custom_pretrained",
    "load_pretrained",
    "pretrained_cfg_for_features",
    "resolve_pretrained_cfg",
    "build_model_with_cfg",
]


def set_pretrained_download_progress(enable=True):
    """Set download progress for pretrained weights."""
    global _DOWNLOAD_PROGRESS
    _DOWNLOAD_PROGRESS = enable


def set_pretrained_check_hash(enable=True):
    """Set hash checking for pretrained weights."""
    global _CHECK_HASH
    _CHECK_HASH = enable


ModelT = TypeVar("ModelT", bound=nnx.Module)  # any subclass of nn.Module


def resolve_pretrained_cfg(
    variant: str,
    pretrained_cfg: Optional[Union[str, Dict[str, Any]]] = None,
    pretrained_cfg_overlay: Optional[Dict[str, Any]] = None,
) -> PretrainedCfg:
    """Resolve pretrained configuration from various sources."""
    model_with_tag = variant
    pretrained_tag = None
    if pretrained_cfg:
        if isinstance(pretrained_cfg, dict):
            # pretrained_cfg dict passed as arg, validate by converting to PretrainedCfg
            pretrained_cfg = PretrainedCfg(**pretrained_cfg)
        elif isinstance(pretrained_cfg, str):
            pretrained_tag = pretrained_cfg
            pretrained_cfg = None

    # fallback to looking up pretrained cfg in model registry by variant identifier
    if not pretrained_cfg:
        if pretrained_tag:
            model_with_tag = ".".join([variant, pretrained_tag])
        pretrained_cfg = get_pretrained_cfg(model_with_tag)

    if not pretrained_cfg:
        _logger.warning(
            f"No pretrained configuration specified for {model_with_tag} model. Using a default."
            f" Please add a config to the model pretrained_cfg registry or pass explicitly."
        )
        pretrained_cfg = PretrainedCfg()  # instance with defaults

    pretrained_cfg_overlay = pretrained_cfg_overlay or {}
    if not pretrained_cfg.architecture:
        pretrained_cfg_overlay.setdefault("architecture", variant)
    pretrained_cfg = dataclasses.replace(pretrained_cfg, **pretrained_cfg_overlay)

    return pretrained_cfg


def pretrained_cfg_for_features(pretrained_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt a pretrained cfg for feature extraction (remove classifier info)."""
    pretrained_cfg = dict(pretrained_cfg)
    # remove default pretrained cfg fields that don't have much relevance for feature backbone
    to_remove = ("num_classes", "classifier", "global_pool")
    for tr in to_remove:
        pretrained_cfg.pop(tr, None)
    return pretrained_cfg


def load_custom_pretrained(
    model: nnx.Module,
    pretrained_cfg: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
) -> None:
    """Load pretrained weights for models with custom load functions.

    The model must implement a `load_pretrained` method.
    """
    pretrained_cfg = pretrained_cfg or getattr(model, "pretrained_cfg", None)
    if not pretrained_cfg:
        _logger.warning("No pretrained config found for custom load.")
        return

    source, location = _resolve_pretrained_source(pretrained_cfg)
    if source is None:
        _logger.warning("No pretrained source found for custom load.")
        return

    if source == "hf-hub":
        location = str(
            load_state_path_from_hf(
                location,
                cache_dir=cache_dir,
                revision=pretrained_cfg.get("hf_hub_revision"),
            )
        )
    elif source == "local-dir":
        pass  # location is already the path
    else:
        _logger.warning("Custom load does not support source '%s'.", source)
        return

    if hasattr(model, "load_pretrained"):
        model.load_pretrained(location)
    else:
        _logger.warning("Model does not implement load_pretrained().")


def _filter_kwargs(kwargs: Dict[str, Any], names: List[str]) -> None:
    if not kwargs or not names:
        return
    for n in names:
        kwargs.pop(n, None)


def _update_default_model_kwargs(pretrained_cfg, kwargs, kwargs_filter) -> None:
    """Update the default_cfg and kwargs before passing to model

    Args:
        pretrained_cfg: input pretrained cfg (updated in-place)
        kwargs: keyword args passed to model build fn (updated in-place)
        kwargs_filter: keyword arg keys that must be removed before model __init__
    """
    default_kwarg_names = ("num_classes", "global_pool", "in_chans")
    if pretrained_cfg.get("fixed_input_size", False):
        # if fixed_input_size exists and is True, model takes an img_size arg that fixes its input size
        default_kwarg_names += ("img_size",)

    for n in default_kwarg_names:
        if n == "img_size":
            input_size = pretrained_cfg.get("input_size", None)
            if input_size is not None:
                assert len(input_size) == 3
                kwargs.setdefault(n, input_size[:2])
        elif n == "in_chans":
            input_size = pretrained_cfg.get("input_size", None)
            if input_size is not None:
                assert len(input_size) == 3
                kwargs.setdefault(n, input_size[-1])
        elif n == "num_classes":
            default_val = pretrained_cfg.get(n, None)
            # if default is < 0, don't pass through to model
            if default_val is not None and default_val >= 0:
                kwargs.setdefault(n, pretrained_cfg[n])
        else:
            default_val = pretrained_cfg.get(n, None)
            if default_val is not None:
                kwargs.setdefault(n, pretrained_cfg[n])

    # Filter keyword args for task specific model variants (some 'features only' models, etc.)
    _filter_kwargs(kwargs, names=kwargs_filter)


def _resolve_pretrained_source(cfg: dict) -> Tuple[Optional[str], Any]:
    if cfg.get("state_dict") is not None:
        return "state_dict", cfg["state_dict"]
    for key in ("file", "local_dir", "folder"):
        loc = cfg.get(key)
        if loc and Path(loc).is_dir():
            return "local-dir", str(loc)
    hf_id = cfg.get("hf_hub_id")
    if hf_id:
        return "hf-hub", hf_id
    return None, None


def _get_checkpointer():
    """Return a checkpointer that can restore without a target tree.

    #TODO: Check ocp.Checkpointer option 'partially=True'

    PyTreeCheckpointer is preferred because it does not require a
    target structure for .restore(), making partial loading safe.
    """
    # PyTreeCheckpointer: restores without target (no warning)
    if hasattr(ocp, "PyTreeCheckpointer"):
        return ocp.PyTreeCheckpointer()
    # Newer orbax: Checkpointer + PyTreeCheckpointHandler
    if hasattr(ocp, "Checkpointer") and hasattr(ocp, "PyTreeCheckpointHandler"):
        return ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
    # Last resort (will warn about missing target)
    return ocp.StandardCheckpointer()


def _strip_variable_state(obj):
    """nnx.State -> nested dict of **numpy** arrays.

    This is the critical step: we strip VariableState wrappers so that
    Orbax saves a plain {key: array} tree. Without this, Orbax serialises
    VariableState internals (.value, .type, .raw_value) as sub-keys,
    and restoring without a matching target produces a mismatched tree.
    """
    if isinstance(obj, nnx.Variable):
        return jnp.array(obj.get_value())  # ← just the array
    if isinstance(obj, (dict, nnx.State)):
        return {str(k): _strip_variable_state(v) for k, v in obj.items()}
    if hasattr(obj, "shape"):
        return jnp.array(obj)
    return obj


def _flatten(d: dict, prefix: str = "", sep: str = ".") -> Dict[str, Any]:
    """{'a': {'b': val}} into {'a.b': val}"""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key, sep))
        else:
            out[key] = v
    return out


def _normalize_flat_dict(flat: Dict[str, Any]) -> Dict[str, Any]:
    """Backward-compat: fix legacy checkpoints that saved VariableState.
    #TODO: remove since new version will be use 'Variable'
    Legacy checkpoints have keys like:
        bn1.bias.value - the actual array
        bn1.bias.type - VariableState metadata (discard)

    We collapse 'x.value' - 'x' and drop '.type' / '.raw_value'.
    """
    has_value_suffix = any(k.endswith(".value") for k in flat)
    if not has_value_suffix:
        return flat

    _logger.info("Detected legacy VariableState checkpoint - normalising keys")
    cleaned: Dict[str, Any] = {}
    for k, v in flat.items():
        if k.endswith(".type") or k.endswith(".raw_value"):
            continue  # metadata - skip
        if k.endswith(".value"):
            cleaned[k[: -len(".value")]] = v  # strip suffix
        else:
            cleaned[k] = v
    return cleaned


def _iter_state_leaves(state, prefix: str = ""):
    """Yield (dotted_key, VariableState) for every leaf in an NNX State."""
    for k, v in state.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, nnx.Variable):
            yield key, v
        elif isinstance(v, (dict, nnx.State)):
            yield from _iter_state_leaves(v, key)


def save_orbax_checkpoint(
    model: nnx.Module,
    checkpoint_dir: Union[str, Path],
    config: Optional[Dict[str, Any]] = None,
    state_subfolder: str = "state",
) -> None:
    """Save an NNX model as an Orbax checkpoint of **pure arrays**.
    #TODO: review pure vs ocp.checkpointer/manager
    Directory layout::

        checkpoint_dir/
            config.json
            state/        orbax pytree (one file per leaf)
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # config.json
    cfg = config or getattr(model, "pretrained_cfg", None) or {}
    if cfg:
        with open(checkpoint_dir / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)

    # state (pure arrays)
    state = nnx.state(model)
    pure_tree = _strip_variable_state(state)

    ckptr = _get_checkpointer()
    ckptr.save(str(checkpoint_dir / state_subfolder), pure_tree)
    _logger.info("Saved checkpoint to %s", checkpoint_dir)


def load_orbax_state_dict(
    checkpoint_dir: Union[str, Path],
    state_subfolder: str = "state",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load an Orbax checkpoint ``(flat_state_dict, config)``.

    Handles both new (pure-array) and legacy (VariableState) formats.
    #TODO: remove legacy
    """
    checkpoint_dir = Path(checkpoint_dir)

    # Config
    config: Dict[str, Any] = {}
    cfg_path = checkpoint_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            config = json.load(f)

    # State
    state_path = checkpoint_dir / state_subfolder
    restore_path = state_path if state_path.exists() else checkpoint_dir

    ckptr = _get_checkpointer()
    restored = ckptr.restore(str(restore_path))

    flat = _flatten(restored) if isinstance(restored, dict) else {}
    flat = _normalize_flat_dict(flat)

    return flat, config


def adapt_input_conv(
    in_chans: int,
    weight: jnp.ndarray,
    channel_axis: int = 2,  # Flax shape[Height Width Input Output]
) -> jnp.ndarray:
    orig = weight.shape[channel_axis]
    if in_chans == orig:
        return weight
    if in_chans == 1:
        return jnp.mean(weight, axis=channel_axis, keepdims=True)
    if orig == 1:
        return jnp.repeat(weight, in_chans, axis=channel_axis) / in_chans
    if in_chans < orig:
        sl = [slice(None)] * weight.ndim
        sl[channel_axis] = slice(0, in_chans)
        return weight[tuple(sl)]
    reps = (in_chans + orig - 1) // orig
    tiled = jnp.concatenate([weight] * reps, axis=channel_axis)
    sl = [slice(None)] * weight.ndim
    sl[channel_axis] = slice(0, in_chans)
    return tiled[tuple(sl)] * (orig / in_chans)


def _apply_flat_state_dict(
    model: nnx.Module,
    flat_dict: Dict[str, Any],
    strict: bool = True,
) -> LoadResult:
    """Merge a flat ``{dotted_key: array}`` dict into an NNX model."""
    state = nnx.state(model)
    leaves = dict(_iter_state_leaves(state))

    model_keys = set(leaves.keys())
    loaded_keys = set(flat_dict)

    missing = sorted(model_keys - loaded_keys)
    unexpected = sorted(loaded_keys - model_keys)

    if strict and (missing or unexpected):
        msgs: List[str] = []
        if missing:
            msgs.append(f"Missing keys: {missing}")
        if unexpected:
            msgs.append(f"Unexpected keys: {unexpected}")
        raise RuntimeError("Error(s) in strict load:\n\t" + "\n\t".join(msgs))

    shape_mismatches: List[str] = []
    for key in sorted(model_keys & loaded_keys):
        var_state = leaves[key]
        arr = jnp.asarray(flat_dict[key])

        if var_state.get_value().shape != arr.shape:
            _logger.warning(
                "Shape mismatch for '%s': model %s vs loaded %s - skipping",
                key,
                var_state.get_value().shape,
                arr.shape,
            )
            shape_mismatches.append(key)
            continue

        var_state.value = arr

    nnx.update(model, state)

    return LoadResult(
        missing_keys=missing + shape_mismatches,
        unexpected_keys=unexpected,
    )


def debug_key_mismatch(
    model: nnx.Module,
    checkpoint_dir: Union[str, Path],
) -> None:
    """Print model keys vs checkpoint keys for debugging."""
    state = nnx.state(model)
    model_keys = {k for k, _ in _iter_state_leaves(state)}
    ckpt_keys_flat, _ = load_orbax_state_dict(checkpoint_dir)
    ckpt_keys = set(ckpt_keys_flat.keys())

    print("=== Keys only in MODEL (would be 'missing') ===")
    for k in sorted(model_keys - ckpt_keys):
        print(f"  {k}")
    print("\n=== Keys only in CHECKPOINT (would be 'unexpected') ===")
    for k in sorted(ckpt_keys - model_keys):
        print(f"  {k}")
    print("\n=== Matched keys ===")
    print(f"  {len(model_keys & ckpt_keys)} / {len(model_keys)} model keys")


def load_pretrained(
    model: nnx.Module,
    pretrained_cfg: Optional[Dict[str, Any]] = None,
    num_classes: int = 1000,
    in_chans: int = 3,
    filter_fn: Optional[Callable] = None,
    strict: bool = True,
    cache_dir: Optional[Union[str, Path]] = None,
) -> None:
    """Load pretrained weights into a Flax NNX model.

    Config keys (pretrained_cfg):
        state_dict  -> ready-made {dotted_key: array} dict
        local_dir   -> path to Orbax checkpoint folder
        hf_hub_id   -> Hugging Face Hub repo id
    """
    pretrained_cfg = pretrained_cfg or getattr(model, "pretrained_cfg", None)
    if not pretrained_cfg:
        raise RuntimeError(
            "Invalid pretrained config. Use pretrained=False for random init."
        )

    source, location = _resolve_pretrained_source(pretrained_cfg)
    if source is None:
        arch = pretrained_cfg.get("architecture", "this model")
        raise RuntimeError(f"No pretrained weights for {arch}.")

    # 1. Obtain flat state dict
    if source == "state_dict":
        _logger.info("Loading pretrained weights from state dict")
        raw = location
        if isinstance(raw, dict) and any(isinstance(v, dict) for v in raw.values()):
            state_dict = _flatten(raw)
        else:
            state_dict = dict(raw)
        state_dict = _normalize_flat_dict(state_dict)

    elif source == "local-dir":
        _logger.info("Loading pretrained weights from %s", location)
        if pretrained_cfg.get("custom_load", False):
            model.load_pretrained(location)
            return
        state_dict, _ = load_orbax_state_dict(location)

    elif source == "hf-hub":
        _logger.info("Loading pretrained weights from HF Hub (%s)", location)
        local = load_state_path_from_hf(
            location,
            cache_dir=cache_dir,
            revision=pretrained_cfg.get("hf_hub_revision"),
        )
        if pretrained_cfg.get("custom_load", False):
            model.load_pretrained(str(local))
            return
        state_dict, _ = load_orbax_state_dict(local)

    else:
        raise RuntimeError(f"Unknown source: {source}")

    # 2. Optional filter
    if filter_fn is not None:
        try:
            state_dict = filter_fn(state_dict, model)
        except TypeError:
            state_dict = filter_fn(state_dict)

    # 3. Adapt input channels
    input_convs = pretrained_cfg.get("first_conv")
    if input_convs and in_chans != 3:
        if isinstance(input_convs, str):
            input_convs = (input_convs,)
        for name in input_convs:
            for suffix in (".kernel", ".weight"):
                wk = name + suffix
                if wk in state_dict:
                    try:
                        state_dict[wk] = adapt_input_conv(in_chans, state_dict[wk])
                        _logger.info("Adapted '%s' from 3->%d channels", name, in_chans)
                    except NotImplementedError:
                        del state_dict[wk]
                        strict = False
                    break

    # 4. Adapt classifier head
    classifiers = pretrained_cfg.get("classifier")
    if classifiers:
        if isinstance(classifiers, str):
            classifiers = (classifiers,)
        pt_classes = pretrained_cfg.get("num_classes", 1000)
        label_offset = pretrained_cfg.get("label_offset") or 0
        if num_classes != pt_classes:
            for c in classifiers:
                for s in (".kernel", ".bias", ".weight", ".scale"):
                    state_dict.pop(c + s, None)
            strict = False
        elif label_offset > 0:
            for c in classifiers:
                for s in (".kernel", ".weight"):
                    k = c + s
                    if k in state_dict:
                        state_dict[k] = state_dict[k][:, label_offset:]
                bk = c + ".bias"
                if bk in state_dict:
                    state_dict[bk] = state_dict[bk][label_offset:]

    # 5. Apply to model
    result = _apply_flat_state_dict(model, state_dict, strict=strict)

    if result.missing_keys:
        _logger.info("Missing keys: %s", ", ".join(result.missing_keys))
    if result.unexpected_keys:
        _logger.warning("Unexpected keys: %s", ", ".join(result.unexpected_keys))


def build_model_with_cfg(
    model_cls: Union[Type[ModelT], Callable[..., ModelT]],
    variant: str,
    pretrained: bool,
    pretrained_cfg: Optional[Dict] = None,
    pretrained_cfg_overlay: Optional[Dict] = None,
    model_cfg: Optional[Any] = None,
    feature_cfg: Optional[Dict] = None,
    pretrained_strict: bool = True,
    pretrained_filter_fn: Optional[Callable] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    kwargs_filter: Optional[Tuple[str]] = None,
    **kwargs,
) -> ModelT:
    """Build model with specified default_cfg and optional model_cfg

    This helper fn aids in the construction of a model including:
      * handling default_cfg and associated pretrained weight loading
      * passing through optional model_cfg for models with config based arch spec
      * features_only model adaptation
      * pruning config / model adaptation

    Args:
        model_cls: Model class
        variant: Model variant name
        pretrained: Load the pretrained weights
        pretrained_cfg: Model's pretrained weight/task config
        pretrained_cfg_overlay: Entries that will override those in pretrained_cfg
        model_cfg: Model's architecture config
        feature_cfg: Feature extraction adapter config
        pretrained_strict: Load pretrained weights strictly
        pretrained_filter_fn: Filter callable for pretrained weights
        cache_dir: Override model cache dir for Hugging Face Hub and Torch checkpoints
        kwargs_filter: Kwargs keys to filter (remove) before passing to model
        **kwargs: Model args passed through to model __init__
    """
    pruned = kwargs.pop("pruned", False)
    features = False
    feature_cfg = feature_cfg or {}

    # resolve and update model pretrained config and model kwargs
    pretrained_cfg = resolve_pretrained_cfg(
        variant,
        pretrained_cfg=pretrained_cfg,
        pretrained_cfg_overlay=pretrained_cfg_overlay,
    )
    pretrained_cfg = pretrained_cfg.to_dict()

    seed = pretrained_cfg.get("rngs", 0)
    rngs = dict(rngs=nnx.Rngs(seed))
    kwargs.update(rngs)

    _update_default_model_kwargs(pretrained_cfg, kwargs, kwargs_filter)

    # Setup for feature extraction wrapper done at end of this fn
    if kwargs.pop("features_only", False):
        features = True
        feature_cfg.setdefault("out_indices", (0, 1, 2, 3, 4))
        if "out_indices" in kwargs:
            feature_cfg["out_indices"] = kwargs.pop("out_indices")
        if "feature_cls" in kwargs:
            feature_cfg["feature_cls"] = kwargs.pop("feature_cls")

    # Instantiate the model
    if model_cfg is None:
        model = model_cls(**kwargs)
    else:
        model = model_cls(cfg=model_cfg, **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg  # alias for backwards compat

    if pruned:
        _logger.warning("Pruned model loading is not yet implemented for Flax/JAX")

    # For classification models, check class attr, then kwargs, then default to 1k, otherwise 0 for feats
    num_classes_pretrained = (
        0
        if features
        else getattr(model, "num_classes", kwargs.get("num_classes", 1000))
    )
    if pretrained:
        load_pretrained(
            model,
            pretrained_cfg=pretrained_cfg,
            num_classes=num_classes_pretrained,
            in_chans=kwargs.get("in_chans", 3),
            filter_fn=pretrained_filter_fn,
            strict=pretrained_strict,
            cache_dir=cache_dir,
        )

    # Wrap the model in a feature extraction module if enabled
    if features:
        feature_cls = FeatureGetterNet
        if "feature_cls" in feature_cfg:
            feature_cls_name = feature_cfg.pop("feature_cls")
            if isinstance(feature_cls_name, str):
                feature_cls_name = feature_cls_name.lower()
                if feature_cls_name in ("getter", "list", "dict"):
                    feature_cls = FeatureGetterNet
                else:
                    assert False, f"Unknown feature class {feature_cls_name}"

        model = feature_cls(model, **feature_cfg)
        model.pretrained_cfg = pretrained_cfg_for_features(pretrained_cfg)
        model.default_cfg = model.pretrained_cfg

    return model
