import hashlib
import json
import logging
import os
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple, Union

from flax import nnx

from jaxnn import __version__
from jaxnn.models._pretrained import filter_pretrained_cfg

try:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    hf_hub_download = partial(
        hf_hub_download, library_name="jaxnn", library_version=__version__
    )
    _has_hf_hub = True
except ImportError:
    hf_hub_download = None
    _has_hf_hub = False

_logger = logging.getLogger(__name__)

__all__ = [
    "get_cache_dir",
    "has_hf_hub",
    "hf_split",
    "load_model_config_from_hf",
    "load_model_config_from_path",
    "load_state_path_from_hf",
    "save_for_hf",
    "push_to_hf_hub",
]

HF_WEIGHTS_NAME = "model.msgpack"
HF_SAFE_WEIGHTS_NAME = "model.safetensors"
HF_CONFIG_NAME = "config.json"


def get_cache_dir(child_dir: str = ""):
    """JAX-friendly cache dir."""
    cache_root = os.path.expanduser("~/.cache/jaxnn")
    if child_dir:
        cache_root = os.path.join(cache_root, child_dir)
    os.makedirs(cache_root, exist_ok=True)
    return cache_root


def has_hf_hub(necessary: bool = False):
    if not _has_hf_hub and necessary:
        raise RuntimeError(
            "Hugging Face hub model specified but package not installed."
            "Run `pip install huggingface_hub`."
        )
    return _has_hf_hub


def hf_split(hf_id: str):
    rev_split = hf_id.split("#")
    assert 0 < len(rev_split) <= 2, (
        "hf_hub id should only contain one # character to identify revision."
    )
    hf_model_id = rev_split[0]
    hf_revision = rev_split[1] if len(rev_split) > 1 else None
    return hf_model_id, hf_revision


def download_from_hf(
    model_id: str,
    filename: str,
    cache_dir: Optional[Union[None, Path]] = None,
):
    hf_model_id, hf_revision = hf_split(model_id)
    return hf_hub_download(
        hf_model_id,
        filename,
        revision=hf_revision,
        cache_dir=cache_dir,
    )


def load_cfg_from_json(json_file: Union[str, Path]):
    with open(json_file, "r", encoding="utf-8") as f:
        config_data = f.read()
    return json.loads(config_data)


def _parse_model_cfg(
    cfg: Dict[str, Any], extra_fields: Dict[str, Any]
) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    """Parse a config.json dict into (pretrained_cfg, architecture_name, model_args).

    Returns a 3-tuple so callers can pass model_args to build_model_with_cfg.
    Both load_model_config_from_hf and load_model_config_from_path use this,
    so _factory.py must always unpack all three values.
    """
    pretrained_cfg = cfg["pretrained_cfg"]
    pretrained_cfg.update(extra_fields)

    if "num_classes" in cfg:
        pretrained_cfg["num_classes"] = cfg["num_classes"]
    if "label_names" in cfg:
        pretrained_cfg["label_names"] = cfg["label_names"]
    if "label_descriptions" in cfg:
        pretrained_cfg["label_descriptions"] = cfg["label_descriptions"]

    model_args = cfg.get("model_args", {})
    model_name = cfg["architecture"]
    return pretrained_cfg, model_name, model_args


def load_model_config_from_hf(
    model_id: str,
    cache_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    """Load config.json from a HF Hub repo.

    Returns:
        (pretrained_cfg, architecture_name, model_args)
    """
    assert has_hf_hub(True)
    cfg_path = download_from_hf(model_id, HF_CONFIG_NAME, cache_dir=cache_dir)
    cfg = load_cfg_from_json(cfg_path)
    return _parse_model_cfg(cfg, {"hf_hub_id": model_id, "source": "hf-hub"})


def load_model_config_from_path(
    model_path: Union[str, Path],
) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    """Load config.json from a local JaxNN checkpoint directory.

    The directory is expected to contain:
        config.json   — preprocessing metadata + architecture name
        state/        — Orbax checkpoint produced by the converter

    Returns:
        (pretrained_cfg, architecture_name, model_args)

    The returned pretrained_cfg contains ``local_dir`` pointing to the
    directory so that ``_resolve_pretrained_source`` in ``_builder.py``
    can locate the Orbax state when loading weights.  This mirrors how
    ``load_model_config_from_hf`` sets ``hf_hub_id`` on the returned cfg.
    """
    model_path = Path(model_path)
    cfg_file = model_path / HF_CONFIG_NAME
    if not cfg_file.is_file():
        raise FileNotFoundError(
            f"config.json not found at {cfg_file}. "
            "Make sure the path points to a directory produced by the JaxNN "
            "converter (it should contain config.json and state/)."
        )
    cfg = load_cfg_from_json(cfg_file)
    extra_fields = {"local_dir": str(model_path), "source": "local-dir"}
    return _parse_model_cfg(cfg, extra_fields=extra_fields)


def verify_hash(file_path: str, expected_hash: str) -> bool:
    with open(file_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    return file_hash.startswith(expected_hash)


def load_state_path_from_hf(
    model_id: str,
    cache_dir: Optional[Union[str, Path]] = None,
    revision: Optional[str] = None,
    ignore_patterns: Optional[Union[None, List[str]]] = ["*.md", ".gitattributes"],
) -> Path:
    """Download model from HF Hub, return local snapshot directory."""
    assert has_hf_hub(True)
    hf_model_id, hf_revision = hf_split(model_id)

    effective_revision = revision or hf_revision

    local_dir = snapshot_download(
        hf_model_id,
        repo_type="model",
        revision=effective_revision,
        cache_dir=cache_dir,
        ignore_patterns=ignore_patterns,
    )

    return Path(local_dir)


def save_config_for_hf(
    model: nnx.Module,
    config_path: str,
    model_config: Optional[dict] = None,
    model_args: Optional[dict] = None,
):
    model_config = model_config or {}
    hf_config = {}

    pretrained_cfg = getattr(model, "pretrained_cfg", {})

    pretrained_cfg = filter_pretrained_cfg(
        pretrained_cfg, remove_source=True, remove_null=True
    )
    hf_config["architecture"] = pretrained_cfg.pop("architecture")
    hf_config["num_classes"] = model_config.pop(
        "num_classes", getattr(model, "num_classes", None)
    )
    if hf_config["num_classes"] is None:
        raise ValueError("num_classes must be defined in model or model_config.")

    hf_config["num_features"] = model_config.pop(
        "num_features", getattr(model, "num_features", None)
    )
    global_pool_type = model_config.pop(
        "global_pool", getattr(model, "global_pool", None)
    )
    if isinstance(global_pool_type, str) and global_pool_type:
        hf_config["global_pool"] = global_pool_type

    label_names = model_config.pop("label_names", None)
    if label_names:
        assert isinstance(label_names, (dict, list, tuple))
        hf_config["label_names"] = label_names

    label_descriptions = model_config.pop("label_descriptions", None)
    if label_descriptions:
        assert isinstance(label_descriptions, dict)
        hf_config["label_descriptions"] = label_descriptions

    if model_args:
        hf_config["model_args"] = model_args
    if pretrained_cfg:
        hf_config["pretrained_cfg"] = pretrained_cfg
    hf_config.update(model_config)

    with config_path.open("w") as f:
        json.dump(hf_config, f, indent=2)


def save_for_hf(
    model: nnx.Module,
    save_directory: str,
    model_config: Optional[dict] = None,
    model_args: Optional[dict] = None,
):
    save_directory = Path(save_directory)

    graphdef, state = nnx.split(model)

    from jaxnn.models._builder import _get_checkpointer, _strip_variable_state

    pure_tree = _strip_variable_state(state)
    ckptr = _get_checkpointer()
    ckptr.save(str(save_directory / "state"), pure_tree)

    config_path = save_directory / HF_CONFIG_NAME
    save_config_for_hf(model, config_path, model_config, model_args)


def push_to_hf_hub(
    model: nnx.Module,
    repo_id: str,
    commit_message: str = "Add model",
    token: Optional[str] = None,
    revision: Optional[str] = None,
    private: bool = False,
    create_pr: bool = False,
    model_config: Optional[dict] = None,
    model_card: Optional[dict] = None,
    model_args: Optional[dict] = None,
    task_name: str = "image-classification",
):
    api = HfApi(token=token, library_name="jaxnn", library_version=__version__)

    repo_url = api.create_repo(repo_id, private=private, exist_ok=True)
    repo_id = repo_url.repo_id

    has_readme = api.file_exists(
        repo_id=repo_id, filename="README.md", revision=revision
    )

    with TemporaryDirectory() as tmpdir:
        save_for_hf(
            model,
            tmpdir,
            model_config=model_config,
            model_args=model_args,
        )

        if not has_readme:
            model_card = model_card or {}
            model_name = repo_id.split("/")[-1]
            readme_path = Path(tmpdir) / "README.md"
            readme_text = generate_readme(model_card, model_name, task_name=task_name)
            readme_path.write_text(readme_text)

        return api.upload_folder(
            repo_id=repo_id,
            folder_path=tmpdir,
            revision=revision,
            create_pr=create_pr,
            commit_message=commit_message,
        )


def generate_readme(
    model_card: dict,
    model_name: str,
    task_name: str = "image-classification",
):
    tags = model_card.get("tags", None) or [task_name, "jaxnn", "transformers"]
    readme_text = "---\n"
    if tags:
        readme_text += "tags:\n"
        for t in tags:
            readme_text += f"- {t}\n"
    readme_text += f"pipeline_tag: {task_name}\n"
    readme_text += f"library_name: {model_card.get('library_name', 'jaxnn')}\n"
    readme_text += f"license: {model_card.get('license', 'apache-2.0')}\n"
    if "license_name" in model_card:
        readme_text += f"license_name: {model_card.get('license_name')}\n"
    if "license_link" in model_card:
        readme_text += f"license_link: {model_card.get('license_link')}\n"
    if "details" in model_card and "Dataset" in model_card["details"]:
        readme_text += "datasets:\n"
        if isinstance(model_card["details"]["Dataset"], (tuple, list)):
            for d in model_card["details"]["Dataset"]:
                readme_text += f"- {d.lower()}\n"
        else:
            readme_text += f"- {model_card['details']['Dataset'].lower()}\n"
        if "Pretrain Dataset" in model_card["details"]:
            if isinstance(model_card["details"]["Pretrain Dataset"], (tuple, list)):
                for d in model_card["details"]["Pretrain Dataset"]:
                    readme_text += f"- {d.lower()}\n"
            else:
                readme_text += (
                    f"- {model_card['details']['Pretrain Dataset'].lower()}\n"
                )
    readme_text += "---\n"
    readme_text += f"# Model card for {model_name}\n"
    if "description" in model_card:
        readme_text += f"\n{model_card['description']}\n"
    if "details" in model_card:
        readme_text += "\n## Model Details\n"
        for k, v in model_card["details"].items():
            if isinstance(v, (list, tuple)):
                readme_text += f"- **{k}:**\n"
                for vi in v:
                    readme_text += f"  - {vi}\n"
            elif isinstance(v, dict):
                readme_text += f"- **{k}:**\n"
                for ki, vi in v.items():
                    readme_text += f"  - {ki}: {vi}\n"
            else:
                readme_text += f"- **{k}:** {v}\n"
    if "usage" in model_card:
        readme_text += "\n## Model Usage\n"
        readme_text += model_card["usage"]
        readme_text += "\n"
    if "comparison" in model_card:
        readme_text += "\n## Model Comparison\n"
        readme_text += model_card["comparison"]
        readme_text += "\n"
    if "citation" in model_card:
        readme_text += "\n## Citation\n"
        if not isinstance(model_card["citation"], (list, tuple)):
            citations = [model_card["citation"]]
        else:
            citations = model_card["citation"]
        for c in citations:
            readme_text += f"```bibtex\n{c}\n```\n"
    return readme_text
