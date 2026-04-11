"""Tests for model builder and loading utilities."""

import json
import pytest

from jaxnn.models._hub import load_model_config_from_path, _parse_model_cfg
from jaxnn.models._builder import _resolve_pretrained_source


class TestResolvePretrainedSource:
    def test_state_dict_source(self):
        cfg = {"state_dict": {"layer.weight": [1, 2, 3]}}
        source, loc = _resolve_pretrained_source(cfg)
        assert source == "state_dict"

    def test_file_source(self, tmp_path):
        # Simulate a local dir with "file" key (set by load_model_config_from_path)
        model_dir = tmp_path / "my_model"
        model_dir.mkdir()
        cfg = {"file": str(model_dir)}
        source, loc = _resolve_pretrained_source(cfg)
        assert source == "local-dir"
        assert loc == str(model_dir)

    def test_local_dir_source(self, tmp_path):
        model_dir = tmp_path / "my_model"
        model_dir.mkdir()
        cfg = {"local_dir": str(model_dir)}
        source, loc = _resolve_pretrained_source(cfg)
        assert source == "local-dir"
        assert loc == str(model_dir)

    def test_hf_hub_source(self):
        cfg = {"hf_hub_id": "JaxNN/resnet34"}
        source, loc = _resolve_pretrained_source(cfg)
        assert source == "hf-hub"
        assert loc == "JaxNN/resnet34"

    def test_no_source(self):
        cfg = {}
        source, loc = _resolve_pretrained_source(cfg)
        assert source is None
        assert loc is None


class TestLoadModelConfigFromPath:
    def test_load_valid_config(self, tmp_path):
        config = {
            "architecture": "resnet34",
            "pretrained_cfg": {
                "num_classes": 1000,
                "input_size": (224, 224, 3),
            },
            "model_args": {"layers": [3, 4, 6, 3]},
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        pretrained_cfg, model_name, model_args = load_model_config_from_path(tmp_path)
        assert model_name == "resnet34"
        assert pretrained_cfg["num_classes"] == 1000
        assert pretrained_cfg["source"] == "local-dir"
        assert pretrained_cfg["file"] == str(tmp_path)
        assert model_args == {"layers": [3, 4, 6, 3]}

    def test_missing_config(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model_config_from_path(tmp_path)


class TestParseModelCfg:
    def test_label_descriptions_not_duplicated(self):
        cfg = {
            "architecture": "resnet34",
            "pretrained_cfg": {"num_classes": 10},
            "label_names": ["cat", "dog"],
            "label_descriptions": {"cat": "A cat", "dog": "A dog"},
        }
        pretrained_cfg, _, _ = _parse_model_cfg(cfg, {})
        assert pretrained_cfg["label_names"] == ["cat", "dog"]
        assert pretrained_cfg["label_descriptions"] == {"cat": "A cat", "dog": "A dog"}

    def test_no_label_descriptions(self):
        cfg = {
            "architecture": "resnet34",
            "pretrained_cfg": {"num_classes": 10},
        }
        pretrained_cfg, _, _ = _parse_model_cfg(cfg, {})
        assert "label_descriptions" not in pretrained_cfg
