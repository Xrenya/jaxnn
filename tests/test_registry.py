"""Tests for model registry and factory functions."""
import pytest
from jaxnn.models._factory import parse_model_name
from jaxnn.models._registry import (
    list_models, list_pretrained, is_model, split_model_name_tag,
    get_pretrained_cfg,
)


class TestParseModelName:
    def test_plain_name(self):
        scheme, name = parse_model_name("resnet34")
        assert scheme is None
        assert name == "resnet34"

    def test_hf_hub(self):
        scheme, name = parse_model_name("hf-hub:JaxNN/resnet34.a1_in1k")
        assert scheme == "hf-hub"
        assert name == "JaxNN/resnet34.a1_in1k"

    def test_local_dir_unix(self):
        scheme, path = parse_model_name("local-dir:/tmp/my_model")
        assert scheme == "local-dir"
        assert path == "/tmp/my_model"

    def test_local_dir_windows(self):
        scheme, path = parse_model_name("local-dir:C:\\Users\\test\\model")
        assert scheme == "local-dir"
        assert path == "C:\\Users\\test\\model"

    def test_plain_with_path(self):
        scheme, name = parse_model_name("path/to/resnet34")
        assert scheme is None
        assert name == "resnet34"


class TestSplitModelNameTag:
    def test_no_tag(self):
        result = split_model_name_tag("resnet34")
        assert result["model_name"] == "resnet34"
        assert result["tag"] == ""

    def test_with_tag(self):
        result = split_model_name_tag("resnet34.a1_in1k")
        assert result["model_name"] == "resnet34"
        assert result["tag"] == "a1_in1k"


class TestRegistry:
    def test_resnet34_registered(self):
        assert is_model("resnet34")

    def test_unknown_model(self):
        assert not is_model("nonexistent_model_xyz")

    def test_list_models_returns_list(self):
        models = list_models()
        assert isinstance(models, list)
        assert "resnet34" in models

    def test_list_models_filter(self):
        models = list_models(filter="resnet*")
        assert all("resnet" in m for m in models)

    def test_list_pretrained(self):
        models = list_pretrained()
        assert isinstance(models, list)

    def test_get_pretrained_cfg(self):
        cfg = get_pretrained_cfg("resnet34")
        assert cfg is not None
        assert cfg.architecture == "resnet34"

    def test_get_pretrained_cfg_with_tag(self):
        cfg = get_pretrained_cfg("resnet34.a1_in1k")
        assert cfg is not None
