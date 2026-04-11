"""Tests for model creation (no pretrained weights)."""

import pytest
import jax.numpy as jnp
from jaxnn import create_model


class TestCreateModel:
    def test_create_resnet34(self):
        model = create_model("resnet34", pretrained=False)
        assert model is not None
        assert model.num_classes == 1000

    def test_create_resnet34_custom_classes(self):
        model = create_model("resnet34", pretrained=False, num_classes=10)
        assert model.num_classes == 10

    def test_create_resnet34_forward(self):
        model = create_model("resnet34", pretrained=False)
        x = jnp.ones((1, 224, 224, 3))
        out = model(x)
        assert out.shape == (1, 1000)

    def test_create_unknown_model(self):
        with pytest.raises(RuntimeError, match="Unknown model"):
            create_model("nonexistent_model_xyz")
