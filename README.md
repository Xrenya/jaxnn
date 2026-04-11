# JaxNN: Foundation Models in JAX/Flax

JaxNN is an open-source library for foundation models in JAX and Flax. It provides a unified framework for loading, creating, and using pretrained models (e.g., ResNet, ViT).

> **Note:** `jaxnn` is still in development. Pip installation is not yet available but will be released soon when more models are ported to Flax/JAX.

## Installation

```bash
pip install jaxnn  # coming soon
```

## Usage

### Image Classification

```python
from urllib.request import urlopen
from PIL import Image
import jaxnn
import jax

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = jaxnn.create_model('resnet34.a1_in1k', pretrained=True)
model.eval()

# Get model-specific transforms (normalization, resize)
data_config = jaxnn.data.resolve_model_data_config(model)
transforms = jaxnn.data.create_transform(**data_config, is_training=False)

output = model(jax.numpy.expand_dims(transforms(img), 0))

top5_probabilities, top5_class_indices = jax.lax.top_k(
    jax.nn.softmax(output, axis=-1) * 100, k=5
)
```

### Feature Map Extraction

```python
from urllib.request import urlopen
from PIL import Image
import jaxnn
import jax

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = jaxnn.create_model(
    'resnet34.a1_in1k',
    pretrained=True,
    features_only=True,
)
model.eval()

data_config = jaxnn.data.resolve_model_data_config(model)
transforms = jaxnn.data.create_transform(**data_config, is_training=False)

output = model(jax.numpy.expand_dims(transforms(img), 0))

for o in output:
    print(o.shape)
# (1, 112, 112, 64)
# (1, 56, 56, 64)
# (1, 28, 28, 128)
# (1, 14, 14, 256)
# (1, 7, 7, 512)
```

### Image Embeddings

```python
from urllib.request import urlopen
from PIL import Image
import jaxnn
import jax

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = jaxnn.create_model(
    'resnet34.a1_in1k',
    pretrained=True,
    num_classes=0,  # remove classifier
)
model.eval()

data_config = jaxnn.data.resolve_model_data_config(model)
transforms = jaxnn.data.create_transform(**data_config, is_training=False)

output = model(jax.numpy.expand_dims(transforms(img), 0))

# Or use forward methods directly:
output = model.forward_features(jax.numpy.expand_dims(transforms(img), 0))  # (1, 7, 7, 512)
output = model.forward_head(output, pre_logits=True)                         # (1, num_features)
```

## Roadmap

| Component | Status |
|---|---|
| Model registry + factory (`create_model`) | ✅ |
| Pretrained ResNet family | ✅ |
| Preprocessing + normalization | ✅ |
| Weight loading from Hugging Face Hub | ✅ |
| CLI tool (`jaxnn list`, `jaxnn info`) | ✅ |
| PyPI package | ⏳ |
| ViT, MobileNet, and more | ⏳ |
| Training/eval loop with `optax` | ⏳ |
| Documentation | ⏳ |

## References

- [ResNet Strikes Back](https://arxiv.org/abs/2110.00476)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
