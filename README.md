# JaxNN: Foundation Models in JAX/Flax

JaxNN is an open-source library for foundation models in JAX and Flax. It provides a unified framework for loading, creating, and using pretrained models (e.g., ResNet, ViT).

## Installation

```bash
pip install jaxnn
```

## Usage

### Image Classification

```python
from urllib.request import urlopen
from PIL import Image
import jax

import jaxnn

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/cats-image/resolve/main/cats_image.jpeg'
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
import jax

import jaxnn

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/cats-image/resolve/main/cats_image.jpeg'
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
import jax

import jaxnn

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/cats-image/resolve/main/cats_image.jpeg'
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
output = model.forward_head(output, pre_logits=True)                        # (1, num_features)
```
## Pretrained models
The data in table for `img/sec` is given for PyTorch. The output tensor (logits) was compared against the PyTorch original weights.
|model                                     |img_size|top1 |top5 |param_count|gmacs|macts|img/sec|
|------------------------------------------|--------|-----|-----|-----------|-----|-----|-------|
|[resnet152d.ra2_in1k](https://huggingface.co/JaxNN/resnet152d.ra2_in1k)|320     |83.67|96.74|60.2       |24.1 |47.7 |706    |
|[resnet152.a1h_in1k](https://huggingface.co/JaxNN/resnet152.a1h_in1k)|288     |83.46|96.54|60.2       |19.1 |37.3 |904    |
|[resnet152d.ra2_in1k](https://huggingface.co/JaxNN/resnet152d.ra2_in1k)|256     |83.14|96.38|60.2       |15.4 |30.5 |1096   |
|[resnet101d.ra2_in1k](https://huggingface.co/JaxNN/resnet101d.ra2_in1k)|320     |83.02|96.45|44.6       |16.5 |34.8 |992    |
|[resnet152.a1h_in1k](https://huggingface.co/JaxNN/resnet152.a1h_in1k)|224     |82.8 |96.13|60.2       |11.6 |22.6 |1486   |
|[resnet101.a1h_in1k](https://huggingface.co/JaxNN/resnet101.a1h_in1k)|288     |82.8 |96.32|44.6       |13.0 |26.8 |1291   |
|[resnet152.a1_in1k](https://huggingface.co/JaxNN/resnet152.a1_in1k)|288     |82.74|95.71|60.2       |19.1 |37.3 |905    |
|[resnet152.a2_in1k](https://huggingface.co/JaxNN/resnet152.a2_in1k)|288     |82.62|95.75|60.2       |19.1 |37.3 |904    |
|[resnet101.a1_in1k](https://huggingface.co/JaxNN/resnet101.a1_in1k)|288     |82.31|95.63|44.6       |13.0 |26.8 |1291   |
|[resnet152.tv2_in1k](https://huggingface.co/JaxNN/resnet152.tv2_in1k)|224     |82.29|96.0 |60.2       |11.6 |22.6 |1484   |
|[resnet101d.ra2_in1k](https://huggingface.co/JaxNN/resnet101d.ra2_in1k)|256     |82.26|96.07|44.6       |10.6 |22.2 |1542   |
|[resnet101.a2_in1k](https://huggingface.co/JaxNN/resnet101.a2_in1k)|288     |82.24|95.73|44.6       |13.0 |26.8 |1290   |
|[resnet152.a1_in1k](https://huggingface.co/JaxNN/resnet152.a1_in1k)|224     |81.97|95.24|60.2       |11.6 |22.6 |1486   |
|[resnet101.a1h_in1k](https://huggingface.co/JaxNN/resnet101.a1h_in1k)|224     |81.93|95.75|44.6       |7.8  |16.2 |2122   |
|[resnet101.tv2_in1k](https://huggingface.co/JaxNN/resnet101.tv2_in1k)|224     |81.9 |95.77|44.6       |7.8  |16.2 |2118   |
|[resnet152.a2_in1k](https://huggingface.co/JaxNN/resnet152.a2_in1k)|224     |81.77|95.22|60.2       |11.6 |22.6 |1485   |
|[resnet101.a1_in1k](https://huggingface.co/JaxNN/resnet101.a1_in1k)|224     |81.5 |95.16|44.6       |7.8  |16.2 |2125   |
|[resnet50d.a1_in1k](https://huggingface.co/JaxNN/resnet50d.a1_in1k)|288     |81.44|95.22|25.6       |7.2  |19.7 |1908   |
|[resnet50d.ra2_in1k](https://huggingface.co/JaxNN/resnet50d.ra2_in1k)|288     |81.37|95.74|25.6       |7.2  |19.7 |1910   |
|[resnet101.a2_in1k](https://huggingface.co/JaxNN/resnet101.a2_in1k)|224     |81.32|95.19|44.6       |7.8  |16.2 |2125   |
|[resnet50.a1_in1k](https://huggingface.co/JaxNN/resnet50.a1_in1k)|288     |81.22|95.11|25.6       |6.8  |18.4 |2089   |
|[resnet50_gn.a1h_in1k](https://huggingface.co/JaxNN/resnet50_gn.a1h_in1k)|288     |81.22|95.63|25.6       |6.8  |18.4 |676    |
|[resnet50d.a2_in1k](https://huggingface.co/JaxNN/resnet50d.a2_in1k)|288     |81.18|95.09|25.6       |7.2  |19.7 |1908   |
|[resnet50.fb_swsl_ig1b_ft_in1k](https://huggingface.co/JaxNN/resnet50.fb_swsl_ig1b_ft_in1k)|224     |81.18|95.98|25.6       |4.1  |11.1 |3455   |
|[resnet152s.gluon_in1k](https://huggingface.co/JaxNN/resnet152s.gluon_in1k)|224     |81.02|95.41|60.3       |12.9 |25.0 |1347   |
|[resnet50.d_in1k](https://huggingface.co/JaxNN/resnet50.d_in1k)|288     |80.97|95.44|25.6       |6.8  |18.4 |2085   |
|[resnet50.c1_in1k](https://huggingface.co/JaxNN/resnet50.c1_in1k)|288     |80.91|95.55|25.6       |6.8  |18.4 |2084   |
|[resnet50.c2_in1k](https://huggingface.co/JaxNN/resnet50.c2_in1k)|288     |80.86|95.52|25.6       |6.8  |18.4 |2085   |
|[resnet50.tv2_in1k](https://huggingface.co/JaxNN/resnet50.tv2_in1k)|224     |80.85|95.43|25.6       |4.1  |11.1 |3450   |
|[resnet50.a2_in1k](https://huggingface.co/JaxNN/resnet50.a2_in1k)|288     |80.78|94.99|25.6       |6.8  |18.4 |2088   |
|[resnet50.b1k_in1k](https://huggingface.co/JaxNN/resnet50.b1k_in1k)|288     |80.71|95.43|25.6       |6.8  |18.4 |2087   |
|[resnet50d.a1_in1k](https://huggingface.co/JaxNN/resnet50d.a1_in1k)|224     |80.68|94.71|25.6       |4.4  |11.9 |3162   |
|[resnet152.a3_in1k](https://huggingface.co/JaxNN/resnet152.a3_in1k)|224     |80.56|95.0 |60.2       |11.6 |22.6 |1483   |
|[resnet50d.ra2_in1k](https://huggingface.co/JaxNN/resnet50d.ra2_in1k)|224     |80.53|95.16|25.6       |4.4  |11.9 |3164   |
|[resnet152d.gluon_in1k](https://huggingface.co/JaxNN/resnet152d.gluon_in1k)|224     |80.47|95.2 |60.2       |11.8 |23.4 |1428   |
|[resnet50.b2k_in1k](https://huggingface.co/JaxNN/resnet50.b2k_in1k)|288     |80.45|95.32|25.6       |6.8  |18.4 |2086   |
|[resnet101d.gluon_in1k](https://huggingface.co/JaxNN/resnet101d.gluon_in1k)|224     |80.42|95.01|44.6       |8.1  |17.0 |2007   |
|[resnet50.a1_in1k](https://huggingface.co/JaxNN/resnet50.a1_in1k)|224     |80.38|94.6 |25.6       |4.1  |11.1 |3461   |
|[resnet101s.gluon_in1k](https://huggingface.co/JaxNN/resnet101s.gluon_in1k)|224     |80.28|95.16|44.7       |9.2  |18.6 |1851   |
|[resnet50d.a2_in1k](https://huggingface.co/JaxNN/resnet50d.a2_in1k)|224     |80.22|94.63|25.6       |4.4  |11.9 |3162   |
|[resnet152.tv2_in1k](https://huggingface.co/JaxNN/resnet152.tv2_in1k)|176     |80.2 |94.64|60.2       |7.2  |14.0 |2346   |
|[resnet50_gn.a1h_in1k](https://huggingface.co/JaxNN/resnet50_gn.a1h_in1k)|224     |80.06|94.95|25.6       |4.1  |11.1 |1109   |
|[resnet50.ram_in1k](https://huggingface.co/JaxNN/resnet50.ram_in1k)|288     |79.97|95.05|25.6       |6.8  |18.4 |2086   |
|[resnet152c.gluon_in1k](https://huggingface.co/JaxNN/resnet152c.gluon_in1k)|224     |79.92|94.84|60.2       |11.8 |23.4 |1455   |
|[resnet50.d_in1k](https://huggingface.co/JaxNN/resnet50.d_in1k)|224     |79.91|94.67|25.6       |4.1  |11.1 |3456   |
|[resnet101.tv2_in1k](https://huggingface.co/JaxNN/resnet101.tv2_in1k)|176     |79.9 |94.6 |44.6       |4.9  |10.1 |3341   |
|[resnet50.c2_in1k](https://huggingface.co/JaxNN/resnet50.c2_in1k)|224     |79.88|94.87|25.6       |4.1  |11.1 |3455   |
|[resnet50.a2_in1k](https://huggingface.co/JaxNN/resnet50.a2_in1k)|224     |79.85|94.56|25.6       |4.1  |11.1 |3460   |
|[resnet50.ra_in1k](https://huggingface.co/JaxNN/resnet50.ra_in1k)|288     |79.83|94.97|25.6       |6.8  |18.4 |2087   |
|[resnet101.a3_in1k](https://huggingface.co/JaxNN/resnet101.a3_in1k)|224     |79.82|94.62|44.6       |7.8  |16.2 |2114   |
|[resnet50.c1_in1k](https://huggingface.co/JaxNN/resnet50.c1_in1k)|224     |79.74|94.95|25.6       |4.1  |11.1 |3455   |
|[resnet152.gluon_in1k](https://huggingface.co/JaxNN/resnet152.gluon_in1k)|224     |79.68|94.74|60.2       |11.6 |22.6 |1486   |
|[resnet50.bt_in1k](https://huggingface.co/JaxNN/resnet50.bt_in1k)|288     |79.63|94.91|25.6       |6.8  |18.4 |2086   |
|[resnet101c.gluon_in1k](https://huggingface.co/JaxNN/resnet101c.gluon_in1k)|224     |79.53|94.58|44.6       |8.1  |17.0 |2062   |
|[resnet50.b1k_in1k](https://huggingface.co/JaxNN/resnet50.b1k_in1k)|224     |79.52|94.61|25.6       |4.1  |11.1 |3459   |
|[resnet50.tv2_in1k](https://huggingface.co/JaxNN/resnet50.tv2_in1k)|176     |79.42|94.64|25.6       |2.6  |6.9  |5397   |
|[resnet50.b2k_in1k](https://huggingface.co/JaxNN/resnet50.b2k_in1k)|224     |79.38|94.57|25.6       |4.1  |11.1 |3459   |
|[resnet101.gluon_in1k](https://huggingface.co/JaxNN/resnet101.gluon_in1k)|224     |79.31|94.53|44.6       |7.8  |16.2 |2125   |
|[resnet50.fb_ssl_yfcc100m_ft_in1k](https://huggingface.co/JaxNN/resnet50.fb_ssl_yfcc100m_ft_in1k)|224     |79.22|94.84|25.6       |4.1  |11.1 |3451   |
|[resnet50d.gluon_in1k](https://huggingface.co/JaxNN/resnet50d.gluon_in1k)|224     |79.07|94.48|25.6       |4.4  |11.9 |3162   |
|[resnet50.ram_in1k](https://huggingface.co/JaxNN/resnet50.ram_in1k)|224     |79.03|94.38|25.6       |4.1  |11.1 |3453   |
|[resnet50.am_in1k](https://huggingface.co/JaxNN/resnet50.am_in1k)|224     |79.01|94.39|25.6       |4.1  |11.1 |3461   |
|[resnet152.a3_in1k](https://huggingface.co/JaxNN/resnet152.a3_in1k)|160     |78.89|94.11|60.2       |5.9  |11.5 |2745   |
|[resnet50.ra_in1k](https://huggingface.co/JaxNN/resnet50.ra_in1k)|224     |78.81|94.32|25.6       |4.1  |11.1 |3454   |
|[resnet50s.gluon_in1k](https://huggingface.co/JaxNN/resnet50s.gluon_in1k)|224     |78.72|94.23|25.7       |5.5  |13.5 |2796   |
|[resnet50d.a3_in1k](https://huggingface.co/JaxNN/resnet50d.a3_in1k)|224     |78.71|94.24|25.6       |4.4  |11.9 |3154   |
|[resnet50.bt_in1k](https://huggingface.co/JaxNN/resnet50.bt_in1k)|224     |78.46|94.27|25.6       |4.1  |11.1 |3454   |
|[resnet34d.ra2_in1k](https://huggingface.co/JaxNN/resnet34d.ra2_in1k)|288     |78.43|94.35|21.8       |6.5  |7.5  |3291   |
|[resnet26t.ra2_in1k](https://huggingface.co/JaxNN/resnet26t.ra2_in1k)|320     |78.33|94.13|16.0       |5.2  |16.4 |2391   |
|[resnet152.tv_in1k](https://huggingface.co/JaxNN/resnet152.tv_in1k)|224     |78.32|94.04|60.2       |11.6 |22.6 |1487   |
|[resnet50.a3_in1k](https://huggingface.co/JaxNN/resnet50.a3_in1k)|224     |78.06|93.78|25.6       |4.1  |11.1 |3450   |
|[resnet50c.gluon_in1k](https://huggingface.co/JaxNN/resnet50c.gluon_in1k)|224     |78.0 |93.99|25.6       |4.4  |11.9 |3286   |
|[resnet34.a1_in1k](https://huggingface.co/JaxNN/resnet34.a1_in1k)|288     |77.92|93.77|21.8       |6.1  |6.2  |3609   |
|[resnet101.a3_in1k](https://huggingface.co/JaxNN/resnet101.a3_in1k)|160     |77.88|93.71|44.6       |4.0  |8.3  |3926   |
|[resnet26t.ra2_in1k](https://huggingface.co/JaxNN/resnet26t.ra2_in1k)|256     |77.87|93.84|16.0       |3.4  |10.5 |3772   |
|[resnet50.gluon_in1k](https://huggingface.co/JaxNN/resnet50.gluon_in1k)|224     |77.58|93.72|25.6       |4.1  |11.1 |3455   |
|[resnet26d.bt_in1k](https://huggingface.co/JaxNN/resnet26d.bt_in1k)|288     |77.41|93.63|16.0       |4.3  |13.5 |2907   |
|[resnet101.tv_in1k](https://huggingface.co/JaxNN/resnet101.tv_in1k)|224     |77.38|93.54|44.6       |7.8  |16.2 |2125   |
|[resnet50d.a3_in1k](https://huggingface.co/JaxNN/resnet50d.a3_in1k)|160     |77.22|93.27|25.6       |2.2  |6.1  |5982   |
|[resnet34.a2_in1k](https://huggingface.co/JaxNN/resnet34.a2_in1k)|288     |77.15|93.27|21.8       |6.1  |6.2  |3615   |
|[resnet34d.ra2_in1k](https://huggingface.co/JaxNN/resnet34d.ra2_in1k)|224     |77.1 |93.37|21.8       |3.9  |4.5  |5436   |
|[resnet26d.bt_in1k](https://huggingface.co/JaxNN/resnet26d.bt_in1k)|224     |76.7 |93.17|16.0       |2.6  |8.2  |4859   |
|[resnet34.bt_in1k](https://huggingface.co/JaxNN/resnet34.bt_in1k)|288     |76.5 |93.35|21.8       |6.1  |6.2  |3617   |
|[resnet34.a1_in1k](https://huggingface.co/JaxNN/resnet34.a1_in1k)|224     |76.42|92.87|21.8       |3.7  |3.7  |5984   |
|[resnet26.bt_in1k](https://huggingface.co/JaxNN/resnet26.bt_in1k)|288     |76.35|93.18|16.0       |3.9  |12.2 |3331   |
|[resnet50.tv_in1k](https://huggingface.co/JaxNN/resnet50.tv_in1k)|224     |76.13|92.86|25.6       |4.1  |11.1 |3457   |
|[resnet50.a3_in1k](https://huggingface.co/JaxNN/resnet50.a3_in1k)|160     |75.96|92.5 |25.6       |2.1  |5.7  |6490   |
|[resnet34.a2_in1k](https://huggingface.co/JaxNN/resnet34.a2_in1k)|224     |75.52|92.44|21.8       |3.7  |3.7  |5991   |
|[resnet26.bt_in1k](https://huggingface.co/JaxNN/resnet26.bt_in1k)|224     |75.3 |92.58|16.0       |2.4  |7.4  |5583   |
|[resnet34.bt_in1k](https://huggingface.co/JaxNN/resnet34.bt_in1k)|224     |75.16|92.18|21.8       |3.7  |3.7  |5994   |
|[resnet34.gluon_in1k](https://huggingface.co/JaxNN/resnet34.gluon_in1k)|224     |74.57|91.98|21.8       |3.7  |3.7  |5984   |
|[resnet18d.ra2_in1k](https://huggingface.co/JaxNN/resnet18d.ra2_in1k)|288     |73.81|91.83|11.7       |3.4  |5.4  |5196   |
|[resnet34.tv_in1k](https://huggingface.co/JaxNN/resnet34.tv_in1k)|224     |73.32|91.42|21.8       |3.7  |3.7  |5979   |
|[resnet18.fb_swsl_ig1b_ft_in1k](https://huggingface.co/JaxNN/resnet18.fb_swsl_ig1b_ft_in1k)|224     |73.28|91.73|11.7       |1.8  |2.5  |10213  |
|[resnet18.a1_in1k](https://huggingface.co/JaxNN/resnet18.a1_in1k)|288     |73.16|91.03|11.7       |3.0  |4.1  |6050   |
|[resnet34.a3_in1k](https://huggingface.co/JaxNN/resnet34.a3_in1k)|224     |72.98|91.11|21.8       |3.7  |3.7  |5967   |
|[resnet18.fb_ssl_yfcc100m_ft_in1k](https://huggingface.co/JaxNN/resnet18.fb_ssl_yfcc100m_ft_in1k)|224     |72.6 |91.42|11.7       |1.8  |2.5  |10213  |
|[resnet18.a2_in1k](https://huggingface.co/JaxNN/resnet18.a2_in1k)|288     |72.37|90.59|11.7       |3.0  |4.1  |6051   |
|[resnet14t.c3_in1k](https://huggingface.co/JaxNN/resnet14t.c3_in1k)|224     |72.26|90.31|10.1       |1.7  |5.8  |7026   |
|[resnet18d.ra2_in1k](https://huggingface.co/JaxNN/resnet18d.ra2_in1k)|224     |72.26|90.68|11.7       |2.1  |3.3  |8707   |
|[resnet18.a1_in1k](https://huggingface.co/JaxNN/resnet18.a1_in1k)|224     |71.49|90.07|11.7       |1.8  |2.5  |10187  |
|[resnet14t.c3_in1k](https://huggingface.co/JaxNN/resnet14t.c3_in1k)|176     |71.31|89.69|10.1       |1.1  |3.6  |10970  |
|[resnet18.gluon_in1k](https://huggingface.co/JaxNN/resnet18.gluon_in1k)|224     |70.84|89.76|11.7       |1.8  |2.5  |10210  |
|[resnet18.a2_in1k](https://huggingface.co/JaxNN/resnet18.a2_in1k)|224     |70.64|89.47|11.7       |1.8  |2.5  |10194  |
|[resnet34.a3_in1k](https://huggingface.co/JaxNN/resnet34.a3_in1k)|160     |70.56|89.52|21.8       |1.9  |1.9  |10737  |
|[resnet18.tv_in1k](https://huggingface.co/JaxNN/resnet18.tv_in1k)|224     |69.76|89.07|11.7       |1.8  |2.5  |10205  |
|[resnet18.a3_in1k](https://huggingface.co/JaxNN/resnet18.a3_in1k)|224     |68.25|88.17|11.7       |1.8  |2.5  |10167  |
|[resnet18.a3_in1k](https://huggingface.co/JaxNN/resnet18.a3_in1k)|160     |65.66|86.26|11.7       |0.9  |1.3  |18229  |


## Roadmap

| Component | Status |
|---|---|
| ViT, MobileNet, and more | ⏳ |
| Training/eval loop with `optax` | ⏳ |
| Documentation | ⏳ |

## References

- [ResNet Strikes Back](https://arxiv.org/abs/2110.00476)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
