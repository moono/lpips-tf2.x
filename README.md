# lpips-tf2.x
* This is tensorflow 2.x conversion of official repo [LPIPS metric][Offical repo] (pytorch)
* Similar to [lpips-tensorflow][TF repo] except,
  * In this repo, network architecture is explicitly implemented rather than converting with ONNX.

## Limitation
* Currently only `model='net-lin', net='vgg'` is implemented

## Example usage
* input image should be [0.0 ~ 255.0], float32, NHWC format

```python
import os
import numpy as np
import tensorflow as tf
from PIL import Image

from models.lpips_tensorflow import learned_perceptual_metric_model


def load_image(fn):
    image = Image.open(fn)
    image = np.asarray(image)
    image = np.expand_dims(image, axis=0)

    image = tf.constant(image, dtype=tf.dtypes.float32)
    return image


image_size = 64
model_dir = './models'
vgg_ckpt_fn = os.path.join(model_dir, 'vgg', 'exported')
lin_ckpt_fn = os.path.join(model_dir, 'lin', 'exported')
lpips = learned_perceptual_metric_model(image_size, vgg_ckpt_fn, lin_ckpt_fn)

# official pytorch model value:
# Distance: ex_ref.png <-> ex_p0.png = 0.569
# Distance: ex_ref.png <-> ex_p1.png = 0.422
image_fn1 = './imgs/ex_ref.png'
image_fn2 = './imgs/ex_p0.png'
# image_fn2 = './imgs/ex_p1.png'

image1 = load_image(image_fn1)
image2 = load_image(image_fn2)
dist01 = lpips([image1, image2])
print('Distance: {:.3f}'.format(dist01))
```

### To reproduce same checkpoint files...
* Clone official repo [LPIPS metric][Offical repo]
* Place `./example_export_script/convert_to_tensorflow.py` and `./models/lpips_tensorflow.py` on root directory
* Run `convert_to_tensorflow.py`

[Offical repo]: https://github.com/richzhang/PerceptualSimilarity
[TF repo]: https://github.com/alexlee-gk/lpips-tensorflow