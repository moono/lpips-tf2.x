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
image_fn3 = './imgs/ex_p1.png'

image1 = load_image(image_fn1)
image2 = load_image(image_fn2)
image3 = load_image(image_fn3)

batch_target = tf.concat([image1, image1], axis=0)
batch_images = tf.concat([image2, image3], axis=0)
dist = lpips([batch_target, batch_images])
print('Distance ref <-> p0: {:.3f}'.format(dist[0]))
print('Distance ref <-> p1: {:.3f}'.format(dist[1]))
