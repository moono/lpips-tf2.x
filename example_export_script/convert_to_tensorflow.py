import os
import numpy as np
import torch
import models

from collections import OrderedDict
from lpips_tensorflow import perceptual_model, linear_model, learned_perceptual_metric_model


def get_official_model_weights():
    # Initializing the pytorch_model
    # pytorch_model.model.net.module.net: vgg16
    # pytorch_model.model.net.module.lins: linear layer
    pytorch_model = models.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)

    # extract pytorch tensors
    vgg_model_weights = OrderedDict()
    for k, v in pytorch_model.model.net.module.net.named_parameters():
        vgg_model_weights[k] = v.data

    model_path = './models/weights/v0.1/vgg.pth'
    lin_model_weights = torch.load(model_path)

    # convert to numpy arrays
    vgg_model_weights_np = OrderedDict({name: tensor.cpu().data.numpy() for name, tensor in vgg_model_weights.items()})
    lin_model_weights_np = OrderedDict({name: tensor.cpu().data.numpy() for name, tensor in lin_model_weights.items()})
    return vgg_model_weights_np, lin_model_weights_np


def get_wright_name_correspondence():
    vgg_correspondence = OrderedDict({
        'slice1.0.weight': 'block1_conv1/kernel',
        'slice1.0.bias': 'block1_conv1/bias',
        'slice1.2.weight': 'block1_conv2/kernel',
        'slice1.2.bias': 'block1_conv2/bias',

        'slice2.5.weight': 'block2_conv1/kernel',
        'slice2.5.bias': 'block2_conv1/bias',
        'slice2.7.weight': 'block2_conv2/kernel',
        'slice2.7.bias': 'block2_conv2/bias',

        'slice3.10.weight': 'block3_conv1/kernel',
        'slice3.10.bias': 'block3_conv1/bias',
        'slice3.12.weight': 'block3_conv2/kernel',
        'slice3.12.bias': 'block3_conv2/bias',
        'slice3.14.weight': 'block3_conv3/kernel',
        'slice3.14.bias': 'block3_conv3/bias',

        'slice4.17.weight': 'block4_conv1/kernel',
        'slice4.17.bias': 'block4_conv1/bias',
        'slice4.19.weight': 'block4_conv2/kernel',
        'slice4.19.bias': 'block4_conv2/bias',
        'slice4.21.weight': 'block4_conv3/kernel',
        'slice4.21.bias': 'block4_conv3/bias',

        'slice5.24.weight': 'block5_conv1/kernel',
        'slice5.24.bias': 'block5_conv1/bias',
        'slice5.26.weight': 'block5_conv2/kernel',
        'slice5.26.bias': 'block5_conv2/bias',
        'slice5.28.weight': 'block5_conv3/kernel',
        'slice5.28.bias': 'block5_conv3/bias',
    })

    lin_correspondence = OrderedDict({
        'lin0.model.1.weight': 'lin0/kernel',
        'lin1.model.1.weight': 'lin1/kernel',
        'lin2.model.1.weight': 'lin2/kernel',
        'lin3.model.1.weight': 'lin3/kernel',
        'lin4.model.1.weight': 'lin4/kernel',
    })
    return vgg_correspondence, lin_correspondence


# pytorch weights copy
def set_pytorch_pretrained_weights(pytorch_weights, dst_model, correspondence):
    n_official_weights = len(pytorch_weights)
    n_successful_copies = 0
    for src_name, dst_name in correspondence.items():
        src_weight = pytorch_weights[src_name]
        if 'weight' in src_name:
            src_weight = np.transpose(src_weight, axes=(2, 3, 1, 0))

        for dst_weight in dst_model.weights:
            if dst_name in dst_weight.name and dst_weight.shape == src_weight.shape:
                dst_weight.assign(src_weight)
                success = np.allclose(dst_weight.numpy(), src_weight)
                print(success)
                if success:
                    n_successful_copies += 1
                break

    assert n_official_weights == n_successful_copies
    return


def export_official_model(image_size, save_dir):
    vgg_channels = [64, 128, 256, 512, 512]

    dummy_inputs = list()
    for ii, channel in enumerate(vgg_channels):
        input_image_size = image_size // 2 ** ii
        dummy_inputs.append(np.ones(shape=(1, channel, input_image_size, input_image_size)))

    # extract official weights first
    vgg_model_weights_np, lin_model_weights_np = get_official_model_weights()

    # instantiate tensorflow models
    vgg_model = perceptual_model(image_size)
    vgg_model.summary()
    lin_model = linear_model(image_size)
    lin_model.summary()

    # find correspondences
    vgg_correspondence, lin_correspondence = get_wright_name_correspondence()

    # copy official weights
    set_pytorch_pretrained_weights(vgg_model_weights_np, vgg_model, vgg_correspondence)
    set_pytorch_pretrained_weights(lin_model_weights_np, lin_model, lin_correspondence)

    # save tf.keras model
    vgg_ckpt_fn = os.path.join(save_dir, 'vgg', 'exported')
    lin_ckpt_fn = os.path.join(save_dir, 'lin', 'exported')
    vgg_model.save_weights(vgg_ckpt_fn, save_format='tf')
    lin_model.save_weights(lin_ckpt_fn, save_format='tf')
    return


def run_lpips_metric(image_size, save_dir):
    import tensorflow as tf
    from PIL import Image
    # from tensorflow.keras.applications.vgg16 import preprocess_input

    def load_image(fn):
        image = Image.open(fn)
        image = np.asarray(image)
        image = np.expand_dims(image, axis=0)

        image = tf.constant(image, dtype=tf.dtypes.float32)
        return image

    vgg_ckpt_fn = os.path.join(save_dir, 'vgg', 'exported')
    lin_ckpt_fn = os.path.join(save_dir, 'lin', 'exported')
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
    return


def main():
    # parameters
    ckpt_dir = './tensorflow_form'
    image_size = 64

    # 1. export pytorch model to tensorflow keras model
    export_official_model(image_size, ckpt_dir)

    # 2. test run
    run_lpips_metric(image_size, ckpt_dir)
    return


if __name__ == '__main__':
    main()
