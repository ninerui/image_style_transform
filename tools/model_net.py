import functools

import numpy as np
import tensorflow as tf
from tensorflow import keras

from tools import vgg_tool

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'


def get_style_features(vgg_path, style_target):
    style_shape = (1,) + style_target.shape  # 风格图片的shape
    style_features = {}
    with tf.Session() as sess:
        style_image = tf.placeholder(tf.float32, style_shape, "style_image")
        style_image_pre = vgg_tool.preprocess(style_image)  # 减去vgg网络处理图片时的均值
        net = vgg_tool.net(vgg_path, style_image_pre)
        for layer in STYLE_LAYERS:
            features = sess.run(net[layer], feed_dict={style_image: [style_target]})
            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T, features) / features.size  # 避免出现太多浮点数, 减小之后的计算
            style_features[layer] = gram
    return style_features


def _tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)


def conv2d_bn_keras(x, filter, filter_size, strides, padding, activation, kernel_init):
    x = keras.layers.Conv2D(
        filter, filter_size, strides, padding=padding, activation=activation, kernel_initializer=kernel_init)(x)
    return keras.layers.BatchNormalization()(x)


def conv_tranpose_kreas(x, filter, filter_size, strides, padding, activation, kernel_init):
    x = keras.layers.Conv2DTranspose(
        filter, filter_size, strides, padding=padding, activation=activation, kernel_initializer=kernel_init)(x)
    return keras.layers.BatchNormalization()(x)


def residual_block_keras(input, filter, filter_size, strides, padding, activation, kernel_init):
    tmp = conv2d_bn_keras(input, filter, filter_size, strides, padding, activation, kernel_init)
    return input + conv2d_bn_keras(tmp, filter, filter_size, strides, padding, None, kernel_init)


def transform_net_keras(image, activation=tf.nn.leaky_relu, kernel_init=tf.initializers.he_normal()):
    conv1 = conv2d_bn_keras(image, 32, 9, 1, "same", activation, kernel_init)
    conv2 = conv2d_bn_keras(conv1, 64, 3, 2, "same", activation, kernel_init)
    conv3 = conv2d_bn_keras(conv2, 128, 3, 2, "same", activation, kernel_init)
    resid1 = residual_block_keras(conv3, 128, 3, 1, "same", activation, kernel_init)
    resid2 = residual_block_keras(resid1, 128, 3, 1, "same", activation, kernel_init)
    resid3 = residual_block_keras(resid2, 128, 3, 1, "same", activation, kernel_init)
    conv_t1 = conv_tranpose_kreas(resid3, 64, 3, 2, "same", activation, kernel_init)
    conv_t2 = conv_tranpose_kreas(conv_t1, 32, 3, 2, "same", activation, kernel_init)
    conv_t3 = conv_tranpose_kreas(conv_t2, 3, 9, 1, "same", None, kernel_init)
    preds = tf.nn.tanh(conv_t3) * 150 + 255. / 2
    return preds


WEIGHTS_INIT_STDEV = .1


def transform_net(image):
    conv1 = _conv_layer(image, 32, 9, 1)
    conv2 = _conv_layer(conv1, 64, 3, 2)
    conv3 = _conv_layer(conv2, 128, 3, 2)
    resid1 = _residual_block(conv3, 3)
    resid2 = _residual_block(resid1, 3)
    resid3 = _residual_block(resid2, 3)
    resid4 = _residual_block(resid3, 3)
    resid5 = _residual_block(resid4, 3)
    conv_t1 = _conv_tranpose_layer(resid5, 64, 3, 2)
    conv_t2 = _conv_tranpose_layer(conv_t1, 32, 3, 2)
    conv_t3 = _conv_layer(conv_t2, 3, 9, 1, relu=False)
    preds = tf.nn.tanh(conv_t3) * 150 + 255. / 2
    return preds


def _conv_layer(net, num_filters, filter_size, strides, relu=True):
    weights_init = _conv_init_vars(net, num_filters, filter_size)
    strides_shape = [1, strides, strides, 1]
    net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME')
    net = _instance_norm(net)
    if relu:
        net = tf.nn.relu(net)

    return net


def _conv_tranpose_layer(net, num_filters, filter_size, strides):
    weights_init = _conv_init_vars(net, num_filters, filter_size, transpose=True)

    batch_size, rows, cols, in_channels = [i.value for i in net.get_shape()]
    new_rows, new_cols = int(rows * strides), int(cols * strides)
    # new_shape = #tf.pack([tf.shape(net)[0], new_rows, new_cols, num_filters])

    new_shape = [batch_size, new_rows, new_cols, num_filters]
    tf_shape = tf.stack(new_shape)
    strides_shape = [1, strides, strides, 1]

    net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME')
    net = _instance_norm(net)
    return tf.nn.relu(net)


def _residual_block(net, filter_size=3):
    tmp = _conv_layer(net, 128, filter_size, 1)
    return net + _conv_layer(tmp, 128, filter_size, 1, relu=False)


def _instance_norm(net, train=True):
    tf.nn.batch_normalization()
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [1, 2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))
    epsilon = 1e-3
    normalized = (net - mu) / (sigma_sq + epsilon) ** .5
    return scale * normalized + shift


def _conv_init_vars(net, out_channels, filter_size, transpose=False):
    _, rows, cols, in_channels = [i.value for i in net.get_shape()]
    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]

    weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev=WEIGHTS_INIT_STDEV, seed=1), dtype=tf.float32)
    return weights_init
