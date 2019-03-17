import os
import sys
import time
import functools

import numpy as np
import tensorflow as tf

import set_params
from tools import utils, model_net, vgg_tool

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'


def _tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)


def map_function(filename):
    image_string = tf.read_file(filename)
    image = tf.image.decode_image(image_string)
    image.set_shape([None, None, None])  # 防止image没有shape而报错
    image = tf.image.resize_images(image, (args.content_image_size, args.content_image_size), method=2)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    # image = tf.cond(image.get_shape()[2] == 1, lambda: tf.image.grayscale_to_rgb(image), lambda: image)
    # image = tf.image.grayscale_to_rgb(image)
    # image = (tf.cast(image, tf.float32) - 127.5) / 128
    return image


def main():
    if str(args.gpu_device):  # 指定gpu设备
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device)

    style_image = utils.get_img(args.style_image)
    style_features = model_net.get_style_features(args.vgg_path, style_image)
    # model_net.get_content_features(
    #     args.vgg_path, args.content_image_size, args.content_weight, style_features, args.style_weight,
    #     args.batch_size, args.tv_weight, args.learning_rate, args.max_epochs
    # )
    content_shape = (args.batch_size, args.content_image_size, args.content_image_size, 3)
    content_features = {}
    with tf.Graph().as_default(), tf.Session() as sess:

        # dataset = tf.data.Dataset.from_tensor_slices(train_images)
        # dataset = dataset.shuffle(100000)
        # dataset = dataset.map(map_function, num_parallel_calls=6)
        # dataset = dataset.batch(args.batch_size)
        # dataset = dataset.repeat()
        # iterator = dataset.make_one_shot_iterator()
        # next_element = iterator.get_next()

        x_content = tf.placeholder(tf.float32, content_shape, name="x_content")
        x_pre = vgg_tool.preprocess(x_content)
        content_net = vgg_tool.net(args.vgg_path, x_pre)  # 不经过生成网络
        content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER]

        preds = model_net.transform_net(x_content / 255.0)
        preds_pre = vgg_tool.preprocess(preds)

        net = vgg_tool.net(args.vgg_path, preds_pre)  # 经过生成网络
        content_size = _tensor_size(content_features[CONTENT_LAYER]) * args.batch_size
        assert _tensor_size(content_features[CONTENT_LAYER]) == _tensor_size(net[CONTENT_LAYER])
        content_loss = args.content_weight * (2 * tf.nn.l2_loss(
            net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) / content_size)

        style_loss = []
        for style_layer in STYLE_LAYERS:
            layer = net[style_layer]
            bs, height, width, filters = map(lambda i: i.value, layer.get_shape())
            size = height * width * filters
            feats = tf.reshape(layer, (bs, height * width, filters))
            feats_T = tf.transpose(feats, perm=[0, 2, 1])
            grams = tf.matmul(feats_T, feats) / size
            style_gram = style_features[style_layer]
            style_loss.append(2 * tf.nn.l2_loss(grams - style_gram) / style_gram.size)
        style_loss = args.style_weight * functools.reduce(tf.add, style_loss) / args.batch_size

        # total variation denoising
        tv_y_size = _tensor_size(preds[:, 1:, :, :])
        tv_x_size = _tensor_size(preds[:, :, 1:, :])
        y_tv = tf.nn.l2_loss(preds[:, 1:, :, :] - preds[:, :content_shape[1] - 1, :, :])
        x_tv = tf.nn.l2_loss(preds[:, :, 1:, :] - preds[:, :, :content_shape[2] - 1, :])
        tv_loss = args.tv_weight * 2 * (x_tv / tv_x_size + y_tv / tv_y_size) / args.batch_size

        loss = content_loss + style_loss + tv_loss

        train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(loss)
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        global_step = 0
        saver = tf.train.Saver()
        train_images = utils._get_files(args.train_path)
        mod = len(train_images) % args.batch_size
        if mod > 0:
            print("Train set has been trimmed slightly..")
            train_images = train_images[:-mod]
        for epoch in range(args.max_epochs):
            num_examples = len(train_images)
            iterations = 0
            while iterations < args.epoch_size:
                start_time = time.time()
                curr = iterations * args.batch_size
                step = curr + args.batch_size
                X_batch = np.zeros(content_shape, dtype=np.float32)
                for j, img_p in enumerate(train_images[curr:step]):
                    X_batch[j] = utils.get_img(
                        img_p, (args.content_image_size, args.content_image_size, 3)).astype(np.float32)
                # X_batch = sess.run(next_element)
                feed_dict = {x_content: X_batch}
                loss_, _ = sess.run([loss, train_step], feed_dict=feed_dict)
                run_time = time.time() - start_time
                print(global_step, loss_, run_time)
                iterations += 1
                global_step += 1
            saver.save(sess, "./save_model/wave.ckpt-{}".format(global_step))


if __name__ == '__main__':
    args = set_params.TrainOneStyle()
    main()
