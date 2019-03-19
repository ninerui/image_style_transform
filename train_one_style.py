# 导入系统库
import os
import sys
import time
import functools
from datetime import datetime

# 导入第三方库
import numpy as np
import tensorflow as tf

# 导入自己的脚本
import set_params
from tools import utils, model_net, vgg_tool

# vgg_net需要提取的网络名
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'


def _tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)


def main():
    # 指定训练设备
    if str(args.gpu_device):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device)
    # 获取风格图像的特征图五张
    style_image = utils.get_img(args.style_image)
    style_features = model_net.get_style_features(args.vgg_path, style_image)

    content_shape = (args.batch_size, args.content_image_size, args.content_image_size, 3)
    content_features = {}

    with tf.Session() as sess:
        learning_rate = tf.placeholder(tf.float32, name="learning_rate")

        x_content = tf.placeholder(tf.float32, content_shape, name="x_content")
        x_pre = vgg_tool.preprocess(x_content)
        # 不经过生成网络得到特征图的值
        content_net = vgg_tool.net(args.vgg_path, x_pre)
        content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER]

        # 经过生成网络获取生成的图片
        preds = model_net.transform_net(x_content / 255.)
        # preds = model_net.transform_net_keras(x_content / 255.)
        preds_pre = vgg_tool.preprocess(preds)
        net = vgg_tool.net(args.vgg_path, preds_pre)  # 经过生成网络

        # 计算内容图的size, 风格图有除以.size
        content_size = _tensor_size(content_features[CONTENT_LAYER]) * args.batch_size
        assert _tensor_size(content_features[CONTENT_LAYER]) == _tensor_size(net[CONTENT_LAYER])
        # 计算内容图的loss
        content_loss = args.content_weight * (2 * tf.nn.l2_loss(
            net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) / content_size)

        # 计算风格图每一个feature的loss
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

        # total variation denoising, 计算tv loss
        tv_y_size = _tensor_size(preds[:, 1:, :, :])
        tv_x_size = _tensor_size(preds[:, :, 1:, :])
        y_tv = tf.nn.l2_loss(preds[:, 1:, :, :] - preds[:, :content_shape[1] - 1, :, :])
        x_tv = tf.nn.l2_loss(preds[:, :, 1:, :] - preds[:, :, :content_shape[2] - 1, :])
        tv_loss = args.tv_weight * 2 * (x_tv / tv_x_size + y_tv / tv_y_size) / args.batch_size

        loss = content_loss + style_loss + tv_loss
        tf.summary.scalar('losses/total_loss', loss)
        tf.summary.scalar('losses/content_loss', content_loss)
        tf.summary.scalar('losses/style_loss', style_loss)
        tf.summary.scalar('losses/tv_loss', tv_loss)
        tf.summary.scalar('learning_rate', learning_rate)

        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        # 创建日志保存目录
        subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
        output_dir = os.path.join(os.path.expanduser(args.output_dir), subdir + "_" + args.style_name)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        model_dir = os.path.join(output_dir, 'save_models')
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        # utils.write_arguments_to_file(args, os.path.join(output_dir, 'arguments.txt'))

        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=2)
        merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(output_dir, sess.graph)

        train_images = utils.get_files(args.train_path)
        file_nums = len(train_images)
        file_count = 0
        global_step = 0
        for epoch in range(args.max_epochs):
            if args.learning_rate > 0.0:
                lr = args.learning_rate
            else:
                lr = utils.get_learning_rate_from_file(args.learning_rate_file, epoch)
            if lr <= 0:
                return False
            batch_number = 0
            while batch_number < args.epoch_size:
                start_time = time.time()
                curr = file_count * args.batch_size
                step = curr + args.batch_size
                if step > file_nums:
                    file_count = 0
                    curr = file_count * args.batch_size
                    step = curr + args.batch_size
                x_batch = np.zeros(content_shape, dtype=np.float32)
                for j, img_p in enumerate(train_images[curr:step]):
                    x_batch[j] = utils.get_img(
                        img_p, (args.content_image_size, args.content_image_size)).astype(np.float32)
                feed_dict = {x_content: x_batch, learning_rate: lr}
                if batch_number % 100 == 0:
                    loss_, _, summary_str_ = sess.run([loss, train_step, merged], feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str_, global_step)
                else:
                    loss_, _ = sess.run([loss, train_step], feed_dict=feed_dict)
                run_time = time.time() - start_time
                sys.stdout.write(
                    "\repoch: {:d}\tglobal_step: {:d}\ttotal_loss: {:f}\tlr: {}\trun_time: {:f}".format(
                        epoch, global_step, loss_, lr, run_time))
                file_count += 1
                batch_number += 1
                global_step += 1
            print("\t\tsave model...")
            checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % args.style_name)
            saver.save(sess, checkpoint_path, global_step=global_step, write_meta_graph=False)
            metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % args.style_name)
            if not os.path.exists(metagraph_filename):
                print('Saving metagraph')
                saver.export_meta_graph(metagraph_filename)


if __name__ == '__main__':
    args = set_params.TrainOneStyle()
    main()
