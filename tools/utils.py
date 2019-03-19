import os
import re

import cv2
import tensorflow as tf


def get_img(src, img_size=False):
    """
    读取单张图片
    :param src: 图片路径
    :param img_size: 图片的大小, ex:(256, 256)
    :return: 图片的矩阵
    """
    img = cv2.imread(src)
    if img_size:
        img = cv2.resize(img, img_size)
    return img


def list_files(in_path):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        files.extend(filenames)
        break

    return files


def get_files(img_dir):
    files = list_files(img_dir)
    return [os.path.join(img_dir, x) for x in files]


def get_learning_rate_from_file(filename, epoch):
    """
    从文件获取学习率
    :param filename: 文件名
    :param epoch: epoch
    :return:
    """
    learning_rate = 0.01
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split('#', 1)[0]
            if line:
                par = line.strip().split(':')
                e = int(par[0])
                if par[1] == '-':
                    lr = -1
                else:
                    lr = float(par[1])
                if e <= epoch:
                    learning_rate = lr
                else:
                    return learning_rate


def write_arguments_to_file(args, filename):
    with open(filename, 'w') as f:
        try:
            data = args.get_all_member()
            for key, value in data.items():
                if not callable(key) and not key.startswith("__") and not key.startswith("get_all_member"):
                    f.write('%s: %s\n' % (key, str(value)))
        except AttributeError:
            for key, value in vars(args).items():
                f.write('%s: %s\n' % (key, str(value)))


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file
