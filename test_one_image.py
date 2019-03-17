import os

import tensorflow as tf
import numpy as np
import scipy
import matplotlib.pyplot as plt

import set_params
from tools import utils, model_net


def main():
    if str(args.gpu_device):  # 指定gpu设备
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device)
    img = utils.get_img(args.image_path)
    img_shape = img.shape
    with tf.Graph().as_default():
        with tf.Session() as sess:
            img_placeholder = tf.placeholder(tf.float32, (1,) + img_shape)
            preds = model_net.transform_net(img_placeholder)
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(args.ckpt_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            _preds = sess.run(preds, feed_dict={img_placeholder: [img]})
            img = np.clip(_preds[0], 0, 255).astype(np.uint8)
            plt.imshow(img)
            plt.show()
            # scipy.misc.imsave(out_path, img)


if __name__ == '__main__':
    args = set_params.TestOneImage()
    main()
