import os
import sys

import cv2
import numpy as np
import scipy.misc

# def get_img(src):
#     image = cv2.imread(src)
#     assert image is not None
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     return image

def get_img(src, img_size=False):
   img = scipy.misc.imread(src, mode='RGB') # misc.imresize(, (256, 256, 3))
   if not (len(img.shape) == 3 and img.shape[2] == 3):
       img = np.dstack((img,img,img))
       print (img.shape)
   if img_size != False:
       img = scipy.misc.imresize(img, img_size)
   return img


def list_files(in_path):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        files.extend(filenames)
        break

    return files


def _get_files(img_dir):
    files = list_files(img_dir)
    # count = 0
    # res = []
    # for i in files:
    #     a = os.path.join(img_dir, i)
    #     if cv2.imread(a).shape[2] == 3:
    #         res.append(a)
    #     sys.stdout.write("\r{}".format(count))
    #     count += 1
    return [os.path.join(img_dir, x) for x in files]
    # return res
