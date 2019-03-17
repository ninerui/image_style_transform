class TrainOneStyle:
    gpu_device = 0
    batch_size = 8
    epoch_size = 2000
    max_epochs = 999

    content_image_size = 256
    vgg_path = "../data/imagenet-vgg-verydeep-19.mat"
    train_path = "../data/train2014"
    style_image = "./style_image/wave.jpg"

    content_weight = 7.5e0
    style_weight = 1e2
    tv_weight = 2e2

    learning_rate = 0.001
    learning_file = r'./data/learning_rate_file.txt'


class TestOneImage:
    gpu_device = -1
    image_path = r'E:\win10_directorys_bak\Pictures\桌面壁纸\2a67b16eddc451da1612cc13b6fd5266d2163290.jpg'
    ckpt_path = r'./save_model'
