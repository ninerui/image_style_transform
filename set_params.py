class TrainOneStyle:
    gpu_device = 0
    batch_size = 8
    epoch_size = 100
    max_epochs = 999
    output_dir = "../logs"

    content_image_size = 256
    vgg_path = "../data/imagenet-vgg-verydeep-19.mat"
    train_path = "../data/train2014"
    style_image = "./style_image/rain_princess.jpg"
    style_name = "rain_princess"

    content_weight = 7.5e0
    style_weight = 1e2
    tv_weight = 2e2

    learning_rate = -1
    learning_rate_file = r'./data/learning_rate_file.txt'


class TestOneImage:
    gpu_device = -1
    image_path = r'./test.jpg'
    ckpt_path = r'./save_model'
