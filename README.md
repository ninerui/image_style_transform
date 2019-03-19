# image_style_transform
图片风格转换

## 训练数据集
训练数据集采用了 [coco train2014](http://cocodataset.org/#download)

## 训练
在 [set_params.py](set_params.py) 中的 TrainOneStyle 指定参数, 
然后运行 [train_one_style.py](train_one_style.py) 脚本开始进行训练

## 测试
运行 [test_one_image.py](test_one_image.py) 脚本可以查看生成图片的效果