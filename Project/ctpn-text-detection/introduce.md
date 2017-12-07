USE_GPU_NMS # whether to use nms implemented in cuda,if you do not have a gpu device,follow here to setup 见wiki
DETECT_MODE # H represents horizontal mode, O represents oriented mode, default is H      ctpn/text.yml

split_label.py  将人工标记的文字部分转换为模型识别的标记方式
ctpn/text.yml  超参数设置


# training
### 环境准备
> pip3 install tensorflow==1.3.0       
pip3 install Cython==0.24.1
pip3 install opencv-python
pip3 install easydict
pip3 install pillow
pip3 install pytesseract
pip3 install -r requirements.txt

```
gpu运行，将58环境代码同步到gpu

wget

nvidia-docker run -ti -v /data1/shixi_xiamin/ctpn_crnn:/ctpn_crnn_in docker.pub.sina.com.cn:5000/tensorflow_gpu:xiamin       
```

```
将gpu代码同步到58
cp -r /ctpn_crnn_in /ctpn_crnn

docker commit 1fb415e5b14b docker.pub.sina.com.cn:5000/tensorflow_gpu:xiamin    

docker push docker.pub.sina.com.cn:5000/tensorflow_gpu:xiamin    

```

```
将镜像内代码同步到58
docker pull docker.pub.sina.com.cn:5000/tensorflow_gpu:xiamin

docker run -ti docker.pub.sina.com.cn:5000/tensorflow_gpu:xiamin  

docker cp ef1343032b94:/ctpn_crnn /data1/Project/shixi_xiamin/ctpn_crnn

python pyftpserver.py
```

```
将58上代码同步到本地
zip -q -r /usr/home/shixi_xiamin/ctpn_crnn.zip ctpn_crnn
zip -q -r /data1/Project/shixi_xiamin/ctpn_crnn.zip ctpn_crnn
python pyftpserver.py

wget
```

显示图片问题
将gpu上的镜像pull到58上，然后从镜像中通过docker cp导回到本机中

### prepare data

1. download the pre-trained model of VGG net and put it in data/pretrain/VGG_imagenet.npy. you can download it from google drive or baidu yun.
https://drive.google.com/drive/folders/0B_WmJoEtfQhDRl82b1dJTjB2ZGc

2. Second, prepare the training data as referred in paper, or you can download the data I prepared from previous link. Or you can prepare your own data according to the following steps.

3. Modify the path and gt_path in prepare_training_data/split_label.py according to your dataset. And run

  ```
  cd prepare_training_data
  python split_label.py
  ```
  it will generate the prepared data in current folder,
4. and then run

  ```
  python ToVoc.py

  ```
  to convert the prepared training data into voc format. It will generate a folder named TEXTVOC. move this folder to data/
5. and then run
  ```  
  cd ../data
  ln -s TEXTVOC VOCdevkit2007
  ```


### train
1. cpu

```
vim /Users/xiamin/virtualenv/python3/lib/python2.7/distutils/distutils.cfg
[build_ext]
include_dirs= /Users/xiamin/virtualenv/tensorflow-py2/lib/python2.7/site-packages/numpy/core/include
或
vim /Users/xiamin/virtualenv/python3/lib/python3.6/distutils/distutils.cfg
[build_ext]
include_dirs=
/Users/xiamin/virtualenv/python3/lib/python3.6/site-packages/numpy/core/include
```

```
  python setup-cpu.py build
```
  然后把build文件夹下面生成的两个so文件拷贝到lib/utils就可以

  ```
  python train_net.py
  ```
2. gpu
  ```
  python train_net.py
  ```
  - you can modify some hyper parameters in ctpn/text.yml, or just used the parameters I set.
  - The model I provided in checkpoints is trained on GTX1070 for 50k iters.
  - If you are using cuda nms, it takes about 0.2s per iter. So it will takes about 2.5 hours to finished 50k iterations.

```
python demo.py
```





# CRNN
Use tensorflow to implement a Deep Neural Network for scene text recognition mainly based on the paper "An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition".You can refer to their paper for details http://arxiv.org/abs/1507.05717. Thanks for the author [Baoguang Shi](https://github.com/bgshih).  
This model consists of a CNN stage, RNN stage and CTC loss for scene text recognition task.

## Installation
This software has only been tested on ubuntu 16.04(x64), python3.5, cuda-8.0, cudnn-6.0 with a GTX-1070 GPU. To install this software you need tensorflow 1.3.0 and other version of tensorflow has not been tested but I think it will be able to work properly in tensorflow above version 1.0. Other required package you may install them by

```
pip3 install -r requirements.txt
```

## Test model
In this repo I uploaded a model trained on a subset of the [Synth 90k](http://www.robots.ox.ac.uk/~vgg/data/text/). During data preparation process the dataset is converted into a tensorflow records which you can find in the data folder.
You can test the trained model on the converted dataset by

```
python tools/test_shadownet.py --dataset_dir data/ --weights_path model/shadownet/shadownet_2017-09-29-19-16-33.ckpt-39999
```
`Expected output is`  
![Test output](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/test_output.png)
If you want to test a single image you can do it by
```
python tools/demo_shadownet.py --image_path data/test_images/test_01.jpg --weights_path model/shadownet/shadownet_2017-09-29-19-16-33.ckpt-39999
```
`Example image_01 is`  
![Example image1](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/text_example_image1.png)  
`Expected output_01 is`  
![Example image1 output](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/test_example_image1_output.png)  
`Example image_02 is`  
![Example image2](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/test_example_image2.png)  
`Expected output_02 is`  
![Example image2 output](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/test_example_image2_output.png)  

## Train your own model
#### Data Preparation
Firstly you need to store all your image data in a root folder then you need to supply a txt file named sample.txt to specify the relative path to the image data dir and it's corresponding text label. For example

```
path/1/2/373_coley_14845.jpg coley
path/17/5/176_Nevadans_51437.jpg nevadans
```

Secondly you are supposed to convert your dataset into tensorflow records which can be done by
```
python tools/write_text_features --dataset_dir path/to/your/dataset --save_dir path/to/tfrecords_dir
```
All your training image will be scaled into (32, 100, 3) the dataset will be divided into train, test, validation set and you can change the parameter to control the ratio of them.

#### Train model
The whole training epoches are 40000 in my experiment. I trained the model with a batch size 32, initialized learning rate is 0.1 and decrease by multiply 0.1 every 10000 epochs. For more training parameters information you can check the global_configuration/config.py for details. To train your own model by

```
python tools/train_shadownet.py --dataset_dir path/to/your/tfrecords
```
You can also continue the training process from the snapshot by
```
python tools/train_shadownet.py --dataset_dir path/to/your/tfrecords --weights_path path/to/your/last/checkpoint
```
After several times of iteration you can check the log file in logs folder you are supposed to see the following contenent
![Training log](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/train_log.png)
The seq distance is computed by calculating the distance between two saparse tensor so the lower the accuracy value is the better the model performs.The train accuracy is computed by calculating the character-wise precision between the prediction and the ground truth so the higher the better the model performs.

During my experiment the `loss` drops as follows  
![Training loss](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/train_loss.png)
The `distance` between the ground truth and the prediction drops as follows  
![Sequence distance](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/seq_distance.png)

## Experiment
The accuracy during training process rises as follows  
![Training accuracy](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/training_accuracy.md)

## TODO
The model is trained on a subet of [Synth 90k](http://www.robots.ox.ac.uk/~vgg/data/text/). So i will train a new model on the whold dataset to get a more robust model.The crnn model needs large of training data to get a rubust model.
