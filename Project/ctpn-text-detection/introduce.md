USE_GPU_NMS # whether to use nms implemented in cuda,if you do not have a gpu device,follow here to setup 见wiki
DETECT_MODE # H represents horizontal mode, O represents oriented mode, default is H      ctpn/text.yml


demo.py  测试demo
text-recognization.py 上线检测版本(ctpn+OCR)

# ctpn
## 环境准备
> pip3 install tensorflow==1.3.0       
pip3 install Cython==0.24.1
pip3 install opencv-python
pip3 install easydict
pip3 install pillow
pip3 install pytesseract
pip3 install -r requirements.txt
pip install tqdm
```
gpu运行，将58环境代码同步到gpu

wget

nvidia-docker run -ti -p 1880:1880 -v /data1/shixi_xiamin:/shixi_xiamin docker.pub.sina.com.cn:5000/tensorflow_gpu:xiamin       

```    


```
将gpu代码同步到58
cp -r /shixi_xiamin/ctpn_side-refinement /ctpn_side-refinement

docker commit d5b24d2e3110 docker.pub.sina.com.cn:5000/tensorflow_gpu:xiamin    

docker push docker.pub.sina.com.cn:5000/tensorflow_gpu:xiamin    

```

```
将镜像内代码同步到58
docker pull docker.pub.sina.com.cn:5000/tensorflow_gpu:xiamin

docker run -ti docker.pub.sina.com.cn:5000/tensorflow_gpu:xiamin  

docker cp 692e9c00a030:/ctpn_side-refinement /data1/Project/shixi_xiamin/ctpn_side-refinement

python pyftpserver.py
```

```
将58上代码同步到本地
zip -q -r /usr/home/shixi_xiamin/ctpn_side-refinement.zip /shixi_xiamin/ctpn_side-refinement
zip -q -r /data1/Project/shixi_xiamin/ctpn_side-refinement.zip ctpn_side-refinement
python pyftpserver.py

wget
```

显示图片问题
将gpu上的镜像pull到58上，然后从镜像中通过docker cp导回到本机中

## prepare data

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


## train
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

下载 Multilingual scene text 数据集
prepare_training_data/split_label.py  转换label,将label标签按照宽16px的形式标注

python ToVoc.py  generate a folder named TEXTVOC. move this folder to data/
cd ../data
ln -s TEXTVOC VOCdevkit2007

CUDA_VISIBLE_DEVICES=0,1 python train_net_ctpn.py

CUDA_VISIBLE_DEVICES=0,1 python text-recognization-not-service.py

## 代码结构
text.yml  超参数设置

split_label.py  将人工标记的文字部分转换为模型识别的标记方式
Tovoc.py   数据转换

VGGnet_train.py 网络结构
network.py中包含 VGGnet_train.py中声明的函数





# CRNN
## prepare data
a txt file named sample.txt to specify the relative path to the image data dir and it's corresponding text label. For example

```
path/1/2/373_coley_14845.jpg coley
path/17/5/176_Nevadans_51437.jpg nevadans
```


Secondly you are supposed to convert your dataset into tensorflow records which can be done by

```
python tools/write_text_features --dataset_dir path/to/your/dataset --save_dir path/to/tfrecords_dir
```

python tools/write_text_features.py --dataset_dir /shixi_xiamin/CRNN_Tensorflow-chinese_version_debug/data --save_dir  /shixi_xiamin/CRNN_Tensorflow-chinese_version_debug/data/output_tfrecords --batch_size=32

## train model
The whole training epoches are 40000 in my experiment. I trained the model with a batch size 32, initialized learning rate is 0.1 and decrease by multiply 0.1 every 10000 epochs. For more training parameters information you can check the global_configuration/config.py for details. To train your own model by

```
python tools/train_shadownet.py --dataset_dir path/to/your/tfrecords
```

You can also continue the training process from the snapshot by
```
python tools/train_shadownet.py --dataset_dir path/to/your/tfrecords --weights_path path/to/your/last/checkpoint
```
