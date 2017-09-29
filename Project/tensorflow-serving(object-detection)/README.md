# 目的：

将模型通过tensor flow serving 封装为服务

restore.py

首先，通过模型的保存与恢复，恢复过程导出tensorflow serving 能用的模型，需要指定输入输出

client.py

对模型按照导出时相似的输入进行测试，返回需要的结果

## tensorflow serving docker环境

将tensorflow serving 映射
>docker run -v $HOME:/mnt/home -p 9999:9999 -it triage/python2.7-tensorflow-serving

>apt-get update
>apt-get install vim
>easy_install pip
>pip install tensorflow
>pip install numpy
>pip install tensorflow-serving-api

#### 测试
>bazel build -c opt tensorflow_serving/example/...

docker环境中已经有了model_servers无需安装
安装model_servers
>bazel build -c opt tensorflow_serving/model_servers:tensorflow_model_server

#### ModelServer运行
>bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server

#### 提交docker容器
>docker commit 0e9cd40d2080 mytensorflowserving

#### 重新运行新镜像的容器
>docker run -v $HOME:/mnt/home -p 9999:9999 -it mytensorflowserving

#### 生成tensorflow-serving所需的模型
>cd /mnt/home/Desktop/code
>python restore.py

#### 运行model_servers
> nohup tensorflow_model_server --port=9000 --model_name=objectdetection --model_base_path=/mnt/output/ &

#### 运行客户端
>python client.py --server=localhost:9000



