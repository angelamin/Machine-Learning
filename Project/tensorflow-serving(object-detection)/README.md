## tensorflow serving docker环境

####docker环境准备
将tensorflow serving 映射
docker run -v $HOME:/mnt/home -p 9999:9999 -it triage/python2.7-tensorflow-serving

安装
apt-get update
apt-get install vim
easy_install pip
pip install tensorflow
pip install numpy
pip install flask
pip install requests
pip install tensorflow-serving-api

测试
bazel build -c opt tensorflow_serving/example/...

提交docker容器
docker commit 91951c2afd5f docker.pub.sina.com.cn:5000/tensorflow-serving-devel2

push到服务器
 docker push docker.pub.sina.com.cn:5000/tensorflow-serving-devel2:latest

#### 重新运行新镜像的容器,加载模型
docker run -v /usr/home/shixi_xiamin/object-detection:/mnt/object-detection-backup -p 8088:8088 -it  docker.pub.sina.com.cn:5000/tensorflow-serving-devel2:latest

docker run  -p 8088:8088 -it   docker.pub.sina.com.cn:5000/tensorflow-serving-devel2

生成tensorflow-serving所需的模型
cd /mnt/home/Desktop/code
python restore.py

运行model_servers
nohup tensorflow_model_server --port=9000 --model_name=objectdetection --model_base_path=/mnt/output/ &

运行客户端
python client.py --server=localhost:9000

#### 进行代理之后
启动docker
docker run  -p 8088:8088 -it   docker.pub.sina.com.cn:5000/tensorflow-serving-devel2

运行model_servers
nohup tensorflow_model_server --port=9000 --model_name=objectdetection --model_base_path=/mnt/output/ &

运行代理
nohup python agent.py --server=localhost:9000 &
挂载在服务器之后
nohup python agent.py --server=10.39.15.87:9000 &
将多个服务器应用之后
nohup python agent.py &

运行客户端
172.16.114.58
python client.py
