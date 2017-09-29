# 完成工作
app.py
以多线程的方式，
将人脸识别封装为服务，并将所有结果存为json格式放入data.json中

http_request.py
发起请求，检测传入的图片的人脸

draw.py
拿到识别出来的矩形框以及对应的五点特征，将这些特征从data.json中独取出来，在不需要的mxnet环境下，在本地画出来标记
# 人脸识别mtcnn docker环境

docker pull raohuaming/mxnet-opencv-dlib

docker run -ti -v /Users/xiamin/Downloads/Machine-Learning/Project/mtcnn-face-detect:/mtcnn raohuaming/mxnet-opencv-dlib:latest

easy_install pip;pip install Flask;pip install requests

python app.py &

sh ./install.sh

python http_request.py
