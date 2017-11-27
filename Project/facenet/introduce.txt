align_dataset_mtcnn.py shezheng shezheng_face 检测人脸（输入文件夹，输出文件夹）
detect_face.py  检测人脸

classifier.py  识别align之后的人物
  use_split_dataset=True  自动分割训练和测试集
  mode  TRAIN CLASSIFY
  model  模型文件位置
  classifier_filename 分类模型文件名


将facenet放到tensorflow_serving中，以便能够很好的用到GPU资源
首先将模型恢复成tensorflow serving能够识别的格式，然后再运行serving服务，此服务能够固定返回值embedding
python restore.py






python align_dataset_mtcnn.py shezheng shezheng_face


python align_dataset_mtcnn.py shezheng1 shezheng_face1

python classifier.py TRAIN  ../shezheng_face ../model ../classifier.pkl  --use_split_dataset

python classifier.py CLASSIFY  ../shezheng_face ../model ../classifier.pkl  --use_split_dataset

#docker中用tensorflow serving环境

docker run -ti -v /Users/xiamin/Downloads/Machine-Learning/Project/facenet:/facenet -p 1109:1109 triage/python2.7-tensorflow-serving:latest

nohup tensorflow_model_server --port=9000 --model_name=face_recognization --model_base_path=/facenet/restored_model &

#安装相关依赖后保存
docker commit 5ab7b40a1785 triage/python2.7-tensorflow-serving:latest

python server.py

curl http://127.0.0.1:1108/\?img_url\=http://www.gov.cn/xinwen/2017-01/23/5162726/images/b4defae138804f19bed6d0d4543724d9.jpg
# docker中
curl http://172.17.0.2:1109/\?img_url\=http://www.gov.cn/xinwen/2017-01/23/5162726/images/b4defae138804f19bed6d0d4543724d9.jpg
