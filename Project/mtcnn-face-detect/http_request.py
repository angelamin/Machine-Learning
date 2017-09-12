#coding:utf-8
import requests
import os

# image_urls_array中可以存储本地路径或者URL

# 导入urls
# image_urls_array = ["https://ss3.bdstatic.com/70cFv8Sh_Q1YnxGkpoWK1HF6hhy/it/u=3995761121,34697482&amp;fm=27&amp;gp=0.jpg","http://img00.hc360.com/cs/201309/201309301340524588.jpg","http://img.qqzhi.com/upload/img_4_2232071970D3116521394_23.jpg","http://img95.699pic.com/photo/2016/08/26/f27bbfbc-7b49-4e07-98b7-62284f2a4f41.jpg_wh860.jpg","http://seopic.699pic.com/photo/2016/08/26/b52f1bcc-bbc2-4f05-8e29-54c3234e0de0.jpg_wh1200.jpg","http://seopic.699pic.com/photo/2016/08/28/bb5fdf43-ce15-4e79-a99e-a45606ca2421.jpg_wh1200.jpg","http://seopic.699pic.com/photo/2016/08/28/b21dc4b0-1f06-4886-8a9c-8b3310001662.jpg_wh1200.jpg","http://seopic.699pic.com/photo/2016/08/26/649e96a3-62c8-4312-894c-5f4f47927f20.jpg_wh1200.jpg"]

#单独传入本地图片
# image_urls_array = ['./test2.jpg']

#从本地文件夹导入
image_urls_array = []
for filename in os.listdir(r"./imgs"):
    path = r"./imgs/"+filename
    image_urls_array.append(path)

print image_urls_array
print len(image_urls_array)

image_urls = str(image_urls_array)

payload = {'img_urls':image_urls}

url = 'http://127.0.0.1:8080/mtcnn'

# for i in range(0, 10000):
r = requests.post(url, payload)
print "服务器返回结果..."
print(r.text)
