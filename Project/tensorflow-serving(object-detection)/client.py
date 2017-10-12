#coding:utf-8
import requests
import os

# image_array中可以存储本地路径或者URL

# 导入urls
# image_array = []

#单独传入本地图片
# image_array = ['./test2.jpg']

#从本地文件夹导入
image_array = []
for filename in os.listdir(r"./imgs"):
    path = r"./imgs/"+filename
    image_array.append(path)

images = str(image_array)

payload = {'img_urls':images}

url = 'http://127.0.0.1:8088/object-detection'



r = requests.post(url,payload)
print "代理返回结果。。。。。"
print(r.text)
#   with open("./data.json","w") as f:
#       json.dump(results,f)
#   log.info("写入文件完成。。。")
