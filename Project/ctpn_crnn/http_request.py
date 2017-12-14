#coding:utf-8
import requests
import os

payload = {'image_url':'./7.jpg'}

# url = 'http://127.0.0.1:8080/adFilter'
url = 'http://10.39.6.204:1889/textRecognization'
# url = 'http://localhost:1889/textRecognization'

for i in range(1000):
    r = requests.post(url, payload)
    print("results from the server........")
    print(r.text)
