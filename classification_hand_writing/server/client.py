import requests
import json
import base64
import cv2

image_name = './4.png'
img = cv2.imread(image_name)
jpg_img = cv2.imencode('.png', img)
b64_string = base64.b64encode(jpg_img[1]).decode('utf-8')

files = {
            "image": b64_string,
        }

r = requests.post("http://127.0.0.1:50000/api/predict", json=files)
print(r.json())