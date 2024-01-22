import base64
from io import BytesIO
import base64

import cv2
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
# 모듈 불러오기
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def load_model(model_path):
    # 모델 불러오기 및 예측
    model = torch.load(model_path)
    return model

model = load_model('../writing_model.pt')


if __name__ == '__main__':
    app = Flask(__name__)
    CORS(app)

    @app.route('/api/predict', methods=['POST'])
    def predictByModel():
        data = request.json
        image = data['image']

        # model input size = (28,28)
        # 이미지 전처리 과정?
        image = base64.b64decode(image)
        image = BytesIO(image)
        image = Image.open(image)
        image = image.convert('L')
        image = image.resize((28, 28))

        tf = transforms.ToTensor()
        image = tf(image)
        print(image.size())

        outputs = model.forward(image.unsqueeze(0))
        _, y_hat = outputs.max(1)
        prediction = y_hat.item()

        return jsonify(
            {
                'prediction' : prediction,
            }
        )
    app.run(host='0.0.0.0', port=50000, debug=True)

