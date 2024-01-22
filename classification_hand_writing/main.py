# 현재 파일은 손글씨 분류 문제를 위한 파일입니다.

# 모듈 불러오기
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from matplotlib import pyplot as plt

# GPU 사용 여부
is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')

print('current device is :', device)

# 하이퍼 파라미터 지정하기
batch_size = 50
epoch_num = 15
learning_rate = 0.0001

# Mnist 데이터셋 불러오기
train_data = datasets.MNIST(root='../../dataset/hands_writing_data', train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST(root='../../dataset/hands_writing_data', train=False, download=True, transform=transforms.ToTensor())

print(len(train_data))
print(len(test_data))

# 미니배치 구성하기
train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)

first_batch = train_loader.__iter__().__next__()
print('{:15s} | {:<25s} | {}'.format('name', 'type', 'size'))

# 간단한 CNN을 사용한다.
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

# Optimizer 및 손실함수 정의
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
criterion = nn.CrossEntropyLoss()

## 모델 훈련 모드
model.train()
i = 0
for epoch in range(epoch_num):
    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if i % 1000 == 0:
            print('Train Step: {}\tLoss : {:.3f}'.format(i, loss.item()))
        i+=1

## 모델 평가 모드
# 모델 평가
model.eval()
correct = 0
for data, target in test_loader:
    data = data.to(device)
    target = target.to(device)
    output = model(data)
    prediction = output.data.max(1)[1]
    correct += prediction.eq(target.data).sum()
print('Test set: Accuracy: {:.2f}%'.format(100 * correct / len(test_loader.dataset)))

# 모델 저장
# 모델 저장하기

MODEL_SAVE_PATH = './writing_model.pt'
torch.save(model, MODEL_SAVE_PATH)

# 모델 불러오기 및 예측
model = torch.load(MODEL_SAVE_PATH)

model.eval()

def get_prediction(image_bytes):
    outputs = model.forward(image_bytes)
    _, y_hat = outputs.max(1)
    prediction = y_hat.item()
    return prediction

get_prediction(train_data[2][0].unsqueeze(0))