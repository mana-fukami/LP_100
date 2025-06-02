import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

# 訓練データをdatasetsからダウンロード
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# テストデータをdatasetsからダウンロード
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# データローダーの作成
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

# 訓練に際して、可能であればGPU（cuda）を設定します。GPUが搭載されていない場合はCPUを使用します
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# modelを定義します
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # 損失誤差を計算
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # バックプロパゲーション
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model)
print("Done!")

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')

# 実行結果
"""
Shape of X [N, C, H, W]:  torch.Size([64, 1, 28, 28])
Shape of y:  torch.Size([64]) torch.int64
Using cuda device
NeuralNetwork(
  (flatten): Flatten()
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
    (5): ReLU()
  )
)
Epoch 1
-------------------------------
loss: 2.305283  [    0/60000]
loss: 2.291645  [ 6400/60000]
loss: 2.283266  [12800/60000]
loss: 2.279228  [19200/60000]
loss: 2.249053  [25600/60000]
loss: 2.266302  [32000/60000]
loss: 2.240939  [38400/60000]
loss: 2.241901  [44800/60000]
loss: 2.251005  [51200/60000]
loss: 2.216974  [57600/60000]
Test Error: 
 Accuracy: 40.8%, Avg loss: 0.034782 

Epoch 2
-------------------------------
loss: 2.236450  [    0/60000]
loss: 2.203226  [ 6400/60000]
loss: 2.202556  [12800/60000]
loss: 2.201932  [19200/60000]
loss: 2.110125  [25600/60000]
loss: 2.186225  [32000/60000]
loss: 2.123379  [38400/60000]
loss: 2.135662  [44800/60000]
loss: 2.168199  [51200/60000]
loss: 2.090869  [57600/60000]
Test Error: 
 Accuracy: 42.8%, Avg loss: 0.032846 

Epoch 3
-------------------------------
loss: 2.143125  [    0/60000]
loss: 2.074858  [ 6400/60000]
loss: 2.078673  [12800/60000]
loss: 2.074430  [19200/60000]
loss: 1.901598  [25600/60000]
loss: 2.069647  [32000/60000]
loss: 1.946246  [38400/60000]
loss: 1.984476  [44800/60000]
loss: 2.051778  [51200/60000]
loss: 1.905491  [57600/60000]
Test Error: 
 Accuracy: 47.7%, Avg loss: 0.030173 

Epoch 4
-------------------------------
loss: 2.024219  [    0/60000]
loss: 1.911421  [ 6400/60000]
loss: 1.926339  [12800/60000]
loss: 1.900999  [19200/60000]
loss: 1.662901  [25600/60000]
loss: 1.947913  [32000/60000]
loss: 1.740148  [38400/60000]
loss: 1.831301  [44800/60000]
loss: 1.916929  [51200/60000]
loss: 1.705622  [57600/60000]
Test Error: 
 Accuracy: 50.4%, Avg loss: 0.027467 

Epoch 5
-------------------------------
loss: 1.902729  [    0/60000]
loss: 1.759549  [ 6400/60000]
loss: 1.787036  [12800/60000]
loss: 1.733298  [19200/60000]
loss: 1.460017  [25600/60000]
loss: 1.838942  [32000/60000]
loss: 1.561063  [38400/60000]
loss: 1.702528  [44800/60000]
loss: 1.784643  [51200/60000]
loss: 1.542662  [57600/60000]
Test Error: 
 Accuracy: 51.0%, Avg loss: 0.025205 

Done!
Saved PyTorch Model State to model.pth
Predicted: "Ankle boot", Actual: "Ankle boot"
"""