import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

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

model = NeuralNetwork()

learning_rate = 1e-3
batch_size = 64
epochs = 5

# loss functionの初期化、定義
loss_fn = nn.CrossEntropyLoss()

"""
Optimizer: モデルパラメータを調整する
"""
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):        
        # 予測と損失の計算
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # バックプロパゲーション
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

# 実行結果
"""
Epoch 1
-------------------------------
loss: 2.300830  [    0/60000]
loss: 2.297529  [ 6400/60000]
loss: 2.287312  [12800/60000]
loss: 2.287802  [19200/60000]
loss: 2.277025  [25600/60000]
loss: 2.251451  [32000/60000]
loss: 2.259195  [38400/60000]
loss: 2.241493  [44800/60000]
loss: 2.253369  [51200/60000]
loss: 2.214656  [57600/60000]
Test Error: 
 Accuracy: 30.7%, Avg loss: 0.035134 

Epoch 2
-------------------------------
loss: 2.242979  [    0/60000]
loss: 2.250427  [ 6400/60000]
loss: 2.221810  [12800/60000]
loss: 2.234164  [19200/60000]
loss: 2.200473  [25600/60000]
loss: 2.144086  [32000/60000]
loss: 2.178388  [38400/60000]
loss: 2.137850  [44800/60000]
loss: 2.177115  [51200/60000]
loss: 2.085247  [57600/60000]
Test Error: 
 Accuracy: 31.5%, Avg loss: 0.033695 

Epoch 3
-------------------------------
loss: 2.160517  [    0/60000]
loss: 2.174405  [ 6400/60000]
loss: 2.122163  [12800/60000]
loss: 2.153355  [19200/60000]
loss: 2.073704  [25600/60000]
loss: 1.986211  [32000/60000]
loss: 2.057021  [38400/60000]
loss: 1.990133  [44800/60000]
loss: 2.072078  [51200/60000]
loss: 1.915163  [57600/60000]
Test Error: 
 Accuracy: 31.5%, Avg loss: 0.031841 

Epoch 4
-------------------------------
loss: 2.051497  [    0/60000]
loss: 2.077999  [ 6400/60000]
loss: 2.011113  [12800/60000]
loss: 2.059331  [19200/60000]
loss: 1.932741  [25600/60000]
loss: 1.837177  [32000/60000]
loss: 1.924431  [38400/60000]
loss: 1.858574  [44800/60000]
loss: 1.959748  [51200/60000]
loss: 1.756533  [57600/60000]
Test Error: 
 Accuracy: 31.7%, Avg loss: 0.030176 

Epoch 5
-------------------------------
loss: 1.945482  [    0/60000]
loss: 1.992871  [ 6400/60000]
loss: 1.923669  [12800/60000]
loss: 1.975332  [19200/60000]
loss: 1.819183  [25600/60000]
loss: 1.730449  [32000/60000]
loss: 1.812216  [38400/60000]
loss: 1.764620  [44800/60000]
loss: 1.868604  [51200/60000]
loss: 1.630421  [57600/60000]
Test Error: 
 Accuracy: 32.8%, Avg loss: 0.028496 

Epoch 6
-------------------------------
loss: 1.865475  [    0/60000]
loss: 1.888528  [ 6400/60000]
loss: 1.810195  [12800/60000]
loss: 1.841795  [19200/60000]
loss: 1.681765  [25600/60000]
loss: 1.638602  [32000/60000]
loss: 1.688911  [38400/60000]
loss: 1.673152  [44800/60000]
loss: 1.784908  [51200/60000]
loss: 1.520203  [57600/60000]
Test Error: 
 Accuracy: 39.9%, Avg loss: 0.026675 

Epoch 7
-------------------------------
loss: 1.793894  [    0/60000]
loss: 1.775456  [ 6400/60000]
loss: 1.694045  [12800/60000]
loss: 1.723832  [19200/60000]
loss: 1.573012  [25600/60000]
loss: 1.554023  [32000/60000]
loss: 1.599307  [38400/60000]
loss: 1.598737  [44800/60000]
loss: 1.707963  [51200/60000]
loss: 1.442770  [57600/60000]
Test Error: 
 Accuracy: 41.4%, Avg loss: 0.025357 

Epoch 8
-------------------------------
loss: 1.731850  [    0/60000]
loss: 1.693330  [ 6400/60000]
loss: 1.612727  [12800/60000]
loss: 1.641081  [19200/60000]
loss: 1.499893  [25600/60000]
loss: 1.491665  [32000/60000]
loss: 1.537959  [38400/60000]
loss: 1.553160  [44800/60000]
loss: 1.655284  [51200/60000]
loss: 1.390849  [57600/60000]
Test Error: 
 Accuracy: 42.6%, Avg loss: 0.024475 

Epoch 9
-------------------------------
loss: 1.682135  [    0/60000]
loss: 1.641048  [ 6400/60000]
loss: 1.558425  [12800/60000]
loss: 1.584876  [19200/60000]
loss: 1.449516  [25600/60000]
loss: 1.445948  [32000/60000]
loss: 1.493909  [38400/60000]
loss: 1.522149  [44800/60000]
loss: 1.614458  [51200/60000]
loss: 1.353219  [57600/60000]
Test Error: 
 Accuracy: 43.8%, Avg loss: 0.023829 

Epoch 10
-------------------------------
loss: 1.639716  [    0/60000]
loss: 1.603553  [ 6400/60000]
loss: 1.517574  [12800/60000]
loss: 1.543680  [19200/60000]
loss: 1.411413  [25600/60000]
loss: 1.409852  [32000/60000]
loss: 1.460323  [38400/60000]
loss: 1.498415  [44800/60000]
loss: 1.580529  [51200/60000]
loss: 1.324543  [57600/60000]
Test Error: 
 Accuracy: 44.9%, Avg loss: 0.023322 

Done!
"""