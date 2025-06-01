import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

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

X = torch.rand(1, 28, 28, device=device)
logits = model(X) 
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

input_image = torch.rand(3,28,28)
print(input_image.size())

flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)

softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)

print("Model structure: ", model, "\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

# 実行結果
"""
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
Predicted class: tensor([3], device='cuda:0')
torch.Size([3, 28, 28])
torch.Size([3, 784])
torch.Size([3, 20])
Before ReLU: tensor([[ 0.2049, -0.0423,  0.0281,  0.1448, -0.3428,  0.0447, -0.6004,  0.3005,
         -0.3728, -0.1028, -0.5047,  0.0646,  0.1845, -0.5534, -0.1144, -0.3416,
         -0.7914, -0.2016,  0.4336,  0.3024],
        [ 0.1361,  0.3259,  0.2139,  0.0092, -0.5555, -0.1904, -0.5703,  0.2857,
         -0.2655,  0.3153, -0.6420,  0.1840,  0.0047, -0.3474, -0.3192, -0.4385,
         -0.5014,  0.1644,  0.4961,  0.4412],
        [ 0.2065, -0.1541, -0.2501,  0.0812, -0.4267,  0.2381, -0.5183, -0.0163,
         -0.3174,  0.0842, -0.2126,  0.0465, -0.0026, -0.3550, -0.2951, -0.3066,
         -0.4321,  0.2515,  0.4237,  0.4028]], grad_fn=<AddmmBackward>)


After ReLU: tensor([[0.2049, 0.0000, 0.0281, 0.1448, 0.0000, 0.0447, 0.0000, 0.3005, 0.0000,
         0.0000, 0.0000, 0.0646, 0.1845, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.4336, 0.3024],
        [0.1361, 0.3259, 0.2139, 0.0092, 0.0000, 0.0000, 0.0000, 0.2857, 0.0000,
         0.3153, 0.0000, 0.1840, 0.0047, 0.0000, 0.0000, 0.0000, 0.0000, 0.1644,
         0.4961, 0.4412],
        [0.2065, 0.0000, 0.0000, 0.0812, 0.0000, 0.2381, 0.0000, 0.0000, 0.0000,
         0.0842, 0.0000, 0.0465, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2515,
         0.4237, 0.4028]], grad_fn=<ReluBackward0>)
Model structure:  NeuralNetwork(
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


Layer: linear_relu_stack.0.weight | Size: torch.Size([512, 784]) | Values : tensor([[-0.0114, -0.0273,  0.0352,  ...,  0.0337,  0.0117,  0.0350],
        [-0.0296, -0.0115, -0.0016,  ...,  0.0004, -0.0085,  0.0346]],
       device='cuda:0', grad_fn=<SliceBackward>) 

Layer: linear_relu_stack.0.bias | Size: torch.Size([512]) | Values : tensor([0.0092, 0.0134], device='cuda:0', grad_fn=<SliceBackward>) 

Layer: linear_relu_stack.2.weight | Size: torch.Size([512, 512]) | Values : tensor([[-0.0283,  0.0375,  0.0126,  ..., -0.0438, -0.0265, -0.0042],
        [ 0.0017, -0.0389, -0.0031,  ..., -0.0248, -0.0135,  0.0194]],
       device='cuda:0', grad_fn=<SliceBackward>) 

Layer: linear_relu_stack.2.bias | Size: torch.Size([512]) | Values : tensor([-0.0205, -0.0041], device='cuda:0', grad_fn=<SliceBackward>) 

Layer: linear_relu_stack.4.weight | Size: torch.Size([10, 512]) | Values : tensor([[ 0.0091,  0.0205,  0.0262,  ..., -0.0416, -0.0019, -0.0322],
        [ 0.0147, -0.0235, -0.0244,  ..., -0.0122, -0.0146,  0.0266]],
       device='cuda:0', grad_fn=<SliceBackward>) 

Layer: linear_relu_stack.4.bias | Size: torch.Size([10]) | Values : tensor([-0.0008, -0.0377], device='cuda:0', grad_fn=<SliceBackward>) 

"""