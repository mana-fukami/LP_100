import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

"""
grad_fn: 勾配が格納される
"""
print('Gradient function for z =',z.grad_fn)
print('Gradient function for loss =', loss.grad_fn)

"""
損失関数の偏微分を求める → loss.backward()を実行してから
backward()は基本1回のみ計算可能。retain_graph=Trueを引数として渡せば、複数可能
"""
loss.backward()
print(w.grad)
print(b.grad)

"""
torch.no_grad(): 勾配計算を不要にする。
    ―訓練済みモデルでの推論
    ―ファインチューニング
    ―順伝搬の計算を高速にしたい
"""
z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)

z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)

"""
grad.zero_()を実行しないと、勾配はどんどん足し合わせて蓄積される
    → optimizerが勾配のリセットを担ってくれる
"""
inp = torch.eye(5, requires_grad=True)
out = (inp+1).pow(2)
out.backward(torch.ones_like(inp), retain_graph=True)
print("First call\n", inp.grad)
out.backward(torch.ones_like(inp), retain_graph=True)
print("\nSecond call\n", inp.grad)
inp.grad.zero_()
out.backward(torch.ones_like(inp), retain_graph=True)
print("\nCall after zeroing gradients\n", inp.grad)

# 実行結果
"""
Gradient function for z = <AddBackward0 object at 0x7f24a8afcf90>
Gradient function for loss = <BinaryCrossEntropyWithLogitsBackward object at 0x7f24a8afcf90>
tensor([[0.2834, 0.1624, 0.0042],
        [0.2834, 0.1624, 0.0042],
        [0.2834, 0.1624, 0.0042],
        [0.2834, 0.1624, 0.0042],
        [0.2834, 0.1624, 0.0042]])
tensor([0.2834, 0.1624, 0.0042])
True
False
False
First call
 tensor([[4., 2., 2., 2., 2.],
        [2., 4., 2., 2., 2.],
        [2., 2., 4., 2., 2.],
        [2., 2., 2., 4., 2.],
        [2., 2., 2., 2., 4.]])

Second call
 tensor([[8., 4., 4., 4., 4.],
        [4., 8., 4., 4., 4.],
        [4., 4., 8., 4., 4.],
        [4., 4., 4., 8., 4.],
        [4., 4., 4., 4., 8.]])

Call after zeroing gradients
 tensor([[4., 2., 2., 2., 2.],
        [2., 4., 2., 2., 2.],
        [2., 2., 4., 2., 2.],
        [2., 2., 2., 4., 2.],
        [2., 2., 2., 2., 4.]])
"""