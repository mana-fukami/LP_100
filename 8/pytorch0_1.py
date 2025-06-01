"""
「0. PyTorch入門(Learn the Basics)」の`[1]`~`[8]`までを写経し,AIXのサーバ上で,
コマンドライン(非JupiterNotebook, 非Google Colab)でGPUを使用して学習を行うこと
    ソースコードはGitHubにアップすること
ニューラルネットワークはtorch.nn.Moduleを継承してモデルクラスを作成すること
    「PyTorch入門 4. モデル構築」参照
データセットはカスタムしたDatasetクラスを自作し，DataLoaderを使ってデータを取り出すこと
    「PyTorch入門 2. データセットとデータローダー」参照
"""
import torch
import numpy as np

data=[[1,2],[3,4]]
x_data=torch.tensor(data)

np_array=np.array(data)
x_np=torch.from_numpy(np_array)

x_ones=torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")

x_rand=torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n {x_rand} \n")

shape=(2,3,)
rand_tensor=torch.rand(shape)
ones_tensor=torch.ones(shape)
zeros_tensor=torch.zeros(shape)
print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

tensor=torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

if torch.cuda.is_available():
    tensor=tensor.to("cuda")

tensor=torch.ones(4,4)
print('First row: ',tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[..., -1])
tensor[:,1] = 0
print(tensor)

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# 2つのテンソル行列のかけ算です。 y1, y2, y3 は同じ結果になります。
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)


# こちらは、要素ごとの積を求めます。 z1, z2, z3 は同じ値になります。
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

agg = tensor.sum()
agg_item = agg.item() # 数値型変数に変換
print(agg_item, type(agg_item))

print(tensor, "\n")
tensor.add_(5) # 末尾に'_'をつけることでtensorの内容そのものを更新する
print(tensor)

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

# tensorを変化させると、numpyも変化する→記述なしでも相互に参照し合う
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

n = np.ones(5)
t = torch.from_numpy(n)

# numpyの変化もtensorに影響する
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")