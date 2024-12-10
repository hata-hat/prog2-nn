import time

import matplotlib.pyplot as plt

import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms

import models

#データセットの前処理を定義
ds_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True)
])

#データセットの読み込み
ds_train = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ds_transform
)
ds_test = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ds_transform
)

#ミニバッチに分割する dataloaderを作る
batch_size = 64
dataloader_train = torch.utils.data.DataLoader(
    ds_train,
    batch_size=batch_size,
    shuffle=True
)
dataloader_test = torch.utils.data.DataLoader(
    ds_test,
    batch_size=batch_size
)

#バッチを取り出す実験
# for image_batch, label_batch in dataloader_test:
#     print(image_batch.shape)
#     print(label_batch.shape)
#     break

#モデルのインスタンスを作成
model = models.MyModel()

#精度を計算
acc_test = models.test_accuracy(model, dataloader_test)
print(f'test accuracy: {acc_test*100:.2f}%')

acc_train = models.test_accuracy(model, dataloader_train)
print(f'train accuracy: {acc_train*100:.2f}%')


#損失関数(誤差関数、ロス関数)の選択
loss_fn = torch.nn.CrossEntropyLoss()

#最適化の方法の選択
learning_rate = 0.003
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#学習
#models.train(model, dataloader_train, loss_fn, optimizer)
#精度の再計算
#acc_test = models.test_accuracy(model, dataloader_test)
#print(f'test accuracy: {acc_test*100:.2f}%')

#学習回数
n_epochs = 5

loss_train_history = []
loss_test_history = []
acc_train_history = []
acc_test_history = []
a = time.time()
#学習
for k in range(n_epochs):
    print(f'epach {k+1}/{n_epochs}', end=': ', flush=True)
    
    loss_train = models.train(model, dataloader_train, loss_fn, optimizer)
    print(f'train loss: {loss_train:.3f}', end=', ', flush=True)
    loss_train_history.append(loss_train)

    loss_test = models.test(model, dataloader_test, loss_fn)
    print(f'test loss: {loss_test:.3f}', end=', ', flush=True)
    loss_test_history.append(loss_test)

    acc_train = models.test_accuracy(model, dataloader_train)
    print(f'train accuracy: {acc_train*100:.2f}%', end=', ', flush=True)
    acc_train_history.append(acc_train)

    acc_test = models.test_accuracy(model, dataloader_test)
    print(f'train accuracy: {acc_test*100:.2f}%', end=', ', flush=True)
    acc_test_history.append(acc_test)
    b = time.time()
    print(f'実行時間: {b-a:.2f}秒')
    a = b

plt.plot(acc_train_history, label='train')
plt.plot(acc_test_history, label='test')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.grid()
plt.show()

plt.plot(loss_train_history, label='train')
plt.plot(loss_test_history, label='test')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.grid()
plt.show()