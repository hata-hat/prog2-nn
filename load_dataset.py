import matplotlib.pyplot as plt
from torchvision import datasets
import torchvision.transforms.v2 as transforms
import torch

ds_train = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
)

print(f'numbers of datasets:{len(ds_train)}')

image, target = ds_train[0]
print(type(image), target)
for i in range(5):
    for j in range(5):
        k = i*5+j
        image, target = ds_train[k+200]
        plt.subplot(5, 5, k+1)
        plt.imshow(image, cmap='gray_r')
        plt.title(target)
        
plt.show()

#トーチテンソール
print('PIL画像をtorch.tensorに変換')
image_tensor = transforms.functional.to_image(image)
image = transforms.functional.to_dtype(image_tensor, dtype=torch.float32, scale=True)
print(image_tensor.shape,image_tensor.dtype)
print(image.shape, image.dtype)
print(image.min(), image.max())