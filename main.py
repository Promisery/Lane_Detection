from dataloader import get_dataset, transform
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms

train_iter = get_dataset(
    'train', transform=transform, batch_size=1)

for idx, item in enumerate(train_iter):
    # print(item)
    data, annotation = item
    data, _ = data
    annotation, _ = annotation
    if idx % 100 == 0:
        print(idx)

print('finished')
