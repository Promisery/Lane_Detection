import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import cv2
import os
from torchvision.datasets import ImageFolder
from torchvision import transforms

train_path = 'CULane/laneseg_label_w16/driver_23_30frame/'
eval_path = 'CULane/laneseg_label_w16/driver_182_30frame/'
test_path = 'CULane/laneseg_label_w16/driver_161_90frame/'

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
     ])

# class CULane_set(Dataset):

#     def __init__(self, path) -> None:
#         super(CULane_set, self).__init__()
#         self.path = path
#         self.vid_paths = os.listdir(path)
#         self.vids = [self.path + vid_path + '/' +
#                      os.listdir(path) for vid_path in self.vid_paths]

#     def __getitem__(self):


def get_dataset(dataset='train', transform=None, batch_size=1):
    if dataset == 'train':
        path = 'CULane/driver_23_30frame/'
        annotation = 'CULane/laneseg_label_w16/driver_23_30frame/'
    elif dataset == 'eval':
        path = 'CULane/driver_182_30frame/'
        annotation = 'CULane/laneseg_label_w16/driver_182_30frame/'
    elif dataset == 'test':
        path = 'CULane/driver_161_90frame/'
        annotation = 'CULane/laneseg_label_w16/driver_161_90frame/'
    else:
        print('Wrong type of dataset!')
        return None

    if transform is None:
        transform = transforms.ToTensor()
    dataset = ImageFolder(path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    annotation_dataset = ImageFolder(
        annotation, transform=transforms.ToTensor())
    annotation_dataloader = DataLoader(
        annotation_dataset, batch_size=batch_size, shuffle=False)

    data_iter = zip(dataloader, annotation_dataloader)
    return data_iter
