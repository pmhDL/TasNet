import os.path as osp
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


class DatasetLoader(Dataset):
    def __init__(self, setname, dataset_dir, backbone_class):
        THE_PATH = osp.join(dataset_dir, 'few-shot-' + setname + '.npz')
        data0 = np.load(THE_PATH)
        data = data0['features']
        label = data0['targets']

        self.data = data
        self.label = label
        self.num_class = len(np.unique(label))

        # Transformation
        image_size = 84
        transforms_list = [
            transforms.Resize(92),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]

        # Transformation
        if backbone_class == 'convnet':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])
        elif backbone_class == 'res12':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
                                     np.array([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]))
            ])
        elif backbone_class == 'res18':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        elif backbone_class == 'wrn28':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        label = self.label[i]
        image = self.transform(Image.fromarray(self.data[i].astype('uint8')).convert('RGB'))
        return image, label
