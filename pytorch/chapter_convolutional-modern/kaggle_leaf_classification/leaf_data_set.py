import torchvision
import torch
import pandas as pd
import torchvision.datasets
from torchvision.io import read_image
from sklearn.preprocessing import LabelEncoder
from typing import Callable, Optional, Tuple, Any
from PIL import Image

class LeafDataset(torchvision.datasets.VisionDataset):
    def __init__(self, data_path, 
                 train: bool=True, 
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) ->None:
        # super().__init__(data_path, transform=transform, target_transform= target_transform)
        self.train = train
        self.data_path = data_path
        self.category_dict = dict()
        self.rev_category_dict = dict()
        self.data_list = None
        self.le = None
        self.data, self.targets = self._load_data()
        self.transform = transform
        self.target_transform = target_transform
    
    def _load_data(self):
        data_list = pd.read_csv(self.data_path)
        data = data_list.image

        targets_test = None
        if self.train:
            self.le = LabelEncoder()
            targets_test = self.le.fit_transform(data_list['label'])

        targets = None
        if self.train:
            targets = torch.zeros(data_list.shape[0],1)
        
        category_num = 0
        for i in range(data_list.shape[0]):
            if self.train and self.category_dict.get(data_list.iloc[i].label) is None:
                self.category_dict[data_list.iloc[i].label] = category_num
                self.rev_category_dict[category_num] = data_list.iloc[i].label
                category_num = category_num + 1
            if targets is not None:
                targets[i] = torch.tensor([self.category_dict[data_list.iloc[i].label]])
        
        # targets = targets_test
        self.data_list = data_list
        return data, targets

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = Image.open(self.data[index])
        target = -1
        if self.train:
            target = int(self.targets[index])

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target