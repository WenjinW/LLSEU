"""
File        :
Description :
Author      :Wang Wenjin
Date        :2019/11/1
Version     :v1.0
"""
import torch
import torch.utils.data
from torchvision import datasets


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, datas, debug):
        super(MyDataset).__init__()
        if eval(debug):
            print("***************Debug*****************")
            datas['x'] = datas['x'][0:100]
            datas['y'] = datas['y'][0:100]
        self.data = datas
        self.length = self.data['x'].size()[0]

    def __getitem__(self, idx):
        x = self.data['x'][idx]
        y = self.data['y'][idx]

        return x, y

    def __len__(self):
        return self.length


class MyEMNIST(datasets.EMNIST):
    url = "https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip"
