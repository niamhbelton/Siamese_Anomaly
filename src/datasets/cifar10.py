from PIL import Image
from torchvision.datasets.utils import download_and_extract_archive, extract_archive, verify_str_arg, check_integrity
import torch.utils.data as data
import os
import pickle

import torch
import torchvision.transforms as transforms
import random
import numpy as np


class CIFAR10(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }


    def __init__(self, indexes, root: str, normal_class,
            task, data_path,
            download_data = False):
        super().__init__()

        self.task = task  # training set or test set
        self.data_path = data_path
        self.indexes = indexes
        self.normal_class = normal_class
        self.download_data = download_data



        if self.download_data:
            self.download()



        if (self.task == 'train') | (self.task == 'validate'):
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.data_path, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self.targets[self.targets != normal_class] = 1
        self.targets[self.targets == normal_class] = 0

        if self.task == 'train':
            self.data = self.data[self.indexes]
            print(self.targets)
            print(type(self.targets))
            self.targets = [x for i,x in enumerate(self.targets) if i in self.indexes]

          #  self.targets = self.targets[self.indexes]


        elif self.task == 'validate':
            lst = list(range(0,len(self.data) ))
            ind = [x for i,x in enumerate(lst) if i not in self.indexes]
            randomlist = random.sample(range(0, len(ind)), 10000)
            data = data[randomlist]
            targets = targets[randomlist]

        self._load_meta()


    def _load_meta(self) -> None:
        path = os.path.join(self.data_path, self.base_folder, self.meta['filename'])
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}


    def download(self) -> None:
        download_and_extract_archive(self.url, self.data_path, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")




    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target = self.data[index], int(self.targets[index])



        if self.task == 'train':
            ind = np.random.randint(len(self.indexes) + 1) -1
            while (ind == index):
                ind = np.random.randint(len(self.indexes) + 1) -1

            img2, target2 = self.data[ind], int(self.targets[ind])

            label = torch.FloatTensor([0])
        else:
            img2 = torch.Tensor([1])
            label = target


        return img, img2, label



    def __len__(self) -> int:
        return len(self.data)
