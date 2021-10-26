import torch
import torch.utils.data as data
from torchvision.datasets.folder import  has_file_allowed_extension, is_image_file, IMG_EXTENSIONS, pil_loader, accimage_loader,default_loader

from PIL import Image

import sys
import os
import os.path
import numpy as np
from random import shuffle


REGIONS_DICT={'Alabama': 'South', 'Arizona': 'SW',
 'California': 'Pacific',
 'Florida': 'South',
 'Indiana': 'MW',
 'Iowa': 'MW',
 'Kansas': 'MW',
 'Massachusetts': 'NE',
 'Michigan': 'MW',
 'Missouri': 'South',
 'Montana': 'RM',
 'New-York': 'MA',
 'North-Carolina': 'South',
 'Ohio': 'MW',
 'Oklahoma': 'SW',
 'Oregon': 'Pacific',
 'Pennsylvania': 'MA',
 'South-Carolina': 'South',
 'South-Dakota': 'MW',
 'Texas': 'SW',
 'Utah': 'RM',
 'Vermont': 'NE',
 'Virginia': 'South',
 'Washington': 'Pacific',
 'Wyoming': 'RM'}

REGIONS_TO_IDX={'RM': 6,'MA': 1,'NE': 2,'South': 3, 'Pacific': 4, 'MW': 0 , 'SW': 5}
IDX_TO_REGIONS={ 6:'RM',1:'MA',2:'NE',3:'South',4: 'Pacific', 0:'MW', 5:'SW'}


def make_dataset(dir, class_to_idx, extensions, domains,start=1934):
    images = []
    meta = []

    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    year=int(path.split('/')[-1].split('_')[0])
                    city=(path.split('/')[-1].split('_')[1])
                    region=REGIONS_DICT[city]
                    pivot_year=start+(year-start)//10*10

                    if (pivot_year, region) in domains:
                        item = (path, class_to_idx[target])
                        images.append(item)
                        meta.append([year,region])

    return images, meta



class ONP(data.Dataset):

    def __init__(self, root, transform=None, target_transform=None,domains=[]):
        extensions = IMG_EXTENSIONS
        loader = default_loader

        # classes, class_to_idx = self._find_classes(root)
        # samples, self.meta = make_dataset(root, class_to_idx, extensions, domains)
        # if len(samples) == 0:
        #   raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
        #                      "Supported extensions are: " + ",".join(extensions)))

        self.root = root

        X = np.load("{}/X.npy".format(self.root))
        Y = np.load("{}/Y.npy".format(self.root))
        A = np.load("{}/A.npy".format(self.root))
        U = np.load("{}/U.npy".format(self.root))

        # print(domains)

        U_ = (U).astype('d')
        # print(U_.max())
        indices = []
        for d in domains:
            # print(d)
            indices += [i for i, x in enumerate(U_) if x == d[0]]
            # print(len(indices))
        self.X = X[indices]
        self.Y = Y[indices]
        self.U = U[indices]
        self.A = A[indices]
        self.loader = loader
        # self.extensions = extensions

        # self.classes = classes
        # self.class_to_idx = class_to_idx
        # self.samples = samples

        # self.transform = transform
        # self.target_transform = target_transform

        # self.imgs = self.samples


    def _find_classes(self, dir):

        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        # path, target = self.samples[index]
        sample = self.X[index]
        target = self.Y[index]
        # print(sample.shape)
        # if self.transform is not None:
        #   sample = self.transform(sample)
        # if self.target_transform is not None:
        #   target = self.target_transform(target)
        y,p = self.U[index], self.A[index]
        # print(targe)
        return np.concatenate([sample,p.reshape(1)]).astype('f'), int(y), target#.reshape(1)

    def get_meta(self):
        return np.array(self.meta)

    def __len__(self):
        return len(self.X)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class ONPSampler(torch.utils.data.sampler.Sampler):
    r"""Base class for all Samplers.

    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """

    def __init__(self, data_source, bs):
        self.data_source=data_source
        self.meta=self.data_source.U
        self.dict_meta={}
        self.indeces={}
        self.keys=[]
        self.bs=bs
        for idx, u in enumerate(self.meta):
            try:
                self.dict_meta[u].append(idx)
            except:
                self.dict_meta[u]=[idx]
                self.keys.append(u)
                self.indeces[u]=0

        for idx in self.keys:
            shuffle(self.dict_meta[idx])

    def _sampling(self,idx, n):
        if self.indeces[idx]+n>=len(self.dict_meta[idx]):
            self.dict_meta[idx]=self.dict_meta[idx]+self.dict_meta[idx]
        self.indeces[idx]=self.indeces[idx]+n
        return self.dict_meta[idx][self.indeces[idx]-n:self.indeces[idx]]



    def _shuffle(self):
        order=np.random.randint(len(self.keys),size=(len(self.data_source)//(self.bs)))
        sIdx=[]
        for i in order:
            sIdx=sIdx+self._sampling(self.keys[i],self.bs)
        return np.array(sIdx)


    def __iter__(self):
        return iter(self._shuffle())

    def __len__(self):
        return len(self.data_source)/self.bs*self.bs

