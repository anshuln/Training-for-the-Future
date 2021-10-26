import numpy as np
import os
import json
from torch.utils import data
from random import shuffle
from torch.utils import data as Data
from utils import read_pickle
from utils import write_pickle
'''
class toydata(data.Dataset):
    def __init__(self, fname, l_domain_id, normalize=False, l_domain_mask=None):
        d_pkl = read_pickle(fname)
        self.get_data(d_pkl, l_domain_id, l_domain_mask)
        self.l_domain_mask = l_domain_mask

        self.normalize = normalize
        if self.normalize:
            self.normalize_data(l_domain_id)
#         print(self.domain)

    def normalize_data(self, l_domain_id):
        self.data_norm = np.zeros(self.data.shape).astype('float32')
        for domain in l_domain_id:
            mean = np.mean(self.data[self.domain == domain, :], axis=0, keepdims=True)
            self.data_norm[self.domain == domain, :] = self.data[self.domain == domain, :] - mean

    def get_data(self, d_pkl, l_domain_id, l_domain_mask=None):
        l_data = []
        l_label = []
        l_domain = []
        l_mask = []
        if l_domain_mask is not None:
            for domain, mask in zip(l_domain_id, l_domain_mask):
                m_data = d_pkl['data'][d_pkl['domain'] == domain, :]
                m_label = d_pkl['label'][d_pkl['domain'] == domain]
                m_domain = np.ones(m_label.shape) * domain

                l_data.append(m_data)
                l_label.append(m_label)
                l_domain.append(m_domain)

                m_mask = np.ones(m_label.shape) * mask
                l_mask.append(m_mask)
        else:
            for domain in l_domain_id:
                m_data = d_pkl['data'][d_pkl['domain'] == domain, :]
                m_label = d_pkl['label'][d_pkl['domain'] == domain]
                m_domain = np.ones(m_label.shape) * domain

                l_data.append(m_data)
                l_label.append(m_label)
                l_domain.append(m_domain)

        self.data = np.concatenate(l_data, axis=0).astype('float32')
        self.label = np.concatenate(l_label, axis=0).astype('int64')
#         print('bingo l_domain', l_domain)
        self.domain = np.concatenate(l_domain, axis=0).astype('float32')

        if l_domain_mask is not None:
#             print('bingo l_mask', l_mask)
            self.mask = np.concatenate(l_mask, axis=0).astype('float32')

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        if self.l_domain_mask is None:
            if self.normalize:
                return self.data[index, :], self.label[index], self.domain[index], self.data_norm[index, :]
            else:
                return self.data[index, :], self.label[index], self.domain[index]
        else:
            if self.normalize:
                return self.data[index, :], self.label[index], self.domain[index], self.data_norm[index, :], self.mask[index]
            else:
                return self.data[index, :], self.label[index], self.domain[index], self.mask[index]

'''
def read_data(root):
    X = np.load("{}/X.npy".format(root))
    Y = np.load("{}/Y.npy".format(root))
    U = np.load("{}/U.npy".format(root))
    indices = json.load(open("{}/indices.json".format(root)))

    return X,U,Y,indices

class classifdata(data.Dataset):
    def __init__(self, fname, l_domain_id, normalize=False, l_domain_mask=None):
        X,U,Y,indices = read_data(fname)
        self.get_data(X,U,Y,indices, l_domain_id, l_domain_mask)
        self.l_domain_mask = l_domain_mask

        self.normalize = normalize
        if self.normalize:
            self.normalize_data(l_domain_id)
#         print(self.domain)

    def normalize_data(self, l_domain_id):
        self.data_norm = np.zeros(self.data.shape).astype('float32')
        for domain in l_domain_id:
            mean = np.mean(self.data[self.domain == domain, :], axis=0, keepdims=True)
            self.data_norm[self.domain == domain, :] = self.data[self.domain == domain, :] - mean

    def get_data(self, X,U,Y,indices, l_domain_id, l_domain_mask=None):
        l_data = []
        l_label = []
        l_domain = []
        l_mask = []
        if l_domain_mask is not None:
            for domain, mask in zip(l_domain_id, l_domain_mask):
                ind = indices[domain]
                m_data = X[ind, :]
                m_label = Y[ind]
                m_domain = np.ones(m_label.shape) * domain

                l_data.append(m_data)
                l_label.append(m_label)
                l_domain.append(m_domain)

                m_mask = np.ones(m_label.shape) * mask
                l_mask.append(m_mask)
        else:
            for domain in l_domain_id:
                ind = indices[domain]
                m_data = X[ind, :]
                m_label = Y[ind]
                m_domain = np.ones(m_label.shape) * domain

                l_data.append(m_data)
                l_label.append(m_label)
                l_domain.append(m_domain)

        self.data = np.concatenate(l_data, axis=0).astype('float32')
        self.label = np.concatenate(l_label, axis=0).astype('float32')
#         print('bingo l_domain', l_domain)
        self.domain = np.concatenate(l_domain, axis=0).astype('float32')

        if l_domain_mask is not None:
#             print('bingo l_mask', l_mask)
            self.mask = np.concatenate(l_mask, axis=0).astype('float32')

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        if self.l_domain_mask is None:
            if self.normalize:
                return self.data[index, :], self.label[index], self.domain[index], self.data_norm[index, :]
            else:
                return self.data[index, :], self.label[index], self.domain[index]
        else:
            if self.normalize:
                return self.data[index, :], self.label[index], self.domain[index], self.data_norm[index, :], self.mask[index]
            else:
                return self.data[index, :], self.label[index], self.domain[index], self.mask[index]
'''
def read_m5_data(root):
    X = np.load("{}/X.npy".format(root))
    Y = np.load("{}/Y.npy".format(root))
    U = np.load("{}/U.npy".format(root))
    indices = json.load(open("{}/indices.json".format(root)))

    return X,U,Y,indices

class M5(data.Dataset):
    def __init__(self, fname, l_domain_id, normalize=False, l_domain_mask=None):
        X,U,Y,indices = read_moons_data(fname)
        self.get_data(X,U,Y,indices, l_domain_id, l_domain_mask)
        self.l_domain_mask = l_domain_mask

        self.normalize = normalize
        if self.normalize:
            self.normalize_data(l_domain_id)
#         print(self.domain)

    def normalize_data(self, l_domain_id):
        self.data_norm = np.zeros(self.data.shape).astype('float32')
        for domain in l_domain_id:
            mean = np.mean(self.data[self.domain == domain, :], axis=0, keepdims=True)
            self.data_norm[self.domain == domain, :] = self.data[self.domain == domain, :] - mean

    def get_data(self, X,U,Y,indices, l_domain_id, l_domain_mask=None):
        l_data = []
        l_label = []
        l_domain = []
        l_mask = []
        if l_domain_mask is not None:
            for domain, mask in zip(l_domain_id, l_domain_mask):
                ind = indices[domain]
                m_data = X[ind, :]
                m_label = Y[ind]
                m_domain = np.ones(m_label.shape) * domain

                l_data.append(m_data)
                l_label.append(m_label)
                l_domain.append(m_domain)

                m_mask = np.ones(m_label.shape) * mask
                l_mask.append(m_mask)
        else:
            for domain in l_domain_id:
                ind = indices[domain]
                m_data = X[ind, :]
                m_label = Y[ind]
                m_domain = np.ones(m_label.shape) * domain

                l_data.append(m_data)
                l_label.append(m_label)
                l_domain.append(m_domain)

        self.data = np.concatenate(l_data, axis=0).astype('float32')
        self.label = np.concatenate(l_label, axis=0).astype('float32')
#         print('bingo l_domain', l_domain)
        self.domain = np.concatenate(l_domain, axis=0).astype('float32')

        if l_domain_mask is not None:
#             print('bingo l_mask', l_mask)
            self.mask = np.concatenate(l_mask, axis=0).astype('float32')

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        if self.l_domain_mask is None:
            if self.normalize:
                return self.data[index, :], self.label[index], self.domain[index], self.data_norm[index, :]
            else:
                return self.data[index, :], self.label[index], self.domain[index]
        else:
            if self.normalize:
                return self.data[index, :], self.label[index], self.domain[index], self.data_norm[index, :], self.mask[index]
            else:
                return self.data[index, :], self.label[index], self.domain[index], self.mask[index]


def read_moons_data(root):
    X = np.load("{}/X.npy".format(root))
    Y = np.load("{}/Y.npy".format(root))
    U = np.load("{}/U.npy".format(root))
    indices = json.load(open("{}/indices.json".format(root)))

    return X,U,Y,indices

class Moonsdata(data.Dataset):
    def __init__(self, fname, l_domain_id, normalize=False, l_domain_mask=None):
        X,U,Y,indices = read_moons_data(fname)
        self.get_data(X,U,Y,indices, l_domain_id, l_domain_mask)
        self.l_domain_mask = l_domain_mask

        self.normalize = normalize
        if self.normalize:
            self.normalize_data(l_domain_id)
#         print(self.domain)

    def normalize_data(self, l_domain_id):
        self.data_norm = np.zeros(self.data.shape).astype('float32')
        for domain in l_domain_id:
            mean = np.mean(self.data[self.domain == domain, :], axis=0, keepdims=True)
            self.data_norm[self.domain == domain, :] = self.data[self.domain == domain, :] - mean

    def get_data(self, X,U,Y,indices, l_domain_id, l_domain_mask=None):
        l_data = []
        l_label = []
        l_domain = []
        l_mask = []
        if l_domain_mask is not None:
            for domain, mask in zip(l_domain_id, l_domain_mask):
                ind = indices[domain]
                m_data = X[ind, :]
                m_label = Y[ind]
                m_domain = np.ones(m_label.shape) * domain

                l_data.append(m_data)
                l_label.append(m_label)
                l_domain.append(m_domain)

                m_mask = np.ones(m_label.shape) * mask
                l_mask.append(m_mask)
        else:
            for domain in l_domain_id:
                ind = indices[domain]
                m_data = X[ind, :]
                m_label = Y[ind]
                m_domain = np.ones(m_label.shape) * domain

                l_data.append(m_data)
                l_label.append(m_label)
                l_domain.append(m_domain)

        self.data = np.concatenate(l_data, axis=0).astype('float32')
        self.label = np.concatenate(l_label, axis=0).astype('float32')
#         print('bingo l_domain', l_domain)
        self.domain = np.concatenate(l_domain, axis=0).astype('float32')

        if l_domain_mask is not None:
#             print('bingo l_mask', l_mask)
            self.mask = np.concatenate(l_mask, axis=0).astype('float32')

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        if self.l_domain_mask is None:
            if self.normalize:
                return self.data[index, :], self.label[index], self.domain[index], self.data_norm[index, :]
            else:
                return self.data[index, :], self.label[index], self.domain[index]
        else:
            if self.normalize:
                return self.data[index, :], self.label[index], self.domain[index], self.data_norm[index, :], self.mask[index]
            else:
                return self.data[index, :], self.label[index], self.domain[index], self.mask[index]

'''