import torch
import numpy as np
import os

from utils import get_closest

class ClassificationDataSet(torch.utils.data.Dataset):
    
    def __init__(self, indices, transported_samples=None,target_bin=None, **kwargs):

        # The basic design here is that you pre-process and load all data as numpy arrays, as well as relevant indices into that array

        
        self.indices = indices # Indices are the indices of the elements from the arrays which are emitted by this data-loader
        self.transported_samples = transported_samples  # a 2-d array of OT maps
        
        self.root = kwargs['root_dir']
        self.device = kwargs['device'] if kwargs.get('device') else 'cpu'
        self.transport_idx_func = kwargs['transport_idx_func'] if kwargs.get('transport_idx_func') else lambda x:x%1000
        self.num_bins = kwargs['num_bins'] if kwargs.get('num_bins') else 6
        self.base_bin = kwargs['num_bins'] if kwargs.get('num_bins') else 0   # Minimum whole number value of U
        #self.num_bins = kwargs['num_bins']  # using this we can get the bin corresponding to a U value
        
        self.target_bin = target_bin
        self.X = np.load("{}/X.npy".format(self.root))
        self.Y = np.load("{}/Y.npy".format(self.root))
        self.A = np.load("{}/A.npy".format(self.root))
        self.U = np.load("{}/U.npy".format(self.root))
        self.drop_cols = kwargs['drop_cols_classifier'] if kwargs.get('drop_cols_classifier') else None
        
    def __getitem__(self,idx):

        index = self.indices[idx]
        data = torch.tensor(self.X[index]).float().to(self.device)   # Check if we need to reshape
        label = torch.tensor(self.Y[index]).long().to(self.device)
        auxiliary = torch.tensor(self.A[index]).float().to(self.device).view(-1, 1)
        domain = torch.tensor(self.U[index]).float().to(self.device).view(-1, 1)

        if self.drop_cols is not None:
            return data[:self.drop_cols], auxiliary, domain, label
        return data, auxiliary, domain, label

    def __len__(self):
        return len(self.indices)
