import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torchvision.models import ResNet,DenseNet, VGG, AlexNet
from torchvision.models.resnet import BasicBlock, Bottleneck, conv3x3
from torchvision import models
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import copy

from models.layers import GraphBN




###############################
##### Define Base Modules #####
###############################

class Identity(nn.Module):

    def __init__(self, sub=1.0):
        super(Identity, self).__init__()
        self.sub=sub

    def forward(self,x):
        return x*self.sub


class LockGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return 0.*grad_output

def grad_lock(x):
    return LockGrad.apply(x)



class DomainPredictor(nn.Module):

    def __init__(self, features=4096, domains=1):
        super(DomainPredictor, self).__init__()
        self.features=features
        self.domains=domains
        self.predictor=nn.Linear(features, domains)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0,0.001)

    def forward(self,x):
        x=self.predictor(F.relu(x))
        return x




###################################
##### Define AdaGraph AlexNet #####
###################################




class MLP_GraphBN(nn.Module):

    def __init__(self):
        
        super(MLP_GraphBN,self).__init__()
        # print(out_shape)


    def forward(self, X, times,logits=False,delta=0.0):
        # Note that we do not give time as an input here directly
        
        # if len(times.size()) == 3:  
        #   times = times.squeeze(1)

        # print(X.size(),times.size())

        # X = torch.cat([X, times.cuda().view(-1,1)], dim=1)
        pass




    def set_bn_from_edges(self,idx, ew=None):
        for m in self.modules():
            if isinstance(m, GraphBN):
                m.set_bn_from_edges(idx,ew=ew)


    def copy_source(self,idx):
        for m in self.modules():
            if isinstance(m, GraphBN):
                m.copy_source(idx)

    def reset_edges(self):
        for m in self.modules():
            if isinstance(m, GraphBN):
                m.reset_edges()


    def init_edges(self,edges):
        for m in self.modules():
            if isinstance(m, GraphBN):
                m.init_edges(edges)






class MLP_moons(MLP_GraphBN):
    """docstring for MLP_moons"""
    def __init__(self, out_shape, domains, data_shape=3, hidden_shape=128):
        super(MLP_moons, self).__init__()
        self.time_dim = 1

        self.layer_0 = nn.Linear(data_shape, hidden_shape)

        nn.init.kaiming_normal_(self.layer_0.weight)
        nn.init.zeros_(self.layer_0.bias)
        self.relu_0 = nn.ReLU() #TimeReLU(hidden_shape, self.time_dim, True,self.time_conditioning)
        self.bn0 = GraphBN(hidden_shape,domains=domains,dim=1)

        self.layer_1 = nn.Linear(hidden_shape, hidden_shape)
        nn.init.kaiming_normal_(self.layer_1.weight)
        nn.init.zeros_(self.layer_1.bias)
        self.relu_1 = nn.ReLU() #TimeReLU(hidden_shape, self.time_dim, True,self.time_conditioning)
        self.bn1 = GraphBN(hidden_shape,domains=domains,dim=1)

        self.layer_2 = nn.Linear(hidden_shape, hidden_shape)
        nn.init.kaiming_normal_(self.layer_2.weight)
        nn.init.zeros_(self.layer_2.bias)
        self.relu_2 = nn.ReLU() #TimeReLU(hidden_shape, self.time_dim, True,self.time_conditioning)
        self.bn2 = GraphBN(hidden_shape,domains=domains,dim=1)

        self.layer_3 = nn.Linear(hidden_shape, out_shape)
        nn.init.kaiming_normal_(self.layer_3.weight)
        nn.init.zeros_(self.layer_3.bias)

    def forward(self,X,times):
        X = self.bn0(self.relu_0(self.layer_0(X)),times)
        X = self.bn1(self.relu_1(self.layer_1(X)),times)
        # X = self.bn2(self.relu_2(self.layer_2(X)),times)
        X = self.layer_3(X)

        # X = X + self.delta_f * delta

        # if not logits:
            # X = torch.sigmoid(X)
        # print(X.size())
        return X




class MLP_house(MLP_GraphBN):
    """docstring for MLP_moons"""
    def __init__(self, out_shape, domains, data_shape=31, hidden_shape=400):
        super(MLP_house, self).__init__()
        self.time_dim = 1

        self.layer_0 = nn.Linear(data_shape, hidden_shape)

        nn.init.kaiming_normal_(self.layer_0.weight)
        nn.init.zeros_(self.layer_0.bias)
        self.relu_0 = nn.ReLU() #TimeReLU(hidden_shape, self.time_dim, True,self.time_conditioning)
        self.bn0 = GraphBN(hidden_shape,domains=domains,dim=1)

        self.layer_1 = nn.Linear(hidden_shape, hidden_shape)
        nn.init.kaiming_normal_(self.layer_1.weight)
        nn.init.zeros_(self.layer_1.bias)
        self.relu_1 = nn.ReLU() #TimeReLU(hidden_shape, self.time_dim, True,self.time_conditioning)
        self.bn1 = GraphBN(hidden_shape,domains=domains,dim=1)

        self.layer_2 = nn.Linear(hidden_shape, hidden_shape)
        nn.init.kaiming_normal_(self.layer_2.weight)
        nn.init.zeros_(self.layer_2.bias)
        self.relu_2 = nn.ReLU() #TimeReLU(hidden_shape, self.time_dim, True,self.time_conditioning)
        self.bn2 = GraphBN(hidden_shape,domains=domains,dim=1)

        self.layer_3 = nn.Linear(hidden_shape, hidden_shape)
        nn.init.kaiming_normal_(self.layer_3.weight)
        nn.init.zeros_(self.layer_3.bias)
        self.relu_3 = nn.ReLU() #TimeReLU(hidden_shape, self.time_dim, True,self.time_conditioning)
        self.bn3 = GraphBN(hidden_shape,domains=domains,dim=1)

        self.layer_4 = nn.Linear(hidden_shape, out_shape)
        nn.init.kaiming_normal_(self.layer_3.weight)
        nn.init.zeros_(self.layer_3.bias)

    def forward(self,X,times):
        X = self.bn0(self.relu_0(self.layer_0(X)),times)
        X = self.bn1(self.relu_1(self.layer_1(X)),times)
        X = self.bn2(self.relu_2(self.layer_2(X)),times)
        # X = self.bn3(self.relu_3(self.layer_3(X)),times)
        X = self.layer_4(X)

        # X = X + self.delta_f * delta

        # if not logits:
            # X = torch.sigmoid(X)
        # print(X.size())
        return X



class MLP_elec(MLP_GraphBN):
    """docstring for MLP_moons"""
    def __init__(self, out_shape, domains, data_shape=9, hidden_shape=64):
        super(MLP_elec, self).__init__()
        self.time_dim = 1

        self.layer_0 = nn.Linear(data_shape, hidden_shape)

        nn.init.kaiming_normal_(self.layer_0.weight)
        nn.init.zeros_(self.layer_0.bias)
        self.relu_0 = nn.ReLU() #TimeReLU(hidden_shape, self.time_dim, True,self.time_conditioning)
        self.bn0 = GraphBN(hidden_shape,domains=domains,dim=1)

        self.layer_1 = nn.Linear(hidden_shape, hidden_shape)
        nn.init.kaiming_normal_(self.layer_1.weight)
        nn.init.zeros_(self.layer_1.bias)
        self.relu_1 = nn.ReLU() #TimeReLU(hidden_shape, self.time_dim, True,self.time_conditioning)
        self.bn1 = GraphBN(hidden_shape,domains=domains,dim=1)

        self.layer_2 = nn.Linear(hidden_shape, hidden_shape)
        nn.init.kaiming_normal_(self.layer_2.weight)
        nn.init.zeros_(self.layer_2.bias)
        self.relu_2 = nn.ReLU() #TimeReLU(hidden_shape, self.time_dim, True,self.time_conditioning)
        self.bn2 = GraphBN(hidden_shape,domains=domains,dim=1)


        self.layer_3 = nn.Linear(hidden_shape, out_shape)
        nn.init.kaiming_normal_(self.layer_3.weight)
        nn.init.zeros_(self.layer_3.bias)

    def forward(self,X,times):
        X = self.bn0(self.relu_0(self.layer_0(X)),times)
        X = self.bn1(self.relu_1(self.layer_1(X)),times)
        X = self.bn2(self.relu_2(self.layer_2(X)),times)
        X = self.layer_3(X)

        # X = X + self.delta_f * delta

        # if not logits:
            # X = torch.sigmoid(X)
        # print(X.size())
        return X



class MLP_m5(MLP_GraphBN):
    """docstring for MLP_moons"""
    def __init__(self, out_shape, domains, data_shape=75, hidden_shape=50):
        super(MLP_m5, self).__init__()
        self.time_dim = 1

        self.layer_0 = nn.Linear(data_shape, hidden_shape)

        nn.init.kaiming_normal_(self.layer_0.weight)
        nn.init.zeros_(self.layer_0.bias)
        self.relu_0 = nn.LeakyReLU() #TimeReLU(hidden_shape, self.time_dim, True,self.time_conditioning)
        self.bn0 = GraphBN(hidden_shape,domains=domains,dim=1)

        self.layer_1 = nn.Linear(hidden_shape, hidden_shape)
        nn.init.kaiming_normal_(self.layer_1.weight)
        nn.init.zeros_(self.layer_1.bias)
        self.relu_1 = nn.LeakyReLU() #TimeReLU(hidden_shape, self.time_dim, True,self.time_conditioning)
        self.bn1 = GraphBN(hidden_shape,domains=domains,dim=1)

        # self.layer_2 = nn.Linear(hidden_shape, hidden_shape)
        # nn.init.kaiming_normal_(self.layer_2.weight)
        # nn.init.zeros_(self.layer_2.bias)
        # self.relu_2 = nn.ReLU() #TimeReLU(hidden_shape, self.time_dim, True,self.time_conditioning)
        # self.bn2 = GraphBN(hidden_shape,domains=domains,dim=1)


        self.layer_3 = nn.Linear(hidden_shape, out_shape)
        nn.init.kaiming_normal_(self.layer_3.weight)
        nn.init.zeros_(self.layer_3.bias)

    def forward(self,X,times):
        X = self.bn0(self.relu_0(self.layer_0(X)),times)
        X = self.bn1(self.relu_1(self.layer_1(X)),times)
        # X = self.bn2(self.relu_2(self.layer_2(X)),times)
        X = self.layer_3(X)

        # X = X + self.delta_f * delta

        # if not logits:
            # X = torch.sigmoid(X)
        # print(X.size())
        return X



class MLP_onp(MLP_GraphBN):
    """docstring for MLP_moons"""
    def __init__(self, out_shape, domains, data_shape=59, hidden_shape=200):
        super(MLP_onp, self).__init__()
        self.time_dim = 1

        self.layer_0 = nn.Linear(data_shape, hidden_shape)

        nn.init.kaiming_normal_(self.layer_0.weight)
        nn.init.zeros_(self.layer_0.bias)
        self.relu_0 = nn.LeakyReLU() #TimeReLU(hidden_shape, self.time_dim, True,self.time_conditioning)
        self.bn0 = GraphBN(hidden_shape,domains=domains,dim=1)

        self.layer_1 = nn.Linear(hidden_shape, hidden_shape)
        nn.init.kaiming_normal_(self.layer_1.weight)
        nn.init.zeros_(self.layer_1.bias)
        self.relu_1 = nn.LeakyReLU() #TimeReLU(hidden_shape, self.time_dim, True,self.time_conditioning)
        self.bn1 = GraphBN(hidden_shape,domains=domains,dim=1)

        # self.layer_2 = nn.Linear(hidden_shape, hidden_shape)
        # nn.init.kaiming_normal_(self.layer_2.weight)
        # nn.init.zeros_(self.layer_2.bias)
        # self.relu_2 = nn.ReLU() #TimeReLU(hidden_shape, self.time_dim, True,self.time_conditioning)
        # self.bn2 = GraphBN(hidden_shape,domains=domains,dim=1)


        self.layer_3 = nn.Linear(hidden_shape, out_shape)
        nn.init.kaiming_normal_(self.layer_3.weight)
        nn.init.zeros_(self.layer_3.bias)

    def forward(self,X,times):
        X = self.bn0(self.relu_0(self.layer_0(X)),times)
        # X = self.bn1(self.relu_1(self.layer_1(X)),times)
        # X = self.bn2(self.relu_2(self.layer_2(X)),times)
        X = self.layer_3(X)

        # X = X + self.delta_f * delta

        # if not logits:
            # X = torch.sigmoid(X)
        # print(X.size())
        return X



##############################
##### INSTANTIATE MODELS #####
##############################

def get_graph_mlp(classes=3, domains=30, url=None,dataset=None):
    print(dataset)
    if dataset == "house":
        model=MLP_house(out_shape=classes,domains=domains)

    if dataset == "moons":
        model=MLP_moons(out_shape=classes,domains=domains)


    if dataset == "elec":
        model=MLP_elec(out_shape=classes,domains=domains)


    if dataset == "m5house":
        model=MLP_m5(out_shape=classes,domains=domains)


    if dataset == "onp":
        model=MLP_onp(out_shape=classes,domains=domains)

    return model
