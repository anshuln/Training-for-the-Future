import torch
import itertools

from configs.opts import *
from src.train import *
from src.test import *
from models.networks import get_network
import random
import os 
import torch 
import copy
import numpy as np
import argparse

def safe_print(x):
    try:
        print(x)
    except:
        return








seed=SEED
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed) 
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
    
# INSTANTIATE TRAINING

NUM_DOMS=1
for i in DOMAINS:
    NUM_DOMS*=len(i)

# LOAD NETWORK
print("Total Domains",NUM_DOMS)
net = get_network(CLASSES, NUM_DOMS, residual=RESIDUAL,mlp=MLP,dataset=DATASET)
net = net.to(DEVICE)

meta_vectors = torch.FloatTensor(NUM_DOMS,NUM_META).fill_(0)
edge_vals=torch.FloatTensor(NUM_DOMS,NUM_DOMS).fill_(0)
edge_vals_no_self=torch.FloatTensor(NUM_DOMS,NUM_DOMS).fill_(0)
full_list=[]

for meta in itertools.product(*DOMAINS):
        full_list.append(meta)
        meta_vectors[domain_converter(meta)]=get_meta_vector(meta)

for i,vector in enumerate(meta_vectors):
        edge_vals[i,:]=compute_edge(vector,meta_vectors,i,1.)
        edge_vals_no_self[i,:]=compute_edge(vector,meta_vectors,i,0.)

EXP=NUM_DOMS*(NUM_DOMS-1)

res_source=[]
res_refined=[]
res_upperbound=[]
res_upperbound_ref=[]
res_adagraph=[]
res_adagraph_refinement=[]




upperbound_loader=init_loader(BATCH_SIZE, domains=full_list, auxiliar= True, size=SIZE, std=STD)

net_std=copy.deepcopy(net).to(DEVICE)
source_loader = init_loader(BATCH_SIZE, domains=[[x] for x in SOURCE_DOMAINS], shuffle=True, auxiliar=True, size=SIZE, std=STD)

net_std.reset_edges()
print("Training for : ",EPOCHS)
training_loop(net_std, source_loader, SOURCE_DOMAINS, epochs=EPOCHS, training_group=SOURCE_GROUP, store=None, auxiliar=False,regression=REGRESSION)

net_upperbound=copy.deepcopy(net_std)
net_upperbound.init_edges(edge_vals)

print("Training batchnorm")
training_loop(net_upperbound,upperbound_loader, SOURCE_DOMAINS,epochs=7, training_group=TRAINING_GROUP, store=None, auxiliar=False,regression=REGRESSION)


current_edges=copy.deepcopy(edge_vals)
net_adagraph=copy.deepcopy(net_std)
net_adagraph.init_edges(current_edges)

auxiliar_loader=init_loader(BATCH_SIZE, domains=[[x] for x in SOURCE_DOMAINS], auxiliar=True, size=SIZE, std=STD)

training_loop(net_adagraph,auxiliar_loader, SOURCE_DOMAINS, target_idx=-1, epochs=13, training_group=TRAINING_GROUP, store=None, auxiliar=True,regression=REGRESSION)


target_loader = init_loader(BATCH_SIZE, domains=[[ACTUAL_TARGET_DOMAIN]], shuffle=True, auxiliar=False, size=SIZE, std=STD)
test_loader = init_loader(TEST_BATCH_SIZE, domains=[[ACTUAL_TARGET_DOMAIN]], shuffle=False, auxiliar=False, size=SIZE, std=STD)

current_res_source = test(net_std, test_loader, ACTUAL_TARGET_DOMAIN,regression=REGRESSION)
# print(edge_vals_no_self)
net_adagraph.set_bn_from_edges(ACTUAL_TARGET_DOMAIN, ew=edge_vals_no_self[ACTUAL_TARGET_DOMAIN,:])
net_adagraph.init_edges(edge_vals)

net_refined = copy.deepcopy(net_std)

current_res_adagraph = test(net_adagraph, test_loader, ACTUAL_TARGET_DOMAIN,regression=REGRESSION)
current_res_upperbound = test(net_upperbound, test_loader, ACTUAL_TARGET_DOMAIN,regression=REGRESSION)


res_source.append(current_res_source)
res_adagraph.append(current_res_adagraph)
res_upperbound.append(current_res_upperbound)


log = open("logs/log_adagraph_{}".format(DATASET),"a")
print("Seed - {}".format(seed),file=log)
print("Num_dom - {}".format(len(SOURCE_DOMAINS)),file=log)
print("Performance - {}".format(np.mean(np.array(res_adagraph))),file=log)
print("Performance Upper - {}".format(np.mean(np.array(res_upperbound))),file=log)

log.close()
