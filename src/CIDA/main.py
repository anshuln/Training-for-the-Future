from easydict import EasyDict
import argparse

from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import os
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
from utils import masked_mse, masked_cross_entropy, gaussian_loss, plain_log, write_pickle, read_pickle
from torch.utils import data
from data_loader import classifdata
from sklearn.metrics import accuracy_score, classification_report

label_noise_std = 0.20
use_label_noise = False
use_inverse_weighted = True
discr_thres = 999.999
normalize = True
train_discr_step_tot = 2
train_discr_step_extra = 0
slow_lrD_decay = 1

train_list = [0,1,2,3,4,5,6,7,8,9,10] 
mask_list =  [1]*11  
test_list =  [11]    


def seed_torch(args, seed):
    if args.cuda:
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



class ClassificationDataSet(torch.utils.data.Dataset):
    
    def __init__(self, indices, transported_samples=None,target_bin=None, **kwargs):
        '''
        TODO: Handle OT
        Pass Transported samples as kwargs?
        '''
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
        if self.transported_samples is not None:
            source_bin = int(np.round(domain.item() * self.num_bins)) # - self.base_bin
            # print(source_bin,self.target_bin)
            transported_X = torch.from_numpy(self.transported_samples[source_bin][self.target_bin][self.transport_idx_func(idx)]).float().to(self.device) #This should be similar to index fun, an indexing function which takes the index of the source sample and returns the corresponding index of the target sample.
            # print(source_bin,self.target_bin,transported_X.size())
            if self.drop_cols is not None:
                return data[:self.drop_cols],transported_X[:self.drop_cols], auxiliary, domain,  label
            return data,transported_X, auxiliary, domain,  label

        if self.drop_cols is not None:
            return data[:self.drop_cols], auxiliary, domain, label
        return data, auxiliary, domain, label

    def __len__(self):
        return len(self.indices)

# Training loop
def train(encoder, predictor, discriminator, train_loader, opt_D, opt_non_D, lr_scheduler_D, lr_scheduler_non_D, epoch, args,classification):

    models = [encoder, predictor, discriminator]
    for model in models:
        model.train()
    sum_discr_loss = 0
    sum_total_loss = 0
    sum_pred_loss = 0

    for batch_idx, data_tuple in tqdm(enumerate(train_loader)):
        # print(batch_idx)
        if args.cuda:
            data_tuple = tuple(ele.cuda() for ele in data_tuple)
        if normalize:
            data_raw, target, domain, data, mask = data_tuple
        else:
            data, target, domain, mask = data_tuple

        # FF encoder and predictor
        encoding = encoder((data, domain))
        prediction = predictor((encoding, domain))

        if use_label_noise:
            noise = (torch.randn(domain.size()).cuda() * label_noise_std).unsqueeze(1)

        # train discriminator
        train_discr_step = 0
        while args.dis_lambda > 0.0:
            train_discr_step += 1
            discr_pred_m, discr_pred_s = discriminator((encoding, domain))
            discr_loss = gaussian_loss(discr_pred_m, discr_pred_s, domain.unsqueeze(1) / args.norm, np.mean(train_list) / args.norm, args.norm)
            for model in models:
                model.zero_grad()
            discr_loss.backward(retain_graph=True)
            opt_D.step()

            # handle extra steps to train the discr's variance branch
            if train_discr_step_extra > 0:
                cur_extra_step = 0
                while True:
                    discr_pred_m, discr_pred_s = discriminator((encoding, domain))
                    discr_loss = gaussian_loss(discr_pred_m.detach(), discr_pred_s, domain.unsqueeze(1) / args.norm)
                    for model in models:
                        model.zero_grad()
                    discr_loss.backward(retain_graph=True)
                    opt_D.step()
                    cur_extra_step += 1
                    if cur_extra_step > train_discr_step_extra:
                        break

            if discr_loss.item() < 1.1 * discr_thres and train_discr_step >= train_discr_step_tot:
                sum_discr_loss += discr_loss.item()
                break

        # handle wgan
        if args.wgan == 'wgan':
            for p in discriminator.parameters():
                p.data.clamp_(args.clamp_lower, args.clamp_upper)

        # train encoder and predictor
        if classification:
            pred_loss = masked_cross_entropy(prediction, target, mask)
        else:
            pred_loss = masked_mse(prediction, target, mask)

        discr_pred_m, discr_pred_s = discriminator((encoding, domain))
        ent_loss = 0

        discr_loss = gaussian_loss(discr_pred_m, discr_pred_s, domain.unsqueeze(1) / args.norm)
        total_loss = pred_loss - discr_loss * args.dis_lambda

        for model in models:
            model.zero_grad()
        total_loss.backward()
        opt_non_D.step()
        sum_pred_loss += pred_loss.item()
        sum_total_loss += total_loss.item()

    lr_scheduler_D.step()
    lr_scheduler_non_D.step()

    avg_discr_loss = sum_discr_loss / len(train_loader.dataset) * args.bs
    avg_pred_loss = sum_pred_loss / len(train_loader.dataset) * args.bs
    avg_total_loss = sum_total_loss / len(train_loader.dataset) * args.bs
    log_txt = 'Train Epoch {}: avg_discr_loss = {:.5f}, avg_pred_loss = {:.3f}, avg_total_loss = {:.3f}'.format(epoch, avg_discr_loss, avg_pred_loss, avg_total_loss)
    print(log_txt)
    plain_log(args.log_file,log_txt+'\n')
    if epoch % args.save_interval == 0 and epoch != 0:
        torch.save(encoder, '%s.model_enc' % args.save_head)
        torch.save(predictor, '%s.model_pred' % args.save_head)
        torch.save(discriminator, '%s.model_discr' % args.save_head)

# Testing loop
def test_regression(encoder, predictor, discriminator, test_loader, args, classification, log_file=None):
    models = [encoder, predictor, discriminator]
    for model in models:
        model.eval()
    test_loss = 0
    rmse_loss = 0
    correct = 0
    l_data = []
    l_label = []
    l_gt = []
    l_encoding = []
    l_domain = []
    l_prob = []
    #for data, target, domain in test_loader:
    for data_tuple in test_loader:
        if args.cuda:
            data_tuple = tuple(ele.cuda() for ele in data_tuple)
        if normalize:
            data_raw, target, domain, data = data_tuple
        else:
            data, target, domain = data_tuple
            data_raw = data
        encoding = encoder((data, domain))
        prediction = predictor((encoding, domain))
        test_loss += F.mse_loss(prediction.squeeze(1), target, reduction='sum').item() # sum up batch loss
        pred = prediction.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += F.l1_loss(prediction.squeeze(1), target, reduction='sum').item()

    test_loss /= len(test_loader.dataset)
    correct   /= len(test_loader.dataset)
    log_txt = 'Test set: MSE loss: {:.7f}, MAE loss: {:.7f}'.format(
        test_loss, correct)
    if log_file is None:
        print(log_txt)
    else:
        print(log_txt,log_file)
    plain_log(args.log_file,log_txt+'\n')
    return test_loss,correct

def test_classification(encoder, predictor, discriminator, test_loader, args, classification, log_file=None):
    models = [encoder, predictor, discriminator]
    for model in models:
        model.eval()
    test_loss = 0
    rmse_loss = 0
    correct = 0
    l_data = []
    l_label = []
    l_gt = []
    l_true = []
    l_encoding = []
    l_domain = []
    l_prob = []
    #for data, target, domain in test_loader:
    for data_tuple in test_loader:
        if args.cuda:
            data_tuple = tuple(ele.cuda() for ele in data_tuple)
        if normalize:
            data_raw, target, domain, data = data_tuple
        else:
            data, target, domain = data_tuple
            data_raw = data
        encoding = encoder((data, domain))
        prediction = predictor((encoding, domain))
        preds = torch.argmax(prediction, 1)
        l_label += list(preds.detach().cpu().numpy())
        l_true += list(target.long().clone().cpu().numpy())
        
    test_loss /= len(test_loader.dataset)
    
    acc = accuracy_score(l_true, l_label)
    print('Accuracy: ', acc)
    log_txt = 'Test set: Accuracy: {:.7f}'.format(acc)
    
    return test_loss, acc

def visualize_trajectory(encoder,predictor,indices,filename=''):
    td = ClassificationDataSet(indices=indices,root_dir=fname,device="cuda:0")
    fig, ax = plt.subplots(3, 3)
    ds = iter(torch.utils.data.DataLoader(td,1,False))
    for i in range(3):
        for j in range(3):
            x,a,u,y = next(ds)
            x_ = []
            y_ = []
            y__ = []
            y___ = []
            actual_time = u.view(1).detach().cpu().numpy()
            for t in tqdm(np.arange(actual_time-0.2,actual_time+0.2,0.005)):
                x_.append(t)
                t = torch.tensor([t*12]).float().to(x.device)
                t.requires_grad_(True)
                delta = (x[0,-1]*12 - t).detach()
                encoding = encoder((x, t))
                y_pred = predictor((encoding, t))
                # y_pred = .classifier(torch.cat([x[:,:-2],x[:,-2].view(-1,1)-delta.view(-1,1), t.view(-1,1)],dim=1), t.view(-1,1)) # TODO change the second last feature also
                partial_Y_pred_t = torch.autograd.grad(y_pred, t, grad_outputs=torch.ones_like(y_pred))[0]
                y_.append(y_pred.item())
                y__.append(partial_Y_pred_t.item())
                y___.append((-partial_Y_pred_t*delta + y_pred).item())
                # TODO gradient addition business
            ax[i,j].plot(x_,y_)
            ax[i,j].plot(x_,y__)
            # ax[i,j].plot(x_,y___)
            ax[i,j].set_title("time-{}".format(actual_time))

            # print(x_,y_)
            ax[i,j].scatter(u.view(-1,1).detach().cpu().numpy(),y.view(-1,1).detach().cpu().numpy())
    plt.savefig('traj_{}.png'.format(filename))
    plt.close()


def main(args):

    seed_torch(args,  args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    '''
    args.dropout=0.0
    args.lr=5e-4
    args.gamma_exp=1000
    args.hidden=800
    args.ratio=1
    args.dis_lambda=1.0
    args.lambda_m=0.0
    args.wgan='wgan'
    args.clamp_lower=-0.15
    args.clamp_upper=0.15
    args.bs=200
    args.num_train=100
    args.loss='default'
    args.evaluate=False
    args.checkpoint='none'
    args.save_head='tmp'
    args.save_interval=20
    args.log_interval=20
    args.log_file='tmp_mlp'
    seed=10
    args.cuda=True
    '''
    global train_list, test_list, mask_list

    args.wgan = 'wgan'
    args.save_head='tmp'
    args.save_interval=20
    args.log_interval=20
    args.log_file='tmp_mlp'
    args.gamma_exp=1000
    args.dis_lambda=1.0
    args.lambda_m=0.0
    args.clamp_lower=-0.15
    args.clamp_upper=0.15
        
    if args.data == "moons":
        from moon_models import DomainEnc, DomainPred, DomainDDisc
        train_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        test_list = [9]
        mask_list = [1]*11
        fname = '../../data/Moons/processed/'
        args.lr = 1e-3
        args.norm = 12
        classification = True

    elif args.data == "house":
        from house_models import DomainEnc, DomainPred, DomainDDisc
        train_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        # 25 epochs
        test_list = [9]
        mask_list = [1]*11
        fname = '../../data/HousePriceTime/'
        args.lr = 1e-3
        args.norm = 1
        classification = False

    elif args.data == "m5":
        from m5_models import DomainEnc, DomainPred, DomainDDisc
        train_list = [0, 1, 2]
        test_list = [3]
        mask_list = [1]*4
        fname = '../../data/M5/processed/'
        args.lr = 1e-3
        args.norm = 5
        classification = False

    elif args.data == "m5_household":

        from m5_models import DomainEnc, DomainPred, DomainDDisc
        train_list = [0, 1, 2]
        test_list = [3]
        mask_list = [1]*4
        fname = '../../data/M5/processed_household/'
        args.lr = 1e-3
        args.norm = 5
        classification = False

    elif args.data == "onp":
        from onp_models import DomainEnc, DomainPred, DomainDDisc
        train_list = [0, 1, 2, 3, 4]
        test_list = [5]
        mask_list = [1]*6
        args.norm = 7        
        fname = '../../data/ONP/processed/'
        args.lr = 1e-3
        classification = True

    elif args.data == "elec":
        from elec_models import DomainEnc, DomainPred, DomainDDisc
        train_list = [x for x in range(29)]
        # 12 epochs
        test_list = [29]
        mask_list = [1]*29
        args.norm = 1   
        fname = '../../data/Elec2/'
        args.lr = 1e-3
        classification = True

    train_loader = torch.utils.data.DataLoader(
        classifdata(fname, train_list, normalize, mask_list),
        shuffle=True,
        batch_size=args.bs, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        classifdata(fname, test_list, normalize),
        batch_size=args.bs, **kwargs)


    encoder = DomainEnc()
    predictor = DomainPred()
    discriminator = DomainDDisc()
    models = [encoder, predictor, discriminator]
    if args.cuda:
        for model in models:
            model.cuda()

    torch.autograd.set_detect_anomaly(True)

    # Set up optimizer
    opt_D = optim.Adam(discriminator.parameters(), lr = args.lr) # lr 
    opt_non_D = optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr = args.lr) # lr 
    lr_scheduler_D = lr_scheduler.ExponentialLR(optimizer=opt_D, gamma=0.5 ** (1/(args.gamma_exp*(train_discr_step_extra+1)) * slow_lrD_decay))
    lr_scheduler_non_D = lr_scheduler.ExponentialLR(optimizer=opt_non_D, gamma=0.5 ** (1/args.gamma_exp))

    ind = list(range(args.bs))
    ind_test = list(range(1000))
    #bce = nn.BCELoss()
    #mse = nn.MSELoss()
    #if classification:
    if classification:

        best_acc = 0

        for ep in range(args.epochs):
            train(encoder, predictor, discriminator, train_loader, opt_D, opt_non_D, lr_scheduler_D, lr_scheduler_non_D, ep, args, classification)
            if ep % 10 == 0:
                loss, acc = test_classification(encoder, predictor, discriminator, test_loader, args, classification)
                if acc > best_acc: best_acc = acc

        log_file = open("cida_%s" %(args.data), "a+")
        print("Seed - {}".format(args.seed),file=log_file)
        print("Accuracy: {}".format(best_acc),file=log_file)
        log_file.close()

    else:

        best_mse,best_mae = 1000000,1000000
        for ep in range(args.epochs):
            train(encoder, predictor, discriminator, train_loader, opt_D, opt_non_D, lr_scheduler_D, lr_scheduler_non_D, ep, args, classification)
            if ep % 5 == 0:
                mse,mae = test_regression(encoder, predictor, discriminator, test_loader, args, classification)
                if mse < best_mse and mae < best_mae:
                    best_mse,best_mae = mse,mae 

        models = [encoder,predictor,discriminator]
        log_file = open("cida_%s" %(args.data), "a+")
        print("Seed - {}".format(args.seed),file=log_file)
        # test(log_file)
        print("MSE: {}".format(best_mse),file=log_file)
        print("MAE: {}".format(best_mae),file=log_file)
        log_file.close()




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help="String, needs to be one of mnist, sleep, moons, m5, elec, m5_household, house")
    parser.add_argument('--epochs',default=10, help="Needs to be int, number of epochs",type=int)
    parser.add_argument('--bs', default=100, help="Batch size",type=int)
    parser.add_argument('--cuda',action='store_true',help="Should we use a GPU")
    parser.add_argument('--seed',default=0,type=int)
    
    args = parser.parse_args()
    main(args)