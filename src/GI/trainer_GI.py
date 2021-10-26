'''Abstraction

This is an abstract class for all trainers, it wraps up datasets etc. The train script will call this with the appropriate params
''' 

import torch 
import numpy as np
import json
import pickle
import os 

from matplotlib import pyplot as plt
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt

from utils import *
from dataset_GI import *
from config_GI import Config
from tqdm import tqdm
from losses import *
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from torchviz import make_dot
import time 

# np.set_printoptions(precision = 3)
np. set_printoptions(suppress=True)
def train_classifier_batch(X,dest_u,dest_a,Y,classifier,classifier_optimizer,batch_size,verbose=False,encoder=None, transformer=None,source_u=None, kernel=None,loss_fn=classification_loss):
    '''Trains classifier on a batch
    
    
    Arguments:
        X {[type]} -- 
        Y {[type]} -- 
        classifier {[type]} -- 
        classifier_optimizer {[type]} -- 
    
    Keyword Arguments:
        transformer {[type]} -- Transformer model. If this is none we just train the classifier on the input data. (default: {None})
        encoder {[type]} --   (default: {None})
    
    Returns:
        [type] -- [description]
    '''
    log = open("Classifier_log.txt","a")
    classifier_optimizer.zero_grad()
    if encoder is not None:
        with torch.no_grad():
            X = encoder(X) #.view(out_shape)

    if transformer is not None:
        # print(source_u.size(),dest_u.size())
        X_pred = transformer(X,torch.cat([source_u.squeeze(-1),dest_u],dim=1))
    else:
        X_pred = X

    Y_pred = classifier(X_pred,dest_a)
    pred_loss = loss_fn(Y_pred, Y)

    if kernel is not None:
        pred_loss = pred_loss * kernel
    pred_loss = pred_loss.mean()  #/batch_size
    pred_loss.backward()
    classifier_optimizer.step()
    


    return pred_loss



def adversarial_finetune(X, U, Y, delta, classifier, classifier_optimizer,classifier_loss_fn,delta_lr=0.1,delta_clamp=0.15,delta_steps=10,lambda_GI=0.5,writer=None,step=None,string=None, verbose=False):
    
    classifier_optimizer.zero_grad()
    
    delta.requires_grad_(True)

    # This block of code computes delta adversarially
    d1 = delta.detach().cpu().numpy()
    for ii in range(delta_steps):
        delta = delta.clone().detach()
        delta.requires_grad_(True)
        U_grad = U.clone() - delta
        U_grad.requires_grad_(True)
        Y_pred = classifier(X, U_grad, logits=True)
        if len(Y.shape)>1 and Y.shape[1] > 1:
            Y_true = torch.argmax(Y, 1).view(-1,1).float()

        partial_logit_pred_t = []
        if len(Y_pred.shape)<2 or Y_pred.shape[1] < 2:
            partial_Y_pred_t = torch.autograd.grad(Y_pred, U_grad, grad_outputs=torch.ones_like(Y_pred), retain_graph=True,create_graph=True)[0]
        else:           
            for idx in range(Y_pred.shape[1]):
                logit = Y_pred[:,idx].view(-1,1)
                partial_logit_pred_t.append(torch.autograd.grad(logit, U_grad, grad_outputs=torch.ones_like(logit), create_graph=True)[0])

            
            partial_Y_pred_t = torch.cat(partial_logit_pred_t, 1)



        Y_pred = Y_pred + delta * partial_Y_pred_t

        if len(Y_pred.shape)>1 and Y_pred.shape[1] > 1:
            Y_pred = torch.softmax(Y_pred,dim=-1)
        loss = classifier_loss_fn(Y_pred,Y).mean()
        partial_loss_delta = torch.autograd.grad(loss, delta, grad_outputs=torch.ones_like(loss), retain_graph=True)[0]
        delta = delta + delta_lr*partial_loss_delta

        if delta.size(0) > 1:
            delta[delta != delta] = 0.
            if torch.norm(partial_loss_delta) < 1e-3*delta.size(0) :
                    break
        else:
            if torch.norm(partial_loss_delta) < 1e-3 or delta > delta_clamp or delta < -1*delta_clamp:
                break
# 
    delta = delta.clamp(-1*delta_clamp, delta_clamp).detach().clone()
    d2 = delta.detach().cpu().numpy()
    if writer is not None:
        writer.add_scalar(string,torch.abs(delta).mean(),step)
        writer.add_scalar(string+"_grad",torch.abs(partial_loss_delta).mean(),step)

    # This block of code actually optimizes our model
    U_grad = U.clone() - delta
    U_grad.requires_grad_(True)
    Y_pred = classifier(X, U_grad, logits=True)

    partial_logit_pred_t = []

    if len(Y_pred.shape)<2 or Y_pred.shape[1] < 2:
        partial_Y_pred_t = torch.autograd.grad(Y_pred, U_grad, grad_outputs=torch.ones_like(Y_pred), create_graph=True)[0]
    else:
        for idx in range(Y_pred.shape[1]):
            logit = Y_pred[:,idx].view(-1,1)
            partial_logit_pred_t.append(torch.autograd.grad(logit, U_grad, grad_outputs=torch.ones_like(logit), retain_graph=True)[0])
        partial_Y_pred_t = torch.cat(partial_logit_pred_t, 1)

    Y_pred = Y_pred + delta * partial_Y_pred_t
    if len(Y_pred.shape)>1 and Y_pred.shape[1] > 1:
        Y_pred = torch.softmax(Y_pred,dim=-1)
    Y_orig_pred = classifier(X,U)

    # TODO INVERT LAMBDA for HOUSE, ELEC
    pred_loss = classifier_loss_fn(Y_pred,Y).mean() + lambda_GI*classifier_loss_fn(Y_orig_pred,Y).mean() 

    pred_loss.backward()
    
    classifier_optimizer.step()

    return pred_loss, delta


def adversarial_finetune_goodfellow(X, U, Y, delta, classifier, classifier_optimizer,classifier_loss_fn,delta_lr=0.1,delta_clamp=0.15,delta_steps=10,lambda_GI=0.5,writer=None,step=None,string=None):
    
    classifier_optimizer.zero_grad()
    
    delta.requires_grad_(True)
    
    for ii in range(delta_steps):

        U_grad = U.clone() + delta
        U_grad.requires_grad_(True)
        Y_pred = classifier(X, U_grad)

        loss = classifier_loss_fn(Y_pred,Y).mean()
        partial_loss_delta = torch.autograd.grad(loss, delta, grad_outputs=torch.ones_like(loss), retain_graph=True)[0]

        delta = delta + delta_lr*partial_loss_delta
        #print('%d %f' %(ii, delta.clone().detach().numpy()))
    
    delta = delta.clamp(-1*delta_clamp, delta_clamp)
    delta = delta.detach()
    U_grad = U.clone() + delta
    U_grad.requires_grad_(True)
    Y_pred = classifier(X, U_grad)

    
    pred_loss = classifier_loss_fn(Y_pred,Y).mean()# + (1 - Y_true) * torch.log(1 - Y_pred + 1e-9))
    #pred_loss = torch.mean((Y_pred - Y_true)**2)
    pred_loss.backward()
    
    classifier_optimizer.step()

    return pred_loss


def finetune_gradient_regularization(X, U, Y, delta, classifier, classifier_optimizer,classifier_loss_fn,delta_lr=0.1,delta_clamp=0.15,delta_steps=10,lambda_GI=0.5,writer=None,step=None,string=None):
    # This does simple gradient regularization

    classifier_optimizer.zero_grad()
    time = U.clone().requires_grad_(True)
    Y_pred = classifier(X,time, logits=True)
    partial_Y_pred_t = []
    if len(Y_pred.shape)<2 or Y_pred.shape[1] < 2:
        partial_Y_pred_t = torch.autograd.grad(Y_pred, time, grad_outputs=torch.ones_like(Y_pred), create_graph=True)[0]
    else:
        for idx in range(Y_pred.shape[1]):
            logit = Y_pred[:,idx].view(-1,1)
            partial_Y_pred_t.append(torch.autograd.grad(logit, time, grad_outputs=torch.ones_like(logit), create_graph=True)[0])
        partial_Y_pred_t = torch.cat(partial_Y_pred_t, 1)

    if len(Y_pred.shape)>1 and Y_pred.shape[1] > 1:
        Y_pred = torch.softmax(Y_pred,dim=-1)

    grad = partial_Y_pred_t**2

    pred_loss = 1.0*classifier_loss_fn(Y_pred,Y).mean() + lambda_GI*(grad.mean())

    pred_loss.backward()
    
    classifier_optimizer.step()


    return pred_loss



class GradRegTrainer():
    def __init__(self,args):

        if args.model == "baseline":
            from config_baseline import Config
            config = Config(args)
        elif args.model == "tbaseline" or args.model == "goodfellow" or args.model == "inc_finetune":
            from config_tbaseline import Config
            config = Config(args)
        elif args.model == "GI" or args.model == "t_inc_finetune" or args.model == "t_goodfellow" or args.model == "t_GI" or args.model == "grad_reg":
            from config_GI import Config
            config = Config(args)

        self.log = config.log
        self.DataSetClassifier = ClassificationDataSet
        self.CLASSIFIER_EPOCHS = config.epoch_classifier
        self.FINETUNING_EPOCHS = config.epoch_finetune 

        self.SUBEPOCHS = 1
        self.BATCH_SIZE = config.bs
        self.CLASSIFICATION_BATCH_SIZE = 100
        # self.PRETRAIN_EPOCH = 5
        self.data = args.data 
        self.model = args.model
        self.update_num_steps = 1
        self.delta = config.delta
        self.use_pretrained = config.use_pretrained

        self.trelu_limit = args.trelu_limit
        self.writer = None 


        self.dataset_kwargs = config.dataset_kwargs
        self.source_domain_indices = config.source_domain_indices   #[6,7,8,9,10]
        self.target_domain_indices = config.target_domain_indices   #[11]
        data_index_file = config.data_index_file    #"../../data/HousePrice/indices.json"
        self.classifier = config.classifier(**config.model_kwargs).to(args.device) 
        self.lr = config.lr
        self.lr_reduce = config.lr_reduce
        self.w_decay = config.w_decay
        self.schedule = config.schedule
        self.warm_start = config.warm_start
        if self.w_decay is None:
            self.classifier_optimizer = torch.optim.Adam(self.classifier.parameters(),config.lr)
        else:
            self.classifier_optimizer = torch.optim.AdamW(self.classifier.parameters(), config.lr, weight_decay=self.w_decay)


        if self.schedule:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.classifier_optimizer, step_size=15, gamma=0.5)
        # loss_type = 'bce' if self.dataset_kwargs['return_binary'] else 'reg'
        
        self.classifier_loss_fn = config.classifier_loss_fn   #reconstruction_loss
        self.task = config.loss_type


        self.inc_finetune = self.model in ["inc_finetune","t_inc_finetune", "t_GI"]
        
        self.encoder = None
        self.delta_lr=config.delta_lr
        self.delta_clamp=config.delta_clamp
        self.delta_steps=config.delta_steps
        self.lambda_GI=config.lambda_GI
        self.num_finetune_domains = config.num_finetune_domains

        data_indices = json.load(open(data_index_file,"r")) #, allow_pickle=True)
        self.source_data_indices = [data_indices[i] for i in self.source_domain_indices]
        self.cumulative_data_indices = get_cumulative_data_indices(self.source_data_indices)
        self.target_indices = [data_indices[i] for i in self.target_domain_indices][0] 
        self.shuffle = True
        self.device = args.device
        self.patience = 2
        self.early_stopping = config.early_stopping
        self.seed = args.seed
        # print(self.delta)

        if self.model in ["GI_t_delta","GI"] :
            self.delta = [(torch.rand(1).float()*(0.1-(-0.1)) - 0.1).to(self.device) for _ in range(len(self.source_data_indices))]


    def train_classifier(self,past_dataset=None,encoder=None,inc_finetune=False):
        '''Train the classifier initially
        
        In its current form the function just trains a baseline model on the entire train data
        
        Keyword Arguments:
            past_dataset {[type]} -- If this is not None, then the `past_dataset` is used to train the model (default: {None})
            encoder {[type]} -- Encoder model to train (default: {None})
        '''
        if not self.inc_finetune:
            class_step = 0
            past_data = ClassificationDataSet(indices=self.cumulative_data_indices[-1],**self.dataset_kwargs)
            past_dataset = torch.utils.data.DataLoader((past_data),self.BATCH_SIZE,True)
            for epoch in range(self.CLASSIFIER_EPOCHS):
                class_loss = 0
                for batch_X, batch_A, batch_U, batch_Y in (past_dataset):

                    l = train_classifier_batch(X=batch_X,dest_u=batch_U,dest_a=batch_U,Y=batch_Y,classifier=self.classifier,classifier_optimizer=self.classifier_optimizer,verbose=(class_step%20)==0,encoder=encoder, batch_size=self.BATCH_SIZE,loss_fn=self.classifier_loss_fn)
                    class_step += 1
                    class_loss += l
                    if self.writer is not None:
                        self.writer.add_scalar("loss/classifier",l.item(),class_step)
                print("Epoch %d Loss %f"%(epoch,class_loss/(len(past_data)/(self.BATCH_SIZE))),flush=False)
                if self.schedule:
                    self.scheduler.step()
        else:
            class_step = 0
            for i in range(len(self.source_domain_indices)):
                past_data = ClassificationDataSet(indices=self.source_data_indices[i],**self.dataset_kwargs)
                past_dataset = torch.utils.data.DataLoader((past_data),self.BATCH_SIZE,True)
                for epoch in range(int(self.CLASSIFIER_EPOCHS*(1-(i/10)))):
                    class_loss = 0
                    for batch_X, batch_A, batch_U, batch_Y in past_dataset:

                        l = train_classifier_batch(X=batch_X,dest_u=batch_U,dest_a=batch_U,Y=batch_Y,classifier=self.classifier,classifier_optimizer=self.classifier_optimizer,verbose=(class_step%20)==0,encoder=encoder, batch_size=self.BATCH_SIZE,loss_fn=self.classifier_loss_fn)
                        class_step += 1
                        class_loss += l
                        if self.writer is not None:
                            self.writer.add_scalar("loss/classifier",l.item(),class_step)
                    print("Epoch %d Loss %f"%(epoch,class_loss/(len(past_data)/(self.BATCH_SIZE))),flush=False)
                if self.schedule:
                    self.scheduler.step()


    def finetune_grad_int(self, num_domains=2):
        '''Finetunes using gradient interpolation
        
        Keyword Arguments:
            num_domains {number} -- Number of domains on which to fine-tune (default: {2})
        '''
        Y_pred = []
        Y_label = []
        dom_indices = np.arange(len(self.source_domain_indices)).astype('int')[-1*num_domains:]
        for i in dom_indices:
            delta_ = torch.FloatTensor(1,1).uniform_(-0.1,0.1).to(self.device) #self.delta 
            # TODO!!!!
            self.classifier_optimizer = torch.optim.Adam(self.classifier.parameters(),5e-4)  # Do we change this?

            past_data = ClassificationDataSet(indices=self.source_data_indices[i],**self.dataset_kwargs)
            past_dataset = torch.utils.data.DataLoader((past_data),self.BATCH_SIZE,True)
            
            # For early stopping we look at the next domain, if that is not possible we look at training loss itself 
            try:
                val_dataset =  torch.utils.data.DataLoader(ClassificationDataSet(indices=self.source_data_indices[i+1],**self.dataset_kwargs),self.BATCH_SIZE,True)
            except:
                val_dataset =  torch.utils.data.DataLoader(ClassificationDataSet(indices=self.source_data_indices[i],**self.dataset_kwargs),self.BATCH_SIZE,True)
            bad_ep = 0
            prev_net_val_loss = 1000000000


            step = 0
            for epoch in range(self.FINETUNING_EPOCHS):
                

                loss = 0
                for batch_X, _, batch_U, batch_Y in tqdm(past_dataset):

                    batch_U = batch_U.view(-1,1)
                    # print(batch_U)
                    if self.model == "goodfellow" or self.model == "t_goodfellow":
                        delta = (torch.zeros(batch_U.size()).float()).to(batch_X.device)
                        l = adversarial_finetune_goodfellow(batch_X, batch_U, batch_Y, delta, self.classifier, self.classifier_optimizer,self.classifier_loss_fn,delta_lr=self.delta_lr,delta_clamp=self.delta_clamp,delta_steps=self.delta_steps,lambda_GI=self.lambda_GI,writer=self.writer,step=step,string="delta_{}".format(i))
                    elif self.model in ["GI","t_GI"] :

                        if self.warm_start:
                            self.delta = torch.tensor(delta_) 
                            if torch.abs(self.delta) < 1e-4 or torch.abs(self.delta-self.delta_clamp) < 1e-4 or torch.abs(self.delta+self.delta_clamp) < 1e-4:
                                # This resets delta upon clamping or hitting 0
                                self.delta = torch.FloatTensor(1,1).uniform_(-0.1,0.1).to(self.device)
                        else:

                            self.delta = (torch.rand(batch_U.size()).float()*(2*self.delta_clamp) - self.delta_clamp).to(batch_X.device)

                        l,delta_ = adversarial_finetune(batch_X, batch_U, batch_Y, self.delta, self.classifier, self.classifier_optimizer,self.classifier_loss_fn,delta_lr=self.delta_lr,delta_clamp=self.delta_clamp,delta_steps=self.delta_steps,lambda_GI=self.lambda_GI,writer=self.writer,step=step,string="delta_{}".format(i), verbose=step%20 == 0)
                    elif self.model in ["grad_reg"]:
                        delta = (torch.rand(batch_U.size()).float()*(0.1-(-0.1)) - 0.1).to(batch_X.device)
                        l = finetune_gradient_regularization(batch_X, batch_U, batch_Y, delta, self.classifier, self.classifier_optimizer,self.classifier_loss_fn,delta_lr=self.delta_lr,delta_clamp=self.delta_clamp,delta_steps=self.delta_steps,lambda_GI=self.lambda_GI,writer=self.writer,step=step,string="delta_{}".format(i))

                    
                    loss = loss + l
                    if self.writer is not None:
                        self.writer.add_scalar("loss/test_{}".format(i),l.item(),step)
                    step += 1

                print("Epoch %d Loss %f"%(epoch,loss/(len(past_dataset))))
            
            # Validation -
                if self.early_stopping:
                    with torch.no_grad():
                        net_val_loss = 0
                        for batch_X, _, batch_U, batch_Y in tqdm(val_dataset):
                            batch_Y_pred = self.classifier(batch_X, batch_U)
                            net_val_loss += self.classifier_loss_fn(batch_Y_pred,batch_Y).sum().item()
                        
                        if net_val_loss > prev_net_val_loss:
                            bad_ep += 1
                        else:
                            # torch.save()
                            best_model = self.classifier.state_dict()
                            bad_ep = 0
                        prev_net_val_loss = min(net_val_loss,prev_net_val_loss)

                    if bad_ep > self.patience:
                        print("Early stopping for domain {}".format(i))
                        self.classifier.load_state_dict(best_model)
                        break


    def eval_classifier(self,log, ensemble=False, verbose=True):

        if self.data == "house":
            self.dataset_kwargs["drop_cols_classifier"] = None
        self.classifier.eval()
        td = ClassificationDataSet(indices=self.target_indices,**self.dataset_kwargs)
        target_dataset = torch.utils.data.DataLoader(td,self.BATCH_SIZE,False,drop_last=False)
        Y_pred = []
        Y_label = []
        for batch_X, batch_A,batch_U, batch_Y in target_dataset:
            batch_U = batch_U.view(-1,1)
            if self.encoder is not None:
                batch_X = self.encoder(batch_X)
            if ensemble:
                batch_Y_pred = self.predict_ensemble(batch_X, batch_U, delta=self.delta,k=self.max_k).detach().cpu().numpy()
            else:
                batch_Y_pred = self.classifier(batch_X, batch_U).detach().cpu().numpy()
            if self.task == 'classification':
                if batch_Y_pred.shape[1] > 1:
                    Y_pred = Y_pred + [np.argmax(batch_Y_pred,axis=1).reshape((-1,1))]
                else:
                    Y_pred = Y_pred + [(batch_Y_pred>0.5)*1.0]

                Y_label = Y_label + [batch_Y.detach().cpu().numpy()]
            elif self.task == 'regression':
                Y_pred = Y_pred + [batch_Y_pred.reshape(-1,1)]
                Y_label = Y_label + [batch_Y.detach().cpu().numpy().reshape(-1,1)]

        if self.task == 'classification':
            Y_pred = np.vstack(Y_pred)
            Y_label = np.hstack(Y_label)
            print('shape: ',Y_pred.shape, Y_label.shape)
            print('Accuracy: ',accuracy_score(Y_label, Y_pred),file=log)
        else:
            Y_pred = np.vstack(Y_pred)
            Y_label = np.vstack(Y_label)
            print('MAE: ',np.mean(np.abs(Y_label-Y_pred),axis=0),file=log)


    def train(self):    

        # if os.path.exists("classifier_{}_{}.pth".format(self.data,self.seed)):
        if self.use_pretrained and self.model == "GI":
            try:
                self.classifier.load_state_dict(torch.load("./saved_models/classifier_{}_{}.pth".format(self.data,self.seed)))
                print("Loading Model")
            except Exception as e:
                print('No pretrained model found:\n', e)
                self.classifier.train()
                self.train_classifier(encoder=self.encoder)  # Train classifier initially
                
        else:
            self.classifier.train()
            self.train_classifier(encoder=self.encoder)  # Train classifier initially



        if self.model not in ["GI","goodfellow","t_goodfellow","grad_reg"]:

            self.eval_classifier(log=self.log)
    
        else:
    
            self.eval_classifier(log=open("dump.txt","w"))
            self.classifier.train()
            self.classifier_optimizer = torch.optim.Adam(self.classifier.parameters(),self.lr/self.lr_reduce)

            self.finetune_grad_int(num_domains=self.num_finetune_domains)
            self.eval_classifier(log=self.log)

