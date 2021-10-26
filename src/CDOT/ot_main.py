import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.tensorboard import SummaryWriter
import time 
import torch
import random
import os
from transport import *
from models import *
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
#dump all these to a config file

torch.set_num_threads(8)

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_data(root):

    X = np.load("{}/X.npy".format(root))
    Y = np.load("{}/Y.npy".format(root))
    U = np.load("{}/U.npy".format(root))
    indices = json.load(open("{}/indices.json".format(root)))

    return X, U, Y, indices

def init_weights(model):

    if type(model) == nn.Linear:
        nn.init.kaiming_normal_(model.weight)
        model.bias.data.fill_(0.01)

def plot_decision_boundary(c, u, X, Y, name):
    
    

    y = np.argmax(Y[u], -1)
    print(y)

    # Set min and max values and give it some padding
    x_min, x_max = -2.5, 2.0
    y_min, y_max = -2.0, 2.0
    h = 0.005
    # Generate a grid of points with distance h between them
    xx,yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = torch.round(F.sigmoid(c(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]), torch.tensor([[u/11]]*900*800)).detach())).numpy()
    Z = Z.reshape(xx.shape)
    #Z = np.zeros_like(Z)
    # Plot the contour and training examples
    #sns.heatmap(Z)
    #plt.show()
    
    plt.title('%dth domain - %s' %(u, name))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Blues, vmin=-1, vmax=2)
    plt.scatter(X[u][:, 0], X[u][:, 1], c=y, cmap=plt.cm.binary)
    plt.savefig('final_plots/%s_%f.pdf' %(name, u))

def plot_overlapping_boundary(c_1, c_2, u_1, u_2, X, Y, name):
        
    matplotlib.rcParams['text.usetex'] = True
    plt.rc('font', family='serif', size=24, weight='bold')
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath,amsfonts}"]
    matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{bm}"]
    plt.rc('axes', linewidth=1)
    plt.rc('font', weight='bold')
    matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']

    Y2 = np.argmax(Y[u_2], -1)
    Y1 = np.argmax(Y[u_1], -1)

    x_min, x_max = -2.5, 2.0
    y_min, y_max = -2.0, 2.0
    h = 0.005

    xx,yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z1 = c_2(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]), torch.tensor([[u_1/11]]*900*900)).detach().numpy()
    Z1 = Z1.reshape(xx.shape)
    Z2 = c_1(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]), torch.tensor([[u_2/11]]*900*900)).detach().numpy()
    Z2 = Z2.reshape(xx.shape)
    # Plot the contour and training examples
    #sns.heatmap(Z)
    #plt.show()
    #print(Z)
    #Z = (Z1 + 2*Z2)/3.0
    
    '''
    y1 = []
    y2 = []
    for i, x in enumerate(xx[0]):
        y = Z1[:,i]
        idx = np.where(y == 1.0)[0]
        y1.append(yy[:,0][int(np.min(idx))])
        y = Z2[:,i]
        idx = np.where(y == 1.0)[0]
        y2.append(yy[:,0][int(np.min(idx))])

    '''
    plt.xlabel(r'\textbf{feature} $x_1$')
    plt.ylabel(r'\textbf{feature} $x_2$')
    plt.xlim(-2.5, 2.0)
    plt.ylim(-2.0, 2.5)
        
    #plt.plot(xx[0], y1, 'c--', linewidth=3.0)
    #plt.plot(xx[0], y2, color='#00004c', linewidth=3.0)
    plt.contour(xx, yy, Z1, levels=[0], cmap=plt.cm.bwr, vmin=-1.0, vmax=2.0)
    plt.contour(xx, yy, Z2, levels=[0], cmap=plt.cm.seismic)
    prev = plt.scatter(X[u_2][:, 0], X[u_2][:, 1], s=25, c=Y2, cmap=plt.cm.seismic, alpha=0.7)
    cur = plt.scatter(X[u_1][:, 0], X[u_1][:, 1], s=25, c=Y1, cmap=plt.cm.bwr, vmin=-1.0, vmax=2.0, alpha=0.7)
    plt.gcf().subplots_adjust(left=0.15, bottom=0.15)
    plt.savefig('final_plots/%s_%f_%f.pdf' %(name, u_1, u_2))
    plt.clf()

class PredictionModelNN(nn.Module):

    
    def __init__(self, input_shape, hidden_shapes, output_shape, **kwargs):
        
        super(PredictionModelNN, self).__init__()

        self.time_conditioning = kwargs['time_conditioning'] if kwargs.get('time_conditioning') else False
        self.leaky = kwargs['leaky']
        
        if self.time_conditioning:

            self.leaky = kwargs['leaky'] if kwargs.get('leaky') else False
            
        use_time2vec = kwargs['use_time2vec'] if kwargs.get('use_time2vec') else False
        self.regress = kwargs['task'] == 'regression' if kwargs.get('task') else False
        self.time_shape = 1

        if use_time2vec:
            self.time_shape = 8
            self.time2vec = Time2Vec(1, 8)
        else:
            self.time_shape = 1
            self.time2vec = None

        self.layers = nn.ModuleList()
        self.relus = nn.ModuleList()

        self.input_shape = input_shape
        self.hidden_shapes = hidden_shapes
        self.output_shape = output_shape
        
        if len(self.hidden_shapes) == 0: # single layer NN, no TReLU

            self.layers.append(nn.Linear(input_shape, output_shape))
            self.relus.append(nn.LeakyReLU())

        else:

            self.layers.append(nn.Linear(self.input_shape, self.hidden_shapes[0]))
            if self.time_conditioning:
                self.relus.append(TimeReLU(data_shape=self.hidden_shapes[0], time_shape=self.time_shape, leaky=self.leaky))
            else:
                if self.leaky:
                    self.relus.append(nn.LeakyReLU())
                else:
                    self.relus.append(nn.ReLU())

            for i in range(len(self.hidden_shapes) - 1):

                self.layers.append(nn.Linear(self.hidden_shapes[i], self.hidden_shapes[i+1]))
                if self.time_conditioning:
                    self.relus.append(TimeReLU(data_shape=self.hidden_shapes[i+1], time_shape=self.time_shape, leaky=self.leaky))
                else:
                    if self.leaky:
                        self.relus.append(nn.LeakyReLU())
                    else:
                        self.relus.append(nn.ReLU())


            self.layers.append(nn.Linear(self.hidden_shapes[-1], self.output_shape))

        self.apply(init_weights)

        for w in self.layers[0].parameters():
            print(w)
    
            
    def forward(self, X, times=None, logits=False, reps=False):

        if self.time_conditioning:
            X = torch.cat([X, times], dim=-1)

        if self.time2vec is not None:
            times = self.time2vec(times)

        #if self.time_conditioning:
        #    X = self.relus[0](self.layers[0](X), times)
        #else:
        #    X = self.relus[0](self.layers[0](X))

        for i in range(0, len(self.layers)-1):

            X = self.layers[i](X)
            if self.time_conditioning:
                X = self.relus[i](X, times)
            else:
                X = self.relus[i](X)

        X = self.layers[-1](X)
        #if self.regress:
        #   X = torch.relu(X)
        #else:
        #   X = torch.softmax(X,dim=1)
        '''
        if not logits:
            if self.output_shape > 1:
                X = F.softmax(X, dim=-1)
            else:
                X = F.sigmoid(X)
        '''
        return X        

"""

Method to train a classifier with a minibatch of examples

Arguments:
    X: Training features
    Y: Training labels
    classifier: Model
    classifier_optimizer: Optimizer

Returns:
    prediction loss

"""

def train_classifier(X, Y, classifier, classifier_optimizer, binary):

    classifier_optimizer.zero_grad()

    if binary:    
        
        Y_pred = torch.sigmoid(classifier(X))
        Y_true = torch.argmax(Y, 1).view(-1,1).float()
        pred_loss = -torch.mean(Y_true * torch.log(Y_pred + 1e-15) + (1 - Y_true) * torch.log(1 - Y_pred + 1e-15))

    else:

        Y_pred = classifier(X)
        Y_pred = torch.softmax(Y_pred, -1)
        pred_loss = -torch.mean(Y * torch.log(Y_pred + 1e-15))
    
    pred_loss.backward()
    
    classifier_optimizer.step()

    return pred_loss

def train(X_data, Y_data, U_data, source_indices, target_indices, args, binary):

    log_file = open('cdot_%s' %(args.data), 'a+')
    X_source = X_data[source_indices[0]]
    if args.data == "mnist":
        X_source = X_source.reshape(-1, 784)
    Y_source = Y_data[source_indices[0]]
    Y_source = np.argmax(Y_source, -1)

    X_aux = list(X_data[source_indices[1:]])
    Y_aux = list(Y_data[source_indices[1:]])
    if args.data == "mnist":
        X_aux = [x.reshape(-1, 784) for x in X_aux]

    Y_aux2 = []
    for i in range(len(Y_aux)):
        Y_aux2.append(np.argmax(Y_aux[i], -1))

    Y_aux = Y_aux2

    X_target = X_data[target_indices[0]]
    if args.data == "mnist":
        X_target = X_target.reshape(-1, 784)

    Y_target = Y_data[target_indices[0]]

    Y_target = np.argmax(Y_target, -1)

    X_source, X_aux, X_target = transform_samples_reg_otda(X_source, Y_source, X_aux, Y_aux, X_target, Y_target)

    if args.data == "mnist":

        X_source = X_source.reshape(-1, 1, 28, 28)
        X_target = X_target.reshape(-1, 1, 28, 28)
        X_aux = [x.reshape(-1, 1, 28, 28) for x in X_aux]

    X_source = np.vstack([X_source] + X_aux)
    Y_source = np.hstack([Y_source] + Y_aux)
    num_classes = np.max(Y_source) + 1
    Y_source = np.eye(num_classes)[Y_source]
    Y_target = np.eye(num_classes)[Y_target]

    BATCH_SIZE = args.bs
    EPOCH = args.epoch

    if args.data == "moons":
        classifier = PredictionModelNN(2, [50, 50], 1, leaky=True)
        classifier_optimizer = torch.optim.Adam(classifier.parameters(), 5e-3)
    elif args.data == "mnist":
        model_kwargs =  {"block": ResidualBlock,
                     "layers": [2, 2, 2, 2],
                     "time_conditioning": False,
                     "leaky": False,
                     "append_time": False,
                     "use_time2vec": False
                     }
        classifier = ResNet(**model_kwargs)

        classifier_optimizer = torch.optim.Adam(classifier.parameters(), 1e-4)

    elif args.data == "onp":

        classifier = PredictionModelNN(58, [200], 1, leaky=True)
        classifier_optimizer = torch.optim.Adam(classifier.parameters(), 1e-3)

    elif args.data == "elec":

        classifier = PredictionModelNN(8, [128, 128], 1, leaky=True)
        classifier_optimizer = torch.optim.Adam(classifier.parameters(), 5e-3)



    writer = SummaryWriter(comment='{}'.format(time.time()))

    past_data = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_source).float(), torch.tensor(Y_source).float()),BATCH_SIZE,False)    
    print('------------------------------------------------------------------------------------------')
    print('TRAINING')
    print('------------------------------------------------------------------------------------------')
    for epoch in range(EPOCH):
        loss = 0
        for batch_X, batch_Y in past_data:
            loss += train_classifier(batch_X, batch_Y, classifier, classifier_optimizer, binary)
        if epoch%1 == 0: print('Epoch %d - %f' % (epoch, loss.detach().cpu().numpy()))
    
    print('------------------------------------------------------------------------------------------')
    print('TESTING')
    print('------------------------------------------------------------------------------------------')
        
    target_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_target).float(), torch.tensor(Y_target).float()),BATCH_SIZE,False)
    Y_pred = []
    for batch_X, batch_Y in target_dataset:
        batch_Y_pred = classifier(batch_X)
        if binary:
            batch_Y_pred = torch.sigmoid(batch_Y_pred).detach().cpu().numpy()
        else:
            batch_Y_pred = torch.softmax(batch_Y_pred, -1).detach().cpu().numpy()

        Y_pred = Y_pred + [batch_Y_pred]  
    Y_pred = np.vstack(Y_pred)
    # print(Y_pred)
    Y_true = np.argmax(Y_target, -1)
    if binary:
        Y_pred = np.array([0 if y < 0.5 else 1 for y in Y_pred])

    else:
        Y_pred = np.argmax(Y_pred, -1)

    print(accuracy_score(Y_true, Y_pred), file=log_file)
    print(confusion_matrix(Y_true, Y_pred), file=log_file)
    print(classification_report(Y_true, Y_pred), file=log_file)    


        
def main(args):

    seed_torch(args.seed)
    if args.use_cuda:
        args.device = "cuda:0"
    else:
        args.device = "cpu"

    if args.data == "moons":
        
        X_data, U_data, Y_data, indices = load_data('../../data/Moons/processed')
        Y_data = np.eye(2)[Y_data]

        X_data = np.array([X_data[ids] for ids in indices])
        Y_data = np.array([Y_data[ids] for ids in indices])
        U_data = np.array([U_data[ids] for ids in indices])
        train(X_data, Y_data, U_data, list(range(0, 9)), [9], args, binary=True)

    elif args.data == "mnist":
        
        X_data, U_data, Y_data, indices = load_data('../../data/MNIST/processed')
        Y_data = np.eye(10)[Y_data]

        X_data = np.array([X_data[ids] for ids in indices])
        Y_data = np.array([Y_data[ids] for ids in indices])
        U_data = np.array([U_data[ids] for ids in indices])

        train(X_data, Y_data, U_data, list(range(0, 4)), [4], args, binary=False)

    elif args.data == "onp":
        
        X_data, U_data, Y_data, indices = load_data('../../data/ONP/processed')
        Y_data = np.eye(2)[Y_data]

        X_data = np.array([X_data[ids] for ids in indices])
        Y_data = np.array([Y_data[ids] for ids in indices])
        U_data = np.array([U_data[ids] for ids in indices])
        train(X_data, Y_data, U_data, list(range(0, 5)), [5], args, binary=True)

    elif args.data == "elec":
        
        X_data, U_data, Y_data, indices = load_data('../../data/Elec2')
        Y_data = np.eye(2)[Y_data]

        X_data = np.array([X_data[ids] for ids in indices])
        Y_data = np.array([Y_data[ids] for ids in indices])
        U_data = np.array([U_data[ids] for ids in indices])
        train(X_data, Y_data, U_data, list(range(20, 29)), [29], args, binary=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--data', help="String, needs to be one of mnist, sleep, moons, cars")
    parser.add_argument('--epoch', default=5, help="Needs to be int, number of epochs for classifier",type=int)
    parser.add_argument('--bs', default=100, help="Batch size",type=int)
    parser.add_argument('--use_cuda', action='store_true', help="Should we use a GPU")
    parser.add_argument('--seed', default=0, type=int)

    args = parser.parse_args()
    main(args)    