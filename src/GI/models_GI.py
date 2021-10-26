import torch
from torch import nn
import torch.nn.functional as F



def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)
        # nn.init.kaiming_normal_(m.bias)

        m.bias.data.fill_(0.01)



class Time2Vec(nn.Module):

    '''
    Time encoding inspired by the Time2Vec paper
    '''

    def __init__(self, in_shape, out_shape):

        super(Time2Vec, self).__init__()
        linear_shape = out_shape//4
        dirac_shape = 0
        sine_shape = out_shape - linear_shape - dirac_shape
        self.model_0 = nn.Linear(in_shape, linear_shape)
        self.model_1 = nn.Linear(in_shape, sine_shape)

    def forward(self, X):

        te_lin = self.model_0(X)
        te_sin = torch.sin(self.model_1(X))
        if len(te_lin.shape) == 3:
            te_lin = te_lin.squeeze(1)
        if len(te_sin.shape) == 3:
            te_sin = te_sin.squeeze(1)
        te = torch.cat([te_lin, te_sin], axis=1)
        return te

class TimeReLU(nn.Module):

    '''
    A ReLU with threshold and alpha as a function of domain indices.
    '''

    def __init__(self, data_shape, time_shape, leaky=False,use_time=True , deep=False):
        
        super(TimeReLU,self).__init__()
        self.deep = deep  # Option for a deeper version of TReLU
        if deep:
            trelu_shape = 16
        else:
            trelu_shape = 1
        self.leaky = leaky

        self.use_time = use_time  # Whether TReLU is active or not
        self.model_0 = nn.Linear(time_shape, trelu_shape)
        
        self.model_1 = nn.Linear(trelu_shape, data_shape)

        self.time_dim = time_shape        

        if self.leaky:

            self.model_alpha_0 = nn.Linear(time_shape, trelu_shape)
            
            self.model_alpha_1 = nn.Linear(trelu_shape, data_shape)

        self.sigmoid = nn.Sigmoid()

        if self.leaky:
            self.relu = nn.LeakyReLU()
        else:
            self.relu = nn.ReLU()


    def forward(self, X, times):

        if not self.use_time:
            return self.relu(X)
        if len(times.size()) == 3:
            times = times.squeeze(2)
        thresholds = self.model_1(self.relu(self.model_0(times)))

        if self.leaky:
            alphas = self.model_alpha_1(self.relu(self.model_alpha_0(times)))
        else:
            alphas = 0.0
        if self.deep:
            X = torch.where(X>thresholds,X,alphas*(X-thresholds) + thresholds)
        else:
            X = torch.where(X>thresholds,X-thresholds,alphas*(X-thresholds))

        return X

class TimeReLUCNN(nn.Module):

    def __init__(self, data_shape, time_shape, leaky=False,use_time=True):
        
        super(TimeReLUCNN,self).__init__()
        self.leaky = leaky
        self.use_time = use_time
        # print("RELU - {}".format(self.use_time))
        if use_time:
            self.model_0 = nn.Linear(time_shape, 16)
            #nn.init.kaiming_normal_(self.model_0.weight)
            #nn.init.zeros_(self.model_0.bias)
            self.model_1 = nn.Linear(16, data_shape)
            #nn.init.kaiming_normal_(self.model_1.weight)
            #nn.init.zeros_(self.model_1.bias)
            
            self.time_dim = time_shape        

            if self.leaky:
                self.model_alpha_0 = nn.Linear(time_shape, 16)
                #nn.init.kaiming_normal_(self.model_alpha_0.weight)
                #nn.init.zeros_(self.model_alpha_0.bias)
                self.model_alpha_1 = nn.Linear(16, data_shape)
                #nn.init.kaiming_normal_(self.model_alpha_1.weight)
                #nn.init.zeros_(self.model_alpha_1.bias)
            
            self.leaky = leaky
            self.time_dim = time_shape
        else:
            if self.leaky:
                self.model = nn.LeakyReLU()
            else:
                self.model = nn.ReLU()
    def forward(self, X, times):
        # times = X[:,-self.time_dim:]
        if not self.use_time:
            return self.model(X) 

        if len(times.size()) == 3:
            times = times.squeeze(2)        
        orig_shape = X.size()
        # print(orig_shape)
        #X = X.view(orig_shape[0],-1)
        #if len(times.size()) == 3:
        #    times = times.squeeze(2)
        
        thresholds = self.model_1(F.relu(self.model_0(times)))
        if self.leaky:
            alphas = self.model_alpha_1(F.relu(self.model_alpha_0(times)))
        else:
            alphas = 0.0

        thresholds = thresholds[:,:,None,None]
        alphas = alphas[:,:,None,None]
        X = torch.where(X>thresholds,X,alphas*(X-thresholds)+thresholds)
        return X


'''
Prediction (classifier/regressor) model
Assumes 2 hidden layers
'''

class PredictionModel(nn.Module):

    
    def __init__(self, input_shape, hidden_shapes, output_shape=1, **kwargs):
        
        super(PredictionModel, self).__init__()

        self.time_conditioning = kwargs['time_conditioning'] if kwargs.get('time_conditioning') else False
        self.leaky = kwargs['leaky']
        self.trelu = kwargs['trelu']
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
            if self.time_conditioning and self.trelu:
                self.relus.append(TimeReLU(data_shape=self.hidden_shapes[0], time_shape=self.time_shape, leaky=self.leaky))
            else:
                if self.leaky:
                    self.relus.append(nn.LeakyReLU())
                else:
                    self.relus.append(nn.ReLU())

            for i in range(len(self.hidden_shapes) - 1):

                self.layers.append(nn.Linear(self.hidden_shapes[i], self.hidden_shapes[i+1]))
                if self.time_conditioning and self.trelu:
                    self.relus.append(TimeReLU(data_shape=self.hidden_shapes[i+1], time_shape=self.time_shape, leaky=self.leaky))
                else:
                    if self.leaky:
                        self.relus.append(nn.LeakyReLU())
                    else:
                        self.relus.append(nn.ReLU())


            self.layers.append(nn.Linear(self.hidden_shapes[-1], self.output_shape))

        self.apply(init_weights)

    
            
    def forward(self, X, times=None, logits=False):

        if self.time_conditioning:
            X = torch.cat([X, times.view(-1, 1)], dim=-1)

        if self.time2vec is not None:
            times = self.time2vec(times)

        if self.time_conditioning and self.trelu:
            X = self.relus[0](self.layers[0](X), times)
        else:
            X = self.relus[0](self.layers[0](X))

        for i in range(1, len(self.layers)-1):

            X = self.layers[i](X)
            if self.time_conditioning and self.trelu:
                X = self.relus[i](X, times)
            else:
                X = self.relus[i](X)

        X = self.layers[-1](X)
        #if self.regress:
        #   X = torch.relu(X)
        #else:
        #   X = torch.softmax(X,dim=1)
        
        if not logits:
            if self.output_shape > 1:
                X = torch.softmax(X, dim=-1)
            else:
                X = torch.sigmoid(X)
        
        return X        



class ElecModel(nn.Module):
    # Similar in size to the CIDA model!
    def __init__(self, data_shape, hidden_shape, out_shape, **kwargs):
        
        super(ElecModel,self).__init__()

        self.time_dim = 1
        self.using_t2v = kwargs['time2vec'] if kwargs.get('time2vec') else False
        self.append_time = kwargs['append_time'] if kwargs.get('append_time') else False
        self.time_conditioning = kwargs['time_conditioning'] if kwargs.get('time_conditioning') else False

        if self.using_t2v:
            # self.using_t2v = True
            self.time_dim = 16
            self.t2v = Time2Vec(1, self.time_dim)


        if not self.append_time:
            self.time_net = nn.Linear(self.time_dim,16)
        self.layer_0 = nn.Linear(data_shape, hidden_shape)

        nn.init.kaiming_normal_(self.layer_0.weight)
        nn.init.zeros_(self.layer_0.bias)
        self.relu_0 = TimeReLU(hidden_shape, self.time_dim, True,self.time_conditioning,deep=True)


        self.layer_3 = nn.Linear(hidden_shape, out_shape)
        nn.init.kaiming_normal_(self.layer_3.weight)
        nn.init.zeros_(self.layer_3.bias)


    def forward(self, X, times,logits=False):
        
        if len(times.size()) == 3:  
            times = times.squeeze(1)

            

        if self.append_time:
            # X_ = self.layer_data(X)
            # times = self.time_net(times)
            X = torch.cat([X, times], dim=1)

        if self.using_t2v:
            times = self.t2v(times)

        X = self.relu_0(self.layer_0(X), times)
        X = self.layer_3(X)


        if not logits:
            X = torch.sigmoid(X)
        return X







class ClassifyNetHuge(nn.Module):
    '''Prediction model for the housing dataset
    
    '''
    def __init__(self,input_shape=32,hidden_shapes=[400,400],output_shape=1, **kwargs):
        super(ClassifyNetHuge, self).__init__()
        assert(len(hidden_shapes) >= 0)

        self.time_conditioning = kwargs['time_conditioning'] if kwargs.get('time_conditioning') else False
        if self.time_conditioning:
            self.leaky = kwargs['leaky'] if kwargs.get('leaky') else False
        use_time2vec = kwargs['use_time2vec'] if kwargs.get('use_time2vec') else False
        self.regress = kwargs['task'] == 'regression' if kwargs.get('task') else False
        self.time_shape = 1
        self.append_time = kwargs['append_time'] if kwargs.get('append_time') else False

        self.trelu_limit = kwargs['trelu_limit'] if kwargs.get('trelu_limit') else 1000
        self.single_trelu = kwargs['single_trelu'] if kwargs.get('single_trelu') else False

        if use_time2vec:
            self.time_shape = 8
            self.time2vec = Time2Vec(1,8)
        else:
            self.time_shape = 1
            self.time2vec = None

        self.layers = nn.ModuleList()
        self.relus = nn.ModuleList()

        self.input_shape = input_shape
        self.hidden_shapes = hidden_shapes
        self.output_shape = output_shape
        
        trelu_counter = 0
        num_layers = len(self.hidden_shapes) + 1
        self.trelu_limit = num_layers - self.trelu_limit - 1
        if len(self.hidden_shapes) == 0:

            self.layers.append(nn.Linear(input_shape, output_shape))
            if self.time_conditioning and trelu_counter > self.trelu_limit:
                self.relus.append(TimeReLU(data_shape=output_shape,time_shape=self.time_shape, leaky=self.leaky,deep=True))
            else:
                self.relus.append(nn.LeakyReLU())
            trelu_counter += 1

        else:
            if use_time2vec:
                self.layers.append(nn.Linear(self.input_shape, 3*self.hidden_shapes[0]//4))
                self.time2vec_linear = nn.Sequential(self.time2vec,nn.Linear(8,self.hidden_shapes[0]//4))
            else:
                self.layers.append(nn.Linear(self.input_shape, self.hidden_shapes[0]))

            if self.time_conditioning and trelu_counter > self.trelu_limit:
                self.relus.append(TimeReLU(data_shape=self.hidden_shapes[0],time_shape=self.time_shape, leaky=self.leaky,deep=True))
            else:
                self.relus.append(nn.LeakyReLU())
            trelu_counter += 1

            for i in range(len(self.hidden_shapes) - 1):

                self.layers.append(nn.Linear(self.hidden_shapes[i], self.hidden_shapes[i+1]))
                if self.time_conditioning and trelu_counter > self.trelu_limit:
                    self.relus.append(TimeReLU(data_shape=self.hidden_shapes[i+1],time_shape=self.time_shape, leaky=self.leaky,deep=True))
                else:
                    self.relus.append(nn.LeakyReLU())
                trelu_counter += 1

            self.layers.append(nn.Linear(self.hidden_shapes[-1],self.output_shape))
            if self.time_conditioning and trelu_counter > self.trelu_limit:
                self.relus.append(TimeReLU(data_shape=output_shape,time_shape=self.time_shape, leaky=self.leaky,deep=True))
            else:
                self.relus.append(nn.LeakyReLU())
            trelu_counter += 1


        self.apply(init_weights)

    def forward(self,X,times=None,logits=False):
        if self.append_time :  # i.e. we use time as an input directly as well. Note that X should not have any time features for this to work!
            X = torch.cat([X,times.view(-1,1)],dim=-1)
        if self.time2vec is not None:
            t1 = self.time2vec_linear(times).squeeze(1)
            times = self.time2vec(times)
            X = self.layers[0](X)
            X = torch.cat([X,t1],dim=-1)
        else:
            # t = times
            X = self.layers[0](X)
        for i in range(1,len(self.layers)):
            X = self.layers[i](X)
            # print(self.relus[i])
            if self.time_conditioning:
                X = self.relus[i](X,times)
            else:
                X = self.relus[i](X)


        if self.regress:
            X = torch.relu(X)
        else:
            X = torch.softmax(X,dim=1)


        return X        


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        #self.relu = TimeReLUCNN()

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, stride=1, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        #print('Shapes')
        #print(x.shape)
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        #print(out.shape)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, output_dim=10,**kwargs):
        super(ResNet, self).__init__()

        use_t2v = kwargs['use_time2vec'] if kwargs.get('use_time2vec') else False

        if use_t2v:
            self.time_shape = 16
        else:
            self.time_shape = 1

        self.in_channels = 16
        self.append_time = kwargs['append_time'] if kwargs.get('append_time') else False
        self.trelu_limit = kwargs['trelu_limit'] if kwargs.get('trelu_limit') else 1000
        self.single_trelu = kwargs['single_trelu'] if kwargs.get('single_trelu') else False

        self.conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.4)
        
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1])
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.layer4 = self.make_layer(block, 128, layers[3], 2)
        
        self.avg_pool = nn.AvgPool2d(2)
        self.fc_time = nn.Linear(self.time_shape, 128 * 7 * 7)
        if self.append_time:
            self.fc1 = nn.Linear(2 * 128 * 7 * 7, 256)
        else:
            self.fc1 = nn.Linear(128 * 7 * 7, 256)

        self.fc2 = nn.Linear(256, output_dim)
        

        if use_t2v:
            self.t2v = Time2Vec(1, self.time_shape)
        else:
            self.time_shape = 1
            self.t2v = None
        
        self.use_time_relu = kwargs['time_conditioning'] if kwargs.get('time_conditioning') else False

        self.relu_conv1 = TimeReLUCNN(16, self.time_shape, True,self.use_time_relu and self.trelu_limit > 4)
        self.relu_conv2 = TimeReLUCNN(32, self.time_shape, True,self.use_time_relu and self.trelu_limit > 3)
        self.relu_conv3 = TimeReLUCNN(64, self.time_shape, True,self.use_time_relu and self.trelu_limit > 2)
        self.relu_conv4 = TimeReLUCNN(128, self.time_shape, True,self.use_time_relu and self.trelu_limit > 1)
        self.relu_fc1 = TimeReLU(256, self.time_shape, True,self.use_time_relu and self.trelu_limit > 0,deep=True)




    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, times=None,logits=False, delta=0.0):
        if self.t2v is not None:
            times = self.t2v(times)
        if self.append_time:
            times_ = self.fc_time(times)
        out = self.conv(x)
        out = self.bn(out)
        out = self.layer1(out)

        out = self.relu_conv1(out, times)
        out = self.dropout(out)
        out = self.layer2(out)
        out = self.relu_conv2(out, times)
        out = self.dropout(out)
        out = self.layer3(out)
        out = self.relu_conv3(out, times)
        out = self.dropout(out)
        out = self.layer4(out)
        out = self.relu_conv4(out, times)
        out = out.view(out.size(0), -1)
        if self.append_time:
            times_ = times_.view(times_.size(0),-1)
            out = torch.cat([out, times_], dim=1)

        out = self.fc1(out)
        out = self.relu_fc1(out, times)
        out = self.fc2(out)


        if not logits:
            out = torch.softmax(out,dim=-1)
        return out








class MLP_house(nn.Module):
    """docstring for MLP_moons"""
    def __init__(self, out_shape=1, data_shape=31, hidden_shape=400, **kwargs):
        super(MLP_house, self).__init__()
        self.time_dim = 1
        self.time_conditioning = True

        self.layer_0 = nn.Linear(data_shape, hidden_shape)

        nn.init.kaiming_normal_(self.layer_0.weight)
        nn.init.zeros_(self.layer_0.bias)
        self.relu_0 = TimeReLU(hidden_shape, self.time_dim, True,self.time_conditioning,deep=True)
        self.bn0 = nn.BatchNorm1d(hidden_shape)

        self.layer_1 = nn.Linear(hidden_shape, hidden_shape)
        nn.init.kaiming_normal_(self.layer_1.weight)
        nn.init.zeros_(self.layer_1.bias)
        self.relu_1 = TimeReLU(hidden_shape, self.time_dim, True,self.time_conditioning,deep=True)
        self.bn1 = nn.BatchNorm1d(hidden_shape)

        self.layer_2 = nn.Linear(hidden_shape, hidden_shape)
        nn.init.kaiming_normal_(self.layer_2.weight)
        nn.init.zeros_(self.layer_2.bias)
        self.relu_2 = TimeReLU(hidden_shape, self.time_dim, True,self.time_conditioning,deep=True)
        self.bn2 = nn.BatchNorm1d(hidden_shape)

        self.layer_3 = nn.Linear(hidden_shape, hidden_shape)
        nn.init.kaiming_normal_(self.layer_3.weight)
        nn.init.zeros_(self.layer_3.bias)
        self.relu_3 = TimeReLU(hidden_shape, self.time_dim, True,self.time_conditioning,deep=True)
        self.bn3 = nn.BatchNorm1d(hidden_shape)

        self.layer_4 = nn.Linear(hidden_shape, out_shape)
        nn.init.kaiming_normal_(self.layer_3.weight)
        nn.init.zeros_(self.layer_3.bias)

    def forward(self,X,times,**kwargs):
        X = torch.cat([X,times.view(-1,1)],dim=-1)
        X = self.bn0(self.relu_0(self.layer_0(X),times))
        X = self.bn1(self.relu_1(self.layer_1(X),times))
        X = self.bn2(self.relu_2(self.layer_2(X),times))
        X = self.layer_4(X)

        return X


class ONPModel(nn.Module):

    def __init__(self, data_shape=59, hidden_shapes=[200], out_shape=1, time_conditioning=False, trelu=False, time2vec=False):
        
        super(ONPModel,self).__init__()

        self.time_dim = 1
        self.using_t2v = False
        self.out_shape = out_shape
        self.time_conditioning = time_conditioning
        self.trelu = trelu
        if time2vec:
            self.using_t2v = True
            self.time_dim = 8
            self.t2v = Time2Vec(1, self.time_dim)

        self.layer_0 = nn.Linear(data_shape, hidden_shapes[0])
        self.relu_0 = TimeReLU(hidden_shapes[0], self.time_dim, True)
        self.bn0 = nn.BatchNorm1d(hidden_shapes[0])
        self.layer_1 = nn.Linear(hidden_shapes[0], out_shape)

        self.apply(init_weights)

    def forward(self, X, times, logits=False):
        
        X = torch.cat([X, times.view(-1, 1)], dim=1)
        if self.using_t2v:
            times = self.t2v(times)
        X = self.relu_0(self.layer_0(X), times)
        #X = F.leaky_relu(self.layer_1(X))

        X = self.layer_1(X)

        if not logits:
            if self.out_shape > 1:
                X = torch.softmax(X, dim=-1)
            else:
                X = torch.sigmoid(X)
        return X

class M5Model(nn.Module):

    def __init__(self, data_shape=75, hidden_shape=50, out_shape=1, time_conditioning=True, trelu=True, time2vec=True):

        super(M5Model,self).__init__()

        self.time_dim = 1
        self.using_t2v = False
        self.trelu = trelu
        self.time_conditioning = time_conditioning
        if time2vec:
            self.using_t2v = True
            self.time_dim = 8
            self.t2v = Time2Vec(1, self.time_dim)

        self.layer_0 = nn.Linear(data_shape, hidden_shape)
        if self.time_conditioning and self.trelu:
            self.relu_0 = TimeReLU(hidden_shape, self.time_dim, True)
        else:
            self.relu_0 = nn.LeakyReLU()

        self.layer_1 = nn.Linear(hidden_shape, hidden_shape)

        if self.time_conditioning and self.trelu:
            self.relu_1 = TimeReLU(hidden_shape, self.time_dim, True)
        else:
            self.relu_1 = nn.LeakyReLU()

        self.layer_3 = nn.Linear(hidden_shape, out_shape)

        self.apply(init_weights)

    def forward(self, X, times, logits=False):

        if self.time_conditioning:
            X = torch.cat([X, times.view(-1, 1)], dim=-1)

        if self.using_t2v:
            times = self.t2v(times)
        if self.trelu:
            X = self.relu_0(self.layer_0(X), times)
        else:
            X = self.relu_0(self.layer_0(X))

        if self.trelu:
            X = self.relu_1(self.layer_1(X), times)
        else:
            X = self.relu_1(self.layer_1(X))
        X = self.layer_3(X)

        return X
