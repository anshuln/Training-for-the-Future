import torch
import torch.nn as nn
import torch.nn.functional as F

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

        #print(self.downsample)
        #print(out.shape, residual.shape)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, output_dim=10,**kwargs):
        super(ResNet, self).__init__()

        self.time_conditioning = kwargs['time_conditioning'] if kwargs.get('time_conditioning') else False
        self.leaky = kwargs['leaky'] if kwargs.get('leaky') else False
        self.append_time = kwargs['append_time'] if kwargs.get('append_time') else False
        self.use_t2v = kwargs['use_time2vec'] if kwargs.get('use_time2vec') else False

        if self.use_t2v:
            self.time_shape = 16
        else:
            self.time_shape = 1

        self.in_channels = 16
        self.append_time = kwargs['append_time'] if kwargs.get('append_time') else False

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

        self.fc2 = nn.Linear(256, 10)
        

        if self.use_t2v:
            self.t2v = Time2Vec(1, self.time_shape)
        else:
            self.time_shape = 1
            self.t2v = None
        
        self.use_time_relu = kwargs['time_conditioning'] if kwargs.get('time_conditioning') else False
        self.relu_conv1 = nn.LeakyReLU()
        self.relu_conv2 = nn.LeakyReLU()
        self.relu_conv3 = nn.LeakyReLU()
        self.relu_conv4 = nn.LeakyReLU()
        self.relu_fc1 = nn.LeakyReLU()

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

    def forward(self, x, times=None,logits=False):
        #times_ = times.unsqueeze(2).repeat(1,28,28)[:, None, :, :]
        #x = torch.cat([x, times_], dim=1)
        if self.t2v is not None:
            times = self.t2v(times)
        if self.append_time:
            times_ = self.fc_time(times)
        out = self.conv(x)
        out = self.bn(out)
        out = self.layer1(out)
        
        if self.time_conditioning:
            out = self.relu_conv1(out, times)
        else:
            out = F.relu(out)

        out = self.dropout(out)
        out = self.layer2(out)

        if self.time_conditioning:
            out = self.relu_conv2(out, times)
        else:
            out = F.relu(out)

        out = self.dropout(out)
        out = self.layer3(out)

        if self.time_conditioning:
            out = self.relu_conv3(out, times)
        else:
            out = F.relu(out)
        
        out = self.dropout(out)
        out = self.layer4(out)
        
        if self.time_conditioning:
            out = self.relu_conv4(out, times)
        else:
            out = F.relu(out)
        #print('L4:',out.shape)
        #print('Out_shape:', out.shape)
        #out = self.avg_pool(out)
        #print(out.shape)
        out = out.view(out.size(0), -1)
        if self.append_time:
            times_ = times_.view(times_.size(0),-1)
            out = torch.cat([out, times_], dim=1)

        #print('Out_shape:', out.shape)
        out = self.fc1(out)

        if self.time_conditioning:
            out = self.relu_fc1(out, times)
        else:
            out = F.relu(out)

        out = self.fc2(out)
        #if not logits:
        #    out = torch.softmax(out,dim=-1)
        return out


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
