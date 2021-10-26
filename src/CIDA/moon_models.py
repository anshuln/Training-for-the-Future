import torch
import torch.nn as nn
import torch.nn.functional as F

class DomainEnc(nn.Module):
    def __init__(self):
        super(DomainEnc, self).__init__()
        
        self.fc1 = nn.Linear(2, 50)
        
        self.fc2 = nn.Linear(60, 60)

        self.fc3 = nn.Linear(60, 60)

        self.fc_final = nn.Linear(60, 20)

        self.fc1_var = nn.Linear(1, 10)
        

    def forward(self, x):
        x, domain = x

        # side branch for variable FC
        domain = domain.unsqueeze(1)
        #print(domain.shape)
        x_domain = F.relu(self.fc1_var(domain))

        # main branch
        x = F.relu(self.fc1(x))
        #x = self.drop1(x)

        # combine feature in the middle
        x = torch.cat((x, x_domain), dim=1)
        x = F.relu(self.fc3(F.relu(self.fc2(x))))
        #x = self.drop2(x)

        # continue main branch
        x = F.relu(self.fc_final(x))

        return x

# Predictor
class DomainPred(nn.Module):
    def __init__(self):
        super(DomainPred, self).__init__()

        self.fc1 = nn.Linear(20, 20)

        self.fc_final = nn.Linear(20, 2)

    def forward(self, x):
        x, domain = x

        x = F.relu(self.fc1(x))

        x = self.fc_final(x)
        x = F.log_softmax(x)
        return x


# Discriminator: with BN layers after each FC, dual output
class DomainDDisc(nn.Module):
    def __init__(self):
        super(DomainDDisc, self).__init__()
        self.hidden = 20
        self.dropout = 0.3

        self.drop2 = nn.Dropout(self.dropout)

        self.fc3_m = nn.Linear(self.hidden, self.hidden)
        self.bn3_m = nn.BatchNorm1d(self.hidden)
        self.drop3_m = nn.Dropout(self.dropout)

        self.fc3_s = nn.Linear(self.hidden, self.hidden)
        self.bn3_s = nn.BatchNorm1d(self.hidden)
        self.drop3_s = nn.Dropout(self.dropout)

        self.fc4_m = nn.Linear(self.hidden, self.hidden)
        self.bn4_m = nn.BatchNorm1d(self.hidden)
        self.drop4_m = nn.Dropout(self.dropout)

        self.fc4_s = nn.Linear(self.hidden, self.hidden)
        self.bn4_s = nn.BatchNorm1d(self.hidden)
        self.drop4_s = nn.Dropout(self.dropout)

        self.fc5_m = nn.Linear(self.hidden, self.hidden)
        self.bn5_m = nn.BatchNorm1d(self.hidden)
        self.drop5_m = nn.Dropout(self.dropout)

        self.fc5_s = nn.Linear(self.hidden, self.hidden)
        self.bn5_s = nn.BatchNorm1d(self.hidden)
        self.drop5_s = nn.Dropout(self.dropout)

        self.fc_final_m = nn.Linear(self.hidden, 1)
        self.fc_final_s = nn.Linear(self.hidden, 1)        

    def forward(self, x):
        x, domain = x
        domain = domain.unsqueeze(1)

        x = self.drop2(x)

        x_m = F.relu(self.bn3_m(self.fc3_m(x)))
        x_s = F.relu(self.bn3_s(self.fc3_s(x)))

        x_m = F.relu(self.bn4_m(self.fc4_m(x)))
        x_s = F.relu(self.bn4_s(self.fc4_s(x)))

        x_m = F.relu(self.bn5_m(self.fc5_m(x)))
        x_s = F.relu(self.bn5_s(self.fc5_s(x)))

        x_m = self.fc_final_m(x_m)
        x_s = self.fc_final_s(x_s) # log sigma^2

        return (x_m, x_s)
