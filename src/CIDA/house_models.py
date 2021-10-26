import torch
import torch.nn as nn
import torch.nn.functional as F


from easydict import EasyDict

args = EasyDict()
args.hidden = 800
args.dropout = 0.0
args.ratio = 1
norm = 1

class DomainEnc(nn.Module):
    def __init__(self):
        super(DomainEnc, self).__init__()
        self.hidden = args.hidden
        self.ratio = float(args.ratio)
        self.dropout = args.dropout

        self.fc1 = nn.Linear(32, self.hidden)
        self.drop1 = nn.Dropout(self.dropout)

        self.fc2 = nn.Linear(self.hidden + int(self.hidden // self.ratio), self.hidden + int(self.hidden // self.ratio))
        self.drop2 = nn.Dropout(self.dropout)

        self.fc3 = nn.Linear(self.hidden + int(self.hidden // self.ratio), self.hidden + int(self.hidden // self.ratio))
        self.drop3 = nn.Dropout(self.dropout)

        self.fc4 = nn.Linear(self.hidden + int(self.hidden // self.ratio), self.hidden + int(self.hidden // self.ratio))
        self.drop4 = nn.Dropout(self.dropout)

        self.fc_final = nn.Linear(self.hidden + int(self.hidden // self.ratio), self.hidden)

        self.fc1_var = nn.Linear(1, int(self.hidden // self.ratio))
        self.fc2_var = nn.Linear(int(self.hidden // self.ratio), int(self.hidden // self.ratio))
        self.fc3_var = nn.Linear(int(self.hidden // self.ratio), int(self.hidden // self.ratio))
        self.drop1_var = nn.Dropout(self.dropout)
        self.drop2_var = nn.Dropout(self.dropout)
        self.drop3_var = nn.Dropout(self.dropout)

    def forward(self, x):
        x, domain = x
        domain = domain.unsqueeze(1) / norm

        # side branch for variable FC
        x_domain = F.relu(self.fc1_var(domain))
        x_domain = self.drop1_var(x_domain)

        # main branch
        x = F.relu(self.fc1(x))
        x = self.drop1(x)

        # combine feature in the middle
        x = torch.cat((x, x_domain), dim=1)

        x = F.relu(self.fc2(x))
        x = self.drop2(x)

        x = F.relu(self.fc3(x))
        x = self.drop3(x)

        x = F.relu(self.fc4(x))
        x = self.drop4(x)

        # continue main branch
        x = F.relu(self.fc_final(x))

        return x

# Predictor
class DomainPred(nn.Module):
    def __init__(self):
        super(DomainPred, self).__init__()
        self.hidden = args.hidden
        self.dropout = args.dropout

        self.drop0 = nn.Dropout(self.dropout)

        self.fc1 = nn.Linear(self.hidden, self.hidden)
        self.drop1 = nn.Dropout(self.dropout)


        self.fc_final = nn.Linear(self.hidden, 1)

    def forward(self, x):
        x, domain = x
        domain = domain.unsqueeze(1) / norm

        x = self.drop0(x)

        x = F.relu(self.fc1(x))
        x = self.drop1(x)

        x = self.fc_final(x)
        
        return x


# Discriminator: with BN layers after each FC, dual output
class DomainDDisc(nn.Module):
    def __init__(self):
        super(DomainDDisc, self).__init__()
        self.hidden = args.hidden
        self.dropout = args.dropout

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

        self.fc6_m = nn.Linear(self.hidden, self.hidden)
        self.bn6_m = nn.BatchNorm1d(self.hidden)
        self.drop6_m = nn.Dropout(self.dropout)

        self.fc6_s = nn.Linear(self.hidden, self.hidden)
        self.bn6_s = nn.BatchNorm1d(self.hidden)
        self.drop6_s = nn.Dropout(self.dropout)

        self.fc7_m = nn.Linear(self.hidden, self.hidden)
        self.bn7_m = nn.BatchNorm1d(self.hidden)
        self.drop7_m = nn.Dropout(self.dropout)

        self.fc7_s = nn.Linear(self.hidden, self.hidden)
        self.bn7_s = nn.BatchNorm1d(self.hidden)
        self.drop7_s = nn.Dropout(self.dropout)

        self.fc_final_m = nn.Linear(self.hidden, 1)
        self.fc_final_s = nn.Linear(self.hidden, 1)        

    def forward(self, x):
        x, domain = x
        domain = domain.unsqueeze(1) / norm

        x = self.drop2(x)

        x_m = F.relu(self.bn3_m(self.fc3_m(x)))
        x_m = self.drop3_m(x_m)

        x_s = F.relu(self.bn3_s(self.fc3_s(x)))
        x_s = self.drop3_s(x_s)

        x_m = F.relu(self.bn4_m(self.fc4_m(x_m)))
        x_m = self.drop4_m(x_m)

        x_s = F.relu(self.bn4_s(self.fc4_s(x_s)))
        x_s = self.drop4_s(x_s)

        x_m = F.relu(self.bn5_m(self.fc5_m(x_m)))
        x_m = self.drop5_m(x_m)

        x_s = F.relu(self.bn5_s(self.fc5_s(x_s)))
        x_s = self.drop5_s(x_s)

        x_m = F.relu(self.bn6_m(self.fc6_m(x_m)))
        x_m = self.drop6_m(x_m)

        x_s = F.relu(self.bn6_s(self.fc6_s(x_s)))
        x_s = self.drop6_s(x_s)

        x_m = F.relu(self.bn7_m(self.fc7_m(x_m)))
        x_m = self.drop7_m(x_m)

        x_s = F.relu(self.bn7_s(self.fc7_s(x_s)))
        x_s = self.drop7_s(x_s)

        x_m = self.fc_final_m(x_m)
        x_s = self.fc_final_s(x_s) # log sigma^2

        return (x_m, x_s)
