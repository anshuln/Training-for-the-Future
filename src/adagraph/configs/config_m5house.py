import torch
import torchvision.transforms as transforms
from dataloaders.m5house import M5house, M5Sampler


# OPTS FOR COMPCARS
BATCH_SIZE = 2048
TEST_BATCH_SIZE = 100

EPOCHS= 30
STEP=20
LR=0.01
DECAY=0.000001
MOMENTUM=0.1
BANDWIDTH=1


DATES=['2009','2010','2011','2012','2013','2014']
VIEWS=['1','2','3','4','5']

CLASSES = 1

DOMAINS = [[x for x in range(37)]]
ACTUAL_TARGET_DOMAIN = 36
SOURCE_DOMAINS = [x for x in range(36)]
NUM_META = 1

DATALIST='../../data/M5/processed_household/'


def domain_converter(meta):
    year = meta
    if isinstance(year,tuple):
        year = year[0]
    # year = int(year)-int(DATES[0])
    # viewpoint = int(viewpoint)-int(VIEWS[0])
    # print(year)
    return year #viewpoint*len(DATES)+year


def init_loader(bs, domains=[], shuffle=False, auxiliar= False, size=224, std=[0.229, 0.224, 0.225]):
    data_transform=transforms.Compose([
            transforms.Resize((size,size)),
                      transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], std)
         ])

    dataset = M5house(DATALIST,domains=domains)
    print("Dataloader with %d samples"%len(dataset))

    if not auxiliar:
        return torch.utils.data.DataLoader(dataset, batch_size=bs, drop_last=False,num_workers=4, shuffle=shuffle)

    else:
        return torch.utils.data.DataLoader(dataset, batch_size=bs, drop_last=False,num_workers=4, sampler=M5Sampler(dataset,bs))


def compute_edge(x,dt,idx, self_connection = 1.):
    x = x
    dt =dt
    edge_w=torch.exp(-torch.pow(torch.norm(x.view(1,-1)-dt,dim=1),2)/(2.*BANDWIDTH))
    edge_w[idx]=edge_w[idx]*self_connection
    return edge_w/edge_w.sum()


def get_meta_vector(meta):
    angle = meta
    return torch.FloatTensor(angle)
