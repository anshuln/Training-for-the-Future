

from easydict import EasyDict
from model import set_default_args, print_args
from model import SO, ADDA, DANN, CUA, CIDA, PCIDA
from torch.utils.data import DataLoader
from model import RotateMNISTDiscrete
import os 
import torch 
import random 
import numpy as np
import argparse

opt = EasyDict()
# choose the method from ["CIDA", "PCIDA", "SO", "ADDA", "DANN" "CUA"]
opt.model = "CIDA"
# choose run on which device ["cuda", "cpu"]
opt.device = "cuda"
set_default_args(opt)
print_args(opt)
# build dataset and data loader

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
        
    parser.add_argument('--seed',type=int,default=5)
        
    args = parser.parse_args()
            
    seed = int(args.seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train_dataset = RotateMNISTDiscrete(fname='../../../data/MNIST/processed',domain_indices=[x for x in range(4)], l_domain_mask=[1]*4)
    test_dataset  = RotateMNISTDiscrete(fname='../../../data/MNIST/processed',domain_indices=[4])
    train_dataloader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=opt.batch_size,
        num_workers=4,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        shuffle=True,
        batch_size=opt.batch_size,
        num_workers=4,
    )
    # build model
    model_pool = {
        'SO': SO,
        'CIDA': CIDA,
        'PCIDA': PCIDA,
        'ADDA': ADDA,
        'DANN': DANN,
        'CUA': CUA,
    }
    model = model_pool[opt.model](opt)
    model = model.to(opt.device)


    best_acc_target = 0
    if not opt.continual_da:
        # Single Step Domain Adaptation
        for epoch in range(opt.num_epoch):
            model.learn(epoch, train_dataloader)
            if (epoch ) % 10 == 0:
                acc_target = model.eval_mnist(test_dataloader)
                if acc_target > best_acc_target:
                    best_acc_target = acc_target
                    print('Best acc target. saved.')
                    model.save(seed)
    else:
        # continual DA training
        continual_dataset = ContinousRotateMNIST()

        print('===> pretrain the classifer')
        model.prepare_trainer(init=True)
        for epoch in range(opt.num_epoch_pre):
            model.learn(epoch, train_dataloader, init=True)
            if (epoch + 1) % 10 == 0:
                model.eval_mnist(test_dataloader)
        print('===> start continual DA')
        model.prepare_trainer(init=False)
        for phase in range(opt.num_da_step):
            continual_dataset.set_phase(phase)
            print(f'Phase {phase}/{opt.num_da_step}')
            print(f'#source {len(continual_dataset.ds_source)} #target {len(continual_dataset.ds_target[phase])} #replay {len(continual_dataset.ds_replay)}')
            continual_dataloader = DataLoader(
                dataset=continual_dataset,
                shuffle=True,
                batch_size=opt.batch_size,
                num_workers=4,
            )
            for epoch in range(opt.num_epoch_sub):
                model.learn(epoch, continual_dataloader, init=False)
                if (epoch + 1) % 10 == 0:
                    model.eval_mnist(test_dataloader)

            target_dataloader = DataLoader(
                dataset=continual_dataset.ds_target[phase],
                shuffle=True,
                batch_size=opt.batch_size,
                num_workers=4,
            )
            acc_target = model.eval_mnist(test_dataloader)
            if acc_target > best_acc_target:
                print('Best acc target. saved.')
                best_acc_target = acc_target
                model.save(seed)
            data_tuple = model.gen_data_tuple(target_dataloader)
            continual_dataset.ds_replay.update(data_tuple)


    log = open("log_cida_mnist.txt","a")
    print("Seed - {}".format(seed),file=log)
    print("Num dom - {}".format(args.num_dom),file=log)
    print(best_acc_target,file=log)