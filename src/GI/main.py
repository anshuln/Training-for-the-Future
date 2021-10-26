'''Main script to run stuff

[description]
'''

import argparse
import os
import random
from trainer_GI import *
from preprocess import *

device = "cuda:0"

def main(args):

	if args.use_cuda:
		args.device = "cuda:0"
	else:
		args.device = "cpu"

	if args.preprocess:
		if args.data == "mnist":
			print("Preprocessing")
			load_Rot_MNIST(args.encoder)
		if args.data == "moons":
			load_moons(11,args.model)
		if args.data == "house":
			load_house_price(args.model)
		if args.data == "house_classifier":
			load_house_price_classification()             
	if args.seed is not None:
		seed = int(args.seed)
		random.seed(seed)
		os.environ['PYTHONHASHSEED'] = str(seed) 
		np.random.seed(seed)
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
		torch.backends.cudnn.benchmark = False
		torch.backends.cudnn.deterministic = True

		# torch.random.manual_seed(int(args.seed))
		# np.random.seed(int(args.seed))

	trainer = GradRegTrainer(args)
	trainer.train()




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	""" Arguments: arg """
	parser.add_argument('--model',help="String, needs to be one of baseline,tbaseline,goodfellow,GI")
	parser.add_argument('--data',help="String, needs to be one of mnist, sleep, moons, cars")
	parser.add_argument('--epoch_finetune',default=5,help="Needs to be int, number of epochs for transformer/ordinal classifier",type=int)
	parser.add_argument('--epoch_classifier',default=5,help="Needs to be int, number of epochs for classifier",type=int)
	parser.add_argument('--bs',default=100,help="Batch size",type=int)
	parser.add_argument('--aug_steps',default=10,help="Number of steps of data augmentation to do",type=int)
	parser.add_argument('--early_stopping',action='store_true',help="Early Stopping for finetuning")
	parser.add_argument('--use_cuda',action='store_true',help="Should we use a GPU")
	parser.add_argument('--preprocess',action='store_true',help="Do we pre-process the data?")

	parser.add_argument('--seed',default=2)
	parser.add_argument('--trelu_limit',default=1000,type=int)
	parser.add_argument('--pretrained',action='store_true',help='Should we load a model?')
	
	args = parser.parse_args()
	# print("seed - {} model - {}".format(args.seed,args.model),file=open('testing_{}_{}_{}.txt'.format(args.model,args.data,args.trelu_limit),'a'))
	main(args)
