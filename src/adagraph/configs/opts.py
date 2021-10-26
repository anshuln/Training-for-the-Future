# OPTS FOR COMPCARS
import argparse

parser = argparse.ArgumentParser(description='AdaGraph')
parser.add_argument('--dataset', default = 'compcars', help='Dataset to test (compcars, portraits)')
parser.add_argument('--seed',type=int,default=1)
# parser.add_argument('--network', default='resnet', type=str, help='Network to use (resnet, decaf)')
# parser.add_argument('--skip', default=None, type=str, help='Skip some settings (required only for portraits, eventually). Options are: regions,decades.')
parser.add_argument('--suffix', default='./logs/adagraph_test', type=str, help='Suffix to give for storing the experiments')
# parser.add_argument('--num_dom', type=int, help='Number of train domains')

args = parser.parse_args()

SEED = args.seed
DEVICE='cuda'
DATASET = args.dataset
RESIDUAL = args.dataset == 'mnist'
MLP = args.dataset in ['moons','elec','onp','house','m5house']
SKIP = None #args.skip
SUFFIX = args.suffix
SOURCE_GROUP = ['conv','bn','layer','downsample','fc']
STD = [0.229, 0.224, 0.225]
SIZE = 28

if DATASET == 'compcars':
	REGRESSION = False
	from configs.config_compcars import *

elif DATASET == 'portraits':
	REGRESSION = False
	from configs.config_portraits import *

elif DATASET == 'mnist':
	REGRESSION = False
	from configs.config_mnist import *

elif DATASET == 'moons':
	REGRESSION = False
	from configs.config_moons import *

elif DATASET == 'elec':
	REGRESSION = False
	from configs.config_elec import *

elif DATASET == 'onp':
	REGRESSION = False
	from configs.config_onp import *


elif DATASET == 'house':
	REGRESSION = True
	from configs.config_house import *

elif DATASET == 'm5house':
	REGRESSION = True
	from configs.config_m5house import *


else:
	print("Please specify a valid dataset in [mnist,onp,elec,house,m5house,moons]")
	exit(1)

TRAINING_GROUP = ['bn','downsample.1']
# if not RESIDUAL:
#   EPOCHS = 10
#   STEP = 7
#   STD = [1./256, 1./256, 1./256]
#   SIZE = 227
#   LR = 0.001
#   SOURCE_GROUP = ['bn','final']

if SKIP == 'regions':
	def skip_rule(meta_source, meta_target):
		source_year, source_region = meta_source
		target_year, target_region = meta_target
		return source_year != target_year
elif SKIP== 'decades':
	def skip_rule(meta_source, meta_target):
		source_year, source_region = meta_source
		target_year, target_region = meta_target
		return source_region != target_region
else:
	def skip_rule(meta_source, meta_target):
		return False
