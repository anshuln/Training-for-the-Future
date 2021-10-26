from losses import *



# TODO 

class Config():
    def __init__(self,args):
        self.epoch_classifier = args.epoch_classifier
        self.epoch_finetune = args.epoch_finetune 
        self.SUBEPOCHS = 1
        self.EPOCH = args.epoch_finetune // self.SUBEPOCHS
        self.bs = args.bs
        self.CLASSIFICATION_BATCH_SIZE = 100

        self.data = args.data 
        self.update_num_steps = 1
        self.num_finetune_domains = 2
        self.use_pretrained = False 
        self.delta = 0.0
        self.max_k = 0.0
        self.schedule = False
        self.warm_start = True
        self.w_decay = None
        self.lr_reduce = 1.0

        log_file_name = './logs/log_{}_{}'.format(args.data,args.model)

        self.log = open(log_file_name,"a")
        print("seed - {}".format(args.seed),file=self.log)


        if args.data == "house":
            self.epoch_classifier = 50 #args.epoch_classifier
            self.epoch_finetune = 20 #args.epoch_finetune 
            self.bs = 100 
            self.early_stopping = True

            self.dataset_kwargs = {"root_dir":"../../data/HousePrice","device":args.device, "drop_cols":None}
            self.source_domain_indices = [6,7,8,9,10]
            self.target_domain_indices = [11]
            self.data_index_file = "../../data/HousePrice/indices.json"
            from models_GI import MLP_house
            self.classifier = MLP_house 
            output_shape = 1
            self.model_kwargs =  {'time_conditioning':True,'task':'regression','use_time2vec':True,'leaky':True,"input_shape":31,"hidden_shapes":[400,400,400],"output_shape":output_shape,'append_time':True, "trelu_limit" : args.trelu_limit}
            self.lr = 1e-3 #5e-4
            self.classifier_loss_fn =  reconstruction_loss
            self.loss_type = 'regression'
            self.encoder = None

            self.delta_lr=0.3
            self.delta_clamp=0.2
            self.delta_steps=5
            self.lambda_GI=1/1.5
            self.warm_start = False

        if args.data == "mnist":

            self.epoch_classifier = 60 #args.epoch_classifier
            self.epoch_finetune = 25 #args.epoch_finetune 
            self.bs = 250

            self.early_stopping = True
            self.use_pretrained = True

            self.dataset_kwargs = {"root_dir":"../../data/MNIST/processed/","device":args.device, "drop_cols":None}
            self.source_domain_indices = [0,1,2,3]
            self.target_domain_indices = [4]
            self.data_index_file = "../../data/MNIST/processed/indices.json"
            from models_GI import ResNet, ResidualBlock
            self.classifier = ResNet 
            self.model_kwargs =  {
                                    "block": ResidualBlock,
                                    "layers": [2, 2, 2, 2],
                                    "append_time": True,
                                    "time_conditioning": True,
                                    "use_time2vec": True,
                                    "trelu_limit" : args.trelu_limit
                                }
            self.lr = 1e-3
            self.classifier_loss_fn = classification_loss
            self.loss_type = 'classification'
            self.encoder = None

            self.delta_lr=1e-1
            self.delta_clamp=0.15
            self.delta_steps=15
            self.lambda_GI=1.0

        if args.data == 'moons':

            self.epoch_classifier = 30 #args.epoch_classifier
            self.epoch_finetune = 25 #args.epoch_finetune 
            self.bs = 200
            self.early_stopping = True

            self.dataset_kwargs = {"root_dir":"../../data/Moons/processed", "device":args.device, "drop_cols":None}
            self.source_domain_indices = [0,1, 2, 3, 4, 5, 6, 7, 8]
            self.target_domain_indices = [9]
            self.data_index_file = "../../data/Moons/processed/indices.json"
            from models_GI import PredictionModel
            self.classifier = PredictionModel
            self.model_kwargs =  {"input_shape":3, "hidden_shapes":[50, 50], "out_shape":1, "time_conditioning": True, "trelu": True, "use_time2vec":True, 
                                    "leaky":True, "regression": False}
            self.lr = 5e-3
            self.classifier_loss_fn = binary_classification_loss
            self.loss_type = 'classification'
            self.encoder = None

            self.delta_lr=0.05
            self.delta_clamp=0.5
            self.delta_steps=5
            self.lambda_GI=1.0e-4
            self.lr_reduce=10
        if args.data == 'elec':

            self.epoch_classifier = 30 #args.epoch_classifier
            self.epoch_finetune = 20 #args.epoch_finetune 
            self.bs = 1024
            self.early_stopping = False

            self.dataset_kwargs = {"root_dir":"../../data/Elec2/","device":args.device, "drop_cols":None}
            self.source_domain_indices = [x for x in range(29)]
            self.target_domain_indices = [29]
            self.data_index_file = "../../data/Elec2/indices.json"
            from models_GI import ElecModel
            self.classifier = ElecModel
            self.model_kwargs =  {"data_shape":9, "hidden_shape":128, "out_shape":1,  "time2vec":True,"append_time":True,"time_conditioning":True}
            self.lr = 5e-3
            self.classifier_loss_fn = binary_classification_loss
            self.num_finetune_domains = 2
            self.loss_type = 'classification'
            self.encoder = None


            self.delta_lr=0.005
            self.delta_clamp=0.2
            self.delta_steps=10
            self.lambda_GI=1.0
            self.warm_start = False

        if args.data == 'onp':

            self.epoch_classifier = 60 #args.epoch_classifier
            self.epoch_finetune = 20 #args.epoch_finetune 
            self.bs = 64
            self.early_stopping = True

            self.dataset_kwargs = {"root_dir":"../../data/ONP/processed","device":args.device, "drop_cols":None}
            self.source_domain_indices = [0, 1, 2, 3, 4]
            self.target_domain_indices = [5]
            self.data_index_file = "../../data/ONP/processed/indices.json"
            from models_GI import PredictionModel
            self.classifier = PredictionModel
            self.model_kwargs =  {"input_shape":59, "hidden_shapes":[200], "out_shape":1, "time_conditioning": True, "trelu": False, "use_time2vec":False, 
                                    "leaky":True, "regression": False}
            self.lr = 1e-3
            self.classifier_loss_fn = binary_classification_loss
            self.loss_type = 'classification'
            self.encoder = None

            self.delta_lr=1.0
            self.delta_clamp=0.1
            self.delta_steps=10
            self.lambda_GI=0.5e-2
            self.lr_reduce=10.0
            self.w_decay = 1e-4


        if args.data == 'm5':

            self.epoch_classifier = 30 #args.epoch_classifier
            self.epoch_finetune = 15 #args.epoch_finetune 
            self.bs = 100
            self.early_stopping = True


            self.dataset_kwargs = {"root_dir":"../../data/M5/processed","device":args.device, "drop_cols":None}
            self.source_domain_indices = [0, 1, 2]
            self.target_domain_indices = [3]
            self.data_index_file = "../../data/M5/processed/indices.json"
            from models_GI import M5Model

            self.classifier = M5Model
            self.model_kwargs = {"data_shape": 75, "hidden_shape": 50, "out_shape": 1, "time_conditioning": True, "trelu": True, "time2vec": True}
            self.lr = 1e-2
            self.classifier_loss_fn = reconstruction_loss
            self.loss_type = 'regression'
            self.encoder = None

            self.w_decay = 1e-4
            self.delta_lr=0.5
            self.delta_clamp=0.5
            self.delta_steps=5
            self.lambda_GI=1.0
            self.lr_reduce=20.0
            self.schedule = True


        if args.data == 'm5_household':

            
            self.epoch_classifier = 35 #args.epoch_classifier
            self.epoch_finetune = 20 #args.epoch_finetune 
            self.bs = 100
            self.early_stopping = True

            self.dataset_kwargs = {"root_dir":"../../data/M5/processed_household","device":args.device, "drop_cols":None}
            self.source_domain_indices = [0, 1, 2]
            self.target_domain_indices = [3]
            self.data_index_file = "../../data/M5/processed_household/indices.json"
            from models_GI import M5Model

            self.classifier = M5Model
            self.model_kwargs = {"data_shape": 75, "hidden_shape": 50, "out_shape": 1, "time_conditioning": True, "trelu": True, "time2vec": True}
            self.lr = 1e-2
            self.classifier_loss_fn = reconstruction_loss
            self.loss_type = 'regression'
            self.encoder = None

            self.w_decay = 1e-4
            self.delta_lr=0.5
            self.delta_clamp=0.5
            self.delta_steps=5
            self.lambda_GI=1.0
            self.lr_reduce=20.0
            self.schedule = True




