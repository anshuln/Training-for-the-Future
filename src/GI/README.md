# Training with Gradient Interpolation

## Setup
* Needs `torch>=1.4` and corresponding `torchvision`. Also needs `tqdm`, `numpy`, `sklearn`. 
* For data setup, populate the `data` directory.

## Running the code
Run `python3 -W ignore main.py --data <DS> --model <MODEL> --use_cuda` with the dataset and model's name. `<DS>` can be one of `[mnist,moons,house,m5_household,m5,elec,onp]` and `<MODEL>` can be one of `[GI,baseline,t_baseline,inc_finetune,t_inc_finetune,t_goodfellow,t_grad_reg]`

The result is dumped to `logs/log_<DS>_<MODEL>`

For other command line args check `python3 main.py --help`
## Code overview
* The file `trainer_GI.py` contains the training algorithm in the function `adversarial_finetune`. It also implements a trainer class for boiler plate code.
* The file `config_GI.py` has the list of configs for different data sets. Change this for hyperparams
* `models_GI.py` has the model definitions, `main.py` is the entrypoint to the code

