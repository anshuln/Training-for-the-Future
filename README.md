# Training for the Future: A Simple Gradient Interpolation Loss to Generalize Along Time

Code accompanying the NeurIPS 2021 article 

> **Training for the Future: A Simple Gradient Interpolation Loss to Generalize Along Time**
>
> Anshul Nasery*, Soumyadeep Thakur*, Vihari Piratla, Abir De, Sunita Sarawagi

This repository contains the training and inference code, as well as codebases for different baselines.

The code and instructions to run for all models can be found in `src/`. The processed datasets should be downloaded to `data/` from [this link](https://drive.google.com/drive/folders/11F7FJYPq0mlL11pSI_FaRKOjqCdmxEnl?usp=sharing)

## Setup
* Install `torch>=1.4` and corresponding `torchvision`. Also install `tqdm`, `numpy`, `sklearn`.
* Install the POT library following [this link](https://pythonot.github.io/).
* Install `pkg-resources==0.0.0`, `six==1.12.0`, `Pillow==8.1.1`.
* Download the datasets into the `data/` directory from from [this link](https://drive.google.com/drive/folders/11F7FJYPq0mlL11pSI_FaRKOjqCdmxEnl?usp=sharing)

## Execution
The file `src/<<MODEL>/main.py` is usually the entry-point for starting a training and inference job for each `<MODEL>`. The standard way to run this file is `python3 main.py --data <DS> --model <MODEL> --seed <SEED>`. However, there are minor differences as illustrated in the files `src/<MODEL>/README.md`. The results are written to `src/<MODEL>/log_<MODEL>_<DS>` for each run. 


## Code Overview
The directory `src/` has four sub-folders, for our method and baselines
* `GI/`
    * `main.py` - Entrypoint to the code
    * `trainer_GI.py` - Contains the training algorithm implementation
    * `config_GI.py` - Contains the hyperparameter configurations for different datasets and models
    * `preprocess.py` - Can be used to generate the processed datasets from raw files

* `CIDA/`
    * `main.py` - Entrypoint to the code, contains dataloader and training algorithm
    * `<DS>_models.py` - Contains model definition and hyperparameters for the dataset `<DS>`.

* `CDOT/`
    * `ot_main.py` - Entrypoint to the code, contains the training algorithm implementation, contains the hyperparameter configurations for different datasets and models
    * `transport.py`, `regularized_OT.py` - Contain implementations of the OT and CDOT algorithms

* `adagraph/`
    * `main_all_source.py` - Entrypoint to the code
    * `configs/` - Contains hyperparams for various datasets
    * `dataloaders/` - Contains dataloaders for various datasets
    * `models/` - Contains model definitions

## Citation
If you find the paper or the code helpful in your research, consider citing us as
```
@inproceedings{
nasery2021training,
title={Training for the Future: A Simple Gradient Interpolation Loss to Generalize Along Time},
author={Anshul Nasery and Soumyadeep Thakur and Vihari Piratla and Abir De and Sunita Sarawagi},
booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
year={2021},
url={https://openreview.net/forum?id=U7SBcmRf65}
}
```
