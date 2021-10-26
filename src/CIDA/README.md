# Continuously Indexed Domain Adaptation (CIDA)
This is modified from the official PyTorch implementation for CIDA. This repo contains code for experiments in the **ICML 2020** paper '[Continuously Indexed Domain Adaptation](http://wanghao.in/paper/ICML20_CIDA.pdf)'.


## Environment
* Python 3.6
* PyTorch 1.5.1
* Numpy 1.19

## Running the code
Run `python3 main.py --seed <SEED> --data <DS> --cuda` where `<DS>` is one of `[moons,house,m5,m5_household,onp]`  The results are dumped to a log file `log_CIDA_<DS>` in the same folder.
For rotating MNIST, run `rot_mnist/main.py`. 
