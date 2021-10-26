# Predictive Domain Adaptation with AdaGraph 
This code is adapted from the official PyTorch code of [AdaGraph: Unifying Predictive and ContinuousDomain Adaptation through Graphs](http://research.mapillary.com/img/publications/CVPR19b.pdf).
![alt text](https://raw.githubusercontent.com/mancinimassimiliano/adagraph/master/img/teaser.png)

This version has been tested on:
* PyTorch 1.0.1
* python 3.5
* cuda 9.2

## Installation
To install all the dependencies, please run:
```
pip install -r requirements.txt
```

## Running the code
Run `python3 main_all_source.py --suffix test --dataset <DS> --seed <SEED>` where `<DS>` is a dataset. The output will be written to `logs/log_adagraph_<DS>`.
