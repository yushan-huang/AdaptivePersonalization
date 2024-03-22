
# Towards Low-Energy Adaptive Personalization for Resource-Constrained Devices
This repository includes the code required to reproduce the experiments and figures in the paper:

Yushan Huang, Josh Millar, Yuxuan Long, Yuchen Zhao, Hamed Haddadi. "Towards Low-Energy Adaptive Personalization for Resource-Constrained Devices." Accepted to *[The 4th Workshop on Machine Learning and Systems (EuroMLSys '24), co-located with EuroSys '24](https://euromlsys.eu/).* [Paper]().

## 1. Requirements
To get started and download all dependencies, run:

```
pip install -r requirements.txt 
```

In addition, prepare the two datasets:

[Cifar10-C](https://github.com/RobustBench/robustbench)[1]. 
[Living17](https://github.com/MadryLab/BREEDS-Benchmarks)[2].

## 2. Motivation Experiments
<img src="./figure/motivation_result.png" width="800"> 

The code is in `./motivation_exp`.
First, train the model, shown as `./motivation_exp/train_origin_resnet.py`. We also release the model utilised in our paper, please refer to `./motivation_exp/resnet26_model.pth`




[1] Croce, F., Andriushchenko, M., Sehwag, V., Debenedetti, E., Flammarion, N., Chiang, M., Mittal, P. and Hein, M., 2020. Robustbench: a standardized adversarial robustness benchmark. arXiv preprint arXiv:2010.09670.

[2] Santurkar, S., Tsipras, D. and Madry, A., 2020. Breeds: Benchmarks for subpopulation shift. arXiv preprint arXiv:2008.04859.




