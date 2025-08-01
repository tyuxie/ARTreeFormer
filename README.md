### <div align="center"> ARTreeFormer: A Faster Attention-based Autoregressive Model for Phylogenetic Inference <div> 

<div align="center">
  <a href="https://arxiv.org/abs/2507.18380"><img src="https://img.shields.io/static/v1?label=Paper&message=arXiv&color=red&logo=arxiv"></a> &ensp;
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a> &ensp;
</div>

<div align="center">
Tianyu Xie, Yicong Mao, Cheng Zhang
</div>

## Installation

This repository is an implementation of ARTreeFormer, and will run automatically on the CUDA device if a CUDA device is found available and on the CPUs otherwise. To create the torch environment, use the following command:
```
conda env create -f environment.yaml
conda activate artreeformer
```

## Data Preparation for TDE
Before conducting the experiments of tree topology density estimation (TDE), please first construct the training data using the following command
```
python -c """from datasets import process_data; process_data($DATASET, $REP_ID)"""
```
and the ground truth using the following command
```
python -c """from datasets import process_data; process_data($DATASET, 'emp')"""
```
The ```$DATASET``` is a string value refering to the name of the dataset, and the ```$REP_ID``` is an integer indicating the index of the phylogenetic analysis (since multiple analysis is a common practice to derive reliable results).

Note that these commands automatically constructs the indexes of decisions when building the tree topologies for DS1-8 (which are standard benchmarks in phylogenetic inference, see [VBPI](https://github.com/zcrabbit/vbpi), [ARTree](https://github.com/tyuxie/ARTree), etc).
One may also manually construct their own training set, as long as there exists a tree file:
- ```data/short_run_data/$DATASET/rep_$REP_ID/$DATASET.trprobs``` and 
- ```data/raw_data/$DATASET/rep_$REP_ID/$DATASET.trprobs```.

## The Maximum Parsimony Task
To reproduce the result on the maximum parsimony task, please run the following command.
```
cd ./MaxParsimony
python main.py data.dataset=DS1 base.mode=train tree.tree_type=transformer optimizer.warm_start_interval=200000
```
You may freely modify the ```data.dataset```, as well as other critical parameters as listed in the file  ```MaxParsimony/config.yaml```.
Upon finishing the training, please run the following command if you would like to compute the marginal likelihood lower bound estimate.
```
cd MaxParsimony
python main.py data.dataset=DS1 base.mode=test tree.tree_type=transformer
```


## The Tree Topology Density Estimation Task
To reproduce the results on the tree topology density estimation task, please run the following command.
```
cd ./TDE
python main.py data.dataset=$DATASET data.repo=$REP_ID base.mode=train
```
This codebase also support monitoring the KL divergence to the ground truth during training by specifying ```data.empFreq=True```.

Once the training is finished, you can compute the KL divergence to the ground truth by
```
cd ./TDE
python main.py data.dataset=$DATASET data.repo=$REP_ID base.mode=test
```


## The Variational Bayesian Phylogenetic Inference Task
To reproduce the results on the variational Bayesian phylogenetic inference task, please run the following command.
```
cd ./VBPI
python main.py data.dataset=$DATASET base.mode=train
```
This codebase also support monitoring the KL divergence to the ground truth during training by specifying ```data.empFreq=True```.

Once the training is finished, you can use the following command to compute the marginal likelihood on training set by
```
cd ./VBPI
python main.py data.dataset=$DATASET base.mode=test
```
and compute the KL divergence to the ground truth by
```
cd ./VBPI
python main.py data.dataset=$DATASET base.mode=test data.empFreq=True
```

## Reference
Please consider citing our work if you find this codebase useful.
```
@article{xie2025artreeformer,
  title={ARTreeFormer: A Faster Attention-based Autoregressive Model for Phylogenetic Inference},
  author={Xie, Tianyu and Mao, Yicong and Zhang, Cheng},
  journal={arXiv preprint arXiv:2507.18380},
  year={2025}
}
```