# Few shot classification with task adaptive semantic feature learning (TasNet)

PyTorch implementation of Few shot classification with task adaptive semantic feature learning

## Dependencies
* python 3.6.5
* numpy 1.16.0
* torch 1.8.0
* tqdm 4.57.0
* scipy 1.5.4
* torchvision 0.9.0

## Abstract
Few-shot classification aims to learn a classifier that categorizes objects of unseen classes with limited samples. One general approach is to mine as much information as possible from limited samples. This can be achieved by incorporating data aspects from multiple modals. However, existing multi-modality methods only use additional modality in support samples while adhering to a single modal in query samples. Such approach could lead to information imbalance between support and query samples, which confounds model generalization from support to query samples. Towards this problem, we propose a task-adaptive semantic feature learning mechanism to incorporates semantic features for both support and query samples. The semantic feature learner is trained episodic-wisely by regressing from the feature vectors of the support samples. Then the query samples can obtain the semantic features with this module. Such method maintains a consistent training scheme between support and query samples and enables direct model transfer from support to query datasets, which significantly improves model generalization. We develop two modality combination implementations: feature concatenation and feature fusion, based on the semantic feature learner. Extensive experiments conducted on four benchmarks demonstrate that our method outperforms state-of-the-arts, proving the effectiveness of our method.

## Architecture
![Image text](https://github.com/pmhDL/TasNet/blob/main/Image/architecture.png)

## Download the Datasets
* [miniImageNet](https://drive.google.com/file/d/1fJAK5WZTjerW7EWHHQAR9pRJVNg1T1Y7/view?usp=sharing) 
* [tieredImageNet](https://drive.google.com/file/d/1Letu5U_kAjQfqJjNPWS_rdjJ7Fd46LbX/view?usp=sharing)
* [CIFAR-FS](https://drive.google.com/file/d/1GjGMI0q3bgcpcB_CjI40fX54WgLPuTpS/view?usp=sharing)
* [CUB](https://drive.google.com/file/d/1hbzc_P1FuxMkcabkgn9ZKinBwW683j45/view)
* [glove word embedding](https://nlp.stanford.edu/projects/glove/)
  
## Pretrained checkpoints for evaluation
Please specify the path of the pretrained checkpoints to "./checkpoints/[dataname]"

## Running Experiments
Run pretrain phase:
```bash
python run_pre.py
```
Run train and test phase:
```bash
If you want to train the models from scratch, please run the run_pre.py to pretrain the backbone, and then run the run_fusion.py or run_concatenation.py to train the meta learning model.
python run_fusion.py
python run_concatenation.py
```

## Acknowledgments
Our project references the codes in the following repos.
[**Meta-Transfer Learning**](https://github.com/yaoyao-liu/meta-transfer-learning)

[**Adaptive Cross-Modal Few-shot Learning**](https://github.com/ElementAI/am3)
