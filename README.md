# VSCM-GZSL

## Visual-Semantic Consistency Matching Network for Generalized Zero-shot Learning



## Requirements

The code implementation of VSCM-GZSL mainly based on [pyTorch]([PyTorch](https://pytorch.org/)). The requirements are as follows:

- python 3.6 
- torch 1.9.0

- Numpy
- Sklearn
- Scipy
- tqdm

## Training

We trained the model on five popular ZSL benchmarks: AWA1, AWA2, CUB, FLO and SUN from 
[Xian et al. (CVPR 2017)](https://openaccess.thecvf.com/content_cvpr_2017/papers/Xian_Zero-Shot_Learning_-_CVPR_2017_paper.pdf).  Download the datasets and take them into dir data.

### Running

```python
python train.py # CUB
```
## Citation:
@article{zhang2023visual,
  title={Visual-semantic consistency matching network for generalized zero-shot learning},
  author={Zhang, Zhenqi and Cao, Wenming},
  journal={Neurocomputing},
  volume={536},
  pages={30--39},
  year={2023},
  publisher={Elsevier}
}

## Ackowledgement

We thank the following repos providing helpful components in our work.
1. [TF-VAEGAN](https://github.com/akshitac8/tfvaegan)
2. [SDGZSL](https://github.com/uqzhichen/SDGZSL)
