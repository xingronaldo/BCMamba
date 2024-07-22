# Code for the mamuscript 'Change Detection Mamba with Boundary Constraining'.
---------------------------------------------
Here I provide the PyTorch implementation for BCMamba.


## ENVIRONMENT
>RTX 3090<br>
>cuda 1.8.0<br>
>PyTorch 2.0.0<br>
>python 3.8<br>
>mmcv 1.6.0<br>
>causal_conv1d 1.1.0<br>
>[mamba-ssm 1p1p1](https://github.com/hustvl/Vim).

## Installation
Clone this repo:

```shell
git clone https://github.com/xingronaldo/BCMamba.git
cd BCMamba
```

* Install dependencies

```shell
sh req.sh
```
All other dependencies can be installed via 'pip'.

## Dataset Preparation
Download data and add them to `./datasets`. 


## Test
Here I provide the trained models for the SV-CD dataset [Baidu Netdisk, code: BCMa](https://pan.baidu.com/s/1VSQRRX4FVwpdUOEHgEbphw)mba.

Put them in `./checkpoints`.

* Test on the SV-CD dataset with the MobileNetV2 backbone

```python
python test.py --name SV-CD--gpu_ids 1
```

## Train & Validation
```python
python trainval.py --gpu_ids 1 --name --name SV-CD
```
All the hyperparameters can be adjusted in `option.py`.


## Contact
Email: guangxingwang@mail.nwpu.edu.cn
