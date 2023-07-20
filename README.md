# MobileSAM-distiller
Reproduction of MobileSAM using pytorch

## Installation

The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

Clone the Repository:

```
git clone --recursive https://github.com/YuYue525/MobileSAM-distiller.git
cd MobileSAM-distiller
mv *.py ./MobileSAM
cd MobileSAM; pip install -e .
```

## Getting Started

Please carefully check all the file paths in the code!
Download the retrained MobileSAM encoder from [here](https://drive.google.com/drive/folders/1kzdY2GuJM3B8ssZWOBJhqWXx-QaEPT9e?usp=drive_link) 

Download SA-1B dataset parts from [here](https://segment-anything.com/dataset/index.html) and unzip them, and then pre-process the dataset using:

```
python preprocess.py --dataset_dir <dataset_dir>
```
for example, after downloading sa_000000.tar, we unzipped the file into the file folder 'sa_000000', we can run ```python preprocess.py --dataset_dir sa_000000``` to pre-process the data to generate features' .npy file

Distill the knowledge from SAM:

```
python train.py --optim <optimizer> --learning_rate <lr> --weight_decay <wd> --work_dir <work_dir>
```

for example, ```python train.py --optim adamw --learning_rate 1e-3 --weight_decay 5e-4 --work_dir exp/adamw_lr_1e-3_wd_5e-4'```


Evaluate the trained model through segmenting everything and visualize the results:

```
python eval_visual.py --ckpt <checkpoint_path> --save_dir <save_dir>
```

for example, ```python eval_visual.py --ckpt exp/adamw_lr_1e-3_v100/ckpt/final.pth --save_dir vis```

Evaluate the trained model through point prompts and output mIoU:

```
python eval_miou.py --ckpt <checkpoint_path>
```

for example, ```python eval_miou.py --ckpt exp/adamw_lr_1e-3_v100/ckpt/final.pth --point_num_h 5 --point_num_w 5```
