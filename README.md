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

Download the MobileSAM encoder we re-trained from [here](https://drive.google.com/drive/folders/1kzdY2GuJM3B8ssZWOBJhqWXx-QaEPT9e?usp=drive_link) 

Download SA-1B dataset parts from [here](https://segment-anything.com/dataset/index.html) and unzip them, and then pre-process the dataset using:

```

python preprocess.py --dataset_dir <dataset_dir>
```
e.g., after downloading ```sa_000000.tar```, we unzipped the file into the file folder ```sa_000000```, we can run ```python preprocess.py --dataset_dir sa_000000``` to pre-process the data to generate features' ```.npy``` file

Distill the knowledge from SAM:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train.py --optim <optimizer> --learning_rate <lr> --weight_decay <wd> --work_dir <work_dir>
```

e.g., ```CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train.py --optim adamw --learning_rate 0.001 --weight_decay 0.01 --batch_size 16 --epochs 8 --work_dir exp_4card/adamw_lr_1e-3_wd_1e-2_bs_16_epoch_8_v100"```


Evaluate the trained model through segmenting everything and visualize the results:

```
python eval_visual.py --ckpt <checkpoint_path> --save_dir <save_dir>
```

e.g., ```python eval_visual.py --ckpt "exp/adamw_lr_1e-3_v100/ckpt/final.pth" --save_dir vis```

Evaluate the trained model through point prompts and output mIoU:

```
python eval_miou.py --ckpt <checkpoint_path>
```

e.g., ```python eval_miou.py --ckpt "exp/adamw_lr_1e-3_v100/ckpt/final.pth" --point_num_h 5 --point_num_w 5```
