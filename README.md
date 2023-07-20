# MobileSAM-distiller
Reproduction of MobileSAM using pytorch

## Installation

Clone the Repository:

```
git clone https://github.com/ChaoningZhang/MobileSAM.git
cd MobileSAM
pip install -e .
```

Download SA-1B dataset parts from [here](https://segment-anything.com/dataset/index.html) and unzip them, and then pre-process the dataset using:

```
python preprocess.py --dataset_dir <dataset_name>

# for example, after downloading sa_000000.tar, we unzipped the file into the file folder 'sa_000000',
# we can run 'python preprocess.py --dataset_dir sa_000000' to pre-process the data to generate features' .npy file
```

Distill the knowledge from SAM:

```
python train.py --optim <optimizer> --learning_rate <lr> --weight_decay <wd> --work_dir <work_dir>

# for example, 'python train.py --optim adamw --learning_rate 1e-3 --weight_decay 5e-4 --work_dir exp/adamw_lr_1e-3_wd_5e-4'
```

Evaluate the trained model through segmenting everything and visualize the results:

```
python eval_visual.py --ckpt <checkpoint_path> --save_dir <save_dir>

# for example, 'python eval_visual.py --ckpt exp/adamw_lr_1e-3_v100/ckpt/final.pth --save_dir vis'
```

Evaluate the trained model through point prompts and output mIoU:

```
python eval_miou.py --ckpt <checkpoint_path>

# for example, 'python eval_miou.py --ckpt exp/adamw_lr_1e-3_v100/ckpt/final.pth --point_num_h 5 --point_num_w 5'
```
