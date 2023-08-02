# MobileSAM-distiller

Reproduction of MobileSAM using pytorch

Download the MobileSAM encoder we re-trained from [here](https://drive.google.com/drive/folders/1kzdY2GuJM3B8ssZWOBJhqWXx-QaEPT9e?usp=drive_link) for your usage.

## Installation

### Clone the Repository:

```
git clone --recursive https://github.com/YuYue525/MobileSAM-distiller.git
cd MobileSAM-distiller
mv *.py ./MobileSAM
cd MobileSAM; pip install -e .
```

### environment preparation

Please use ```conda``` to create an environment and download all the packages required (we use ```python==3.8.13```, ```ptorch==1.13.0+cu117```, ```torchvision==0.14.0+cu117```):

```
pip install -r requirements.txt 
```

## Dataset Preparation

Download SA-1B dataset parts from [here](https://segment-anything.com/dataset/index.html) and unzip them, and then pre-process the dataset using:

```
# dataset downloading in dataset dir
wget -b -c -O "sa_<index>.tar" "<link>"

# unzip the downloaded file
mkdir sa_<index>
tar -xvf sa_<index>.tar -C sa_<index>

# data preprocess: extract features by SAM teacher as "target" and save them as .npy
python preprocess.py --dataset_dir sa_<index>
```

e.g., after downloading ```sa_000000.tar```, we unzipped the file into the file folder ```sa_000000```, we can run ```python preprocess.py --dataset_dir sa_000000``` to pre-process the data to generate features' ```.npy``` file. In our experiments, we downloaded 2% SA-1B dataset as our training set (from ```sa_000000``` to ```sa_000019```) and another 0.1% as our validation set (```sa_000020```).


## Distillation Process

Please carefully check all the file paths in the code!

We can distill the knowledge from SAM to our MobileSAM using the following command:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train.py --optim <optimizer> --learning_rate <lr> --weight_decay <wd> --work_dir <work_dir>
```

e.g., ```CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train.py --optim adamw --learning_rate 0.001 --weight_decay 0.0005 --batch_size 8 --epochs 16 --work_dir exp/adamw_lr_1e-3_wd_5e-4_bs_8_epoch_16"```

## Model Aggregation

Noticed that our trained model is only the <b>image encoder</b> part of the whole MobileSAM model. To aggregate the trained encoder with freezed prompt encoder and mask decoder, please run:

```
python model_aggregation.py --ckpt <pth_path> --save_model_path <save_path> --save_model_name <model_name>
```
e.g., ```python model_aggregation.py --ckpt exp/adamw_lr_1e-3_wd_5e-4_bs_8_epoch_16/ckpt/final.pth --save_model_path weights --save_model_name adamw_lr_1e-3_wd_5e-4_bs_8_epoch_16.pth```

## Evaluation

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


## ONNX Export

Export the model with

```
python scripts/export_onnx_model.py --checkpoint ./weights/mobile_sam.pt --model-type vit_t --output ./mobile_sam.onnx
```

Also check the [example notebook](https://github.com/ChaoningZhang/MobileSAM/blob/master/notebooks/onnx_model_example.ipynb) to follow detailed steps.


## Acknowledgement

<details>
<summary>
<a href="https://github.com/ChaoningZhang/MobileSAM">MobileSAM</a> (Faster Segment Anything) [<b>bib</b>]
</summary>

```bibtex
@article{mobile_sam,
  title={Faster Segment Anything: Towards Lightweight SAM for Mobile Applications},
  author={Zhang, Chaoning and Han, Dongshen and Qiao, Yu and Kim, Jung Uk and Bae, Sung-Ho and Lee, Seungkyu and Hong, Choong Seon},
  journal={arXiv preprint arXiv:2306.14289},
  year={2023}
}
```
</details>

<details>
<summary>
<a href="https://github.com/facebookresearch/segment-anything">SAM</a> (Segment Anything) [<b>bib</b>]
</summary>

```bibtex
@article{kirillov2023segany,
  title={Segment Anything}, 
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```
</details>

<details>
<summary>
<a href="https://github.com/microsoft/Cream/tree/main/TinyViT">TinyViT</a> (TinyViT: Fast Pretraining Distillation for Small Vision Transformers) [<b>bib</b>]
</summary>

```bibtex
@InProceedings{tiny_vit,
  title={TinyViT: Fast Pretraining Distillation for Small Vision Transformers},
  author={Wu, Kan and Zhang, Jinnian and Peng, Houwen and Liu, Mengchen and Xiao, Bin and Fu, Jianlong and Yuan, Lu},
  booktitle={European conference on computer vision (ECCV)},
  year={2022}
```
</details>
