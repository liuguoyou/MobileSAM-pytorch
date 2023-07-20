import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F

from dataset import transform
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from mobile_sam.modeling import TinyViT
from mobile_sam.utils.transforms import ResizeLongestSide

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def eval_miou(pred_masks, target_masks):
    assert len(pred_masks.shape) == 2 or len(pred_masks.shape) == 3
    if len(pred_masks.shape) == 2:
        return (pred_masks & target_masks).sum() / ((pred_masks | target_masks).sum() + 1e-10)
    return [(pred_mask & target_mask).sum() / ((pred_mask | target_mask).sum() + 1e-10) for pred_mask, target_mask in zip(pred_masks, target_masks)]

def parse_option():
    parser = argparse.ArgumentParser('argument for evaluation')

    parser.add_argument('--dataset_path', type=str, default="/dataset/vyueyu/sa-1b/sa_000010", help='root path of dataset')

    parser.add_argument('--device', type=str, default='cuda', help='device')

    parser.add_argument('--point_num_h', type=int, default=5)
    parser.add_argument('--point_num_w', type=int, default=5)
    parser.add_argument('--eval_num', type=int, default=200)
    parser.add_argument('--data_idx_offset', type=int, default=111877)

    parser.add_argument('--ckpt', type=str, default="/dataset/vyueyu/project/MobileSAM/exp/adamw_lr_1e-3_v100/ckpt/final.pth")

    parser.add_argument('--sam_type', type=str, default="vit_h")
    parser.add_argument('--sam_ckpt', type=str, default="/dataset/vyueyu/project/MobileSAM/sam_vit_h_4b8939.pth")

    parser.add_argument('--mobile_sam_type', type=str, default="vit_t")
    parser.add_argument('--mobile_sam_ckpt', type=str, default="/dataset/vyueyu/project/MobileSAM/weights/mobile_sam.pt")


    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_option()

    test_img_dir = args.dataset_path
    retrained_mobile_sam_encoder_checkpoint = args.ckpt

    # models
    mobile_sam_type = args.mobile_sam_type
    mobile_sam_checkpoint = args.mobile_sam_ckpt

    sam_type = args.sam_type
    sam_checkpoint = args.sam_ckpt

    device = args.device

    # original sam model
    sam = sam_model_registry[sam_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam.eval()
    sam_predictor = SamPredictor(sam)

    # mobile sam provided
    mobile_sam = sam_model_registry[mobile_sam_type](checkpoint=mobile_sam_checkpoint)
    mobile_sam.to(device=device)
    mobile_sam.eval()
    mobile_sam_predictor = SamPredictor(mobile_sam)

    # our retrained mobile sam 
    mobile_sam_retrained = sam_model_registry[mobile_sam_type](checkpoint=mobile_sam_checkpoint)
    mobile_sam_retrained.image_encoder.load_state_dict(torch.load(retrained_mobile_sam_encoder_checkpoint))
    mobile_sam_retrained.to(device=device)
    mobile_sam_retrained.eval()
    mobile_sam_retrained_predictor = SamPredictor(mobile_sam_retrained)

    mobile_sam_iou = []
    mobile_sam_retrained_iou = []

    data_idx_offset = args.data_idx_offset
    eval_num = args.eval_num

    for i in tqdm(range(data_idx_offset, data_idx_offset + eval_num)):
        test_img_path = os.path.join(test_img_dir, "sa_" + str(i) + ".jpg")
        test_img = cv2.imread(test_img_path)
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        # print(test_img.shape)

        sam_predictor.set_image(test_img)
        mobile_sam_predictor.set_image(test_img)
        mobile_sam_retrained_predictor.set_image(test_img)

        h, w, c = test_img.shape
        point_num_h, point_num_w = args.point_num_h, args.point_num_w
        margin_h, margin_w = h // point_num_h, w // point_num_w
        start_point_pos = (margin_h // 2, margin_w // 2)
        
        input_label = np.array([1])
        for point_h in range(point_num_h):
            for point_w in range(point_num_w):
                input_point = np.array([[start_point_pos[1] + point_w * margin_w, start_point_pos[0] + point_h * margin_h]])

                sam_masks, _, _ = sam_predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True,
                )

                mobile_sam_predictor.set_image(test_img)
                mobile_sam_masks, _, _ = mobile_sam_predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True,
                )

                mobile_sam_retrained_predictor.set_image(test_img)
                mobile_sam_retrained_masks, _, _ = mobile_sam_retrained_predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True,
                )

                mobile_sam_iou += eval_miou(sam_masks, mobile_sam_masks)
                mobile_sam_retrained_iou += eval_miou(sam_masks, mobile_sam_retrained_masks)

    print("MobileSAM mIoU: {:.3f}, our MobileSAM mIoU: {:.3f}".format(np.array(mobile_sam_iou).mean() * 100, np.array(mobile_sam_retrained_iou).mean() * 100))