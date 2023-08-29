import os
import cv2
import numpy as np
# import matplotlib.pyplot as plt
import argparse
import time
import json
import torch
from skimage.metrics import structural_similarity

from dataset import transform
from train import customized_mseloss
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def parse_option():
    parser = argparse.ArgumentParser('argument for evaluation')

    parser.add_argument('--device', type=str, default='cuda', help='device')

    # eval dataset settings
    parser.add_argument('--dataset_path', type=str, default="/dataset/sharedir/research/vyueyu/sa-1b/sa_000080", help='root path of dataset')
    parser.add_argument('--eval_num', type=int, default=20)
    parser.add_argument('--data_idx_offset', type=int, default=895052)

    # our mobile sam model
    parser.add_argument('--mobile_sam_type', type=str, default="vit_t")
    parser.add_argument('--mobile_sam_ckpt', type=str, default="/dataset/vyueyu/project/MobileSAM/weights/mobile_sam.pt")
    parser.add_argument('--ckpt', type=str, default=None, help="mobile sam encoder ckpt")

    # sam model 
    parser.add_argument('--sam_type', type=str, default="vit_h")
    parser.add_argument('--sam_ckpt', type=str, default="/dataset/vyueyu/project/MobileSAM/sam_vit_h_4b8939.pth")

    parser.add_argument('--threshold', type=float, default=0)

    # paths
    parser.add_argument('--work_dir', type=str, default="./work_dir", help='work dir')
    parser.add_argument('--log', type=str, default=None)

    args = parser.parse_args()
    return args

def eval_miou(pred_masks, target_masks):
    assert len(pred_masks.shape) == 2 or len(pred_masks.shape) == 3
    if len(pred_masks.shape) == 2:
        return (pred_masks & target_masks).sum() / ((pred_masks | target_masks).sum() + 1e-10)
    return [(pred_mask & target_mask).sum() / ((pred_mask | target_mask).sum() + 1e-10) for pred_mask, target_mask in zip(pred_masks, target_masks)]
    

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def calculate_average_precision(mobile_sam_masks, sam_masks, threshold=0):
    sam_masks = sorted(sam_masks, key=lambda mask: mask['stability_score'])
    mobile_sam_masks = sorted(mobile_sam_masks, key=lambda mask: mask['stability_score'])

    precision_list = []
    recall_list = []
    matched_num = 0

    miou = 0

    flag = [False] * len(sam_masks)

    for i in range(len(mobile_sam_masks)):
        max_iou = -1
        matched_idx = -1
        for j in range(len(sam_masks)):
            if not flag[j]:
                iou = eval_miou(mobile_sam_masks[i]['segmentation'], sam_masks[j]['segmentation'])
                if iou >= threshold and iou > max_iou:
                    max_iou = iou
                    matched_idx = j
        if matched_idx != -1:
            flag[matched_idx] = True
            miou += max_iou
            matched_num += 1
        
        precision_list.append(matched_num / (i + 1))
        recall_list.append(matched_num / len(sam_masks))

    avg_precision = 0.0
    max_precision = 0.0

    for i in range(len(recall_list)):
        max_precision = max(max_precision, precision_list[i])
        avg_precision += max_precision * (recall_list[i] - recall_list[i - 1] if i > 0 else recall_list[i])

    return avg_precision, miou / matched_num

def calculate_gray_scale_ssim(mobile_sam_masks, sam_masks):
    mobile_sam_gray_scale = np.stack([x['segmentation'] for x in mobile_sam_masks], axis=0).sum(0)
    sam_gray_scale = np.stack([x['segmentation'] for x in sam_masks], axis=0).sum(0)
    mobile_sam_gray_scale = 255 * (mobile_sam_gray_scale - np.min(mobile_sam_gray_scale)) / (np.max(mobile_sam_gray_scale) - np.min(mobile_sam_gray_scale))
    sam_gray_scale = 255 * (sam_gray_scale - np.min(sam_gray_scale)) / (np.max(sam_gray_scale) - np.min(sam_gray_scale))
    
    return structural_similarity(mobile_sam_gray_scale.astype(int), sam_gray_scale.astype(int), data_range=255)


if __name__ == "__main__":

    args = parse_option()

    if args.log is not None:
        if not os.path.exists(args.work_dir):
            os.makedirs(args.work_dir)

    # original sam model
    sam = sam_model_registry[args.sam_type](checkpoint=args.sam_ckpt)
    sam.to(device=args.device)
    sam.eval()
    # sam_predictor = SamPredictor(sam)

    # our retrained mobile sam 
    mobile_sam = sam_model_registry[args.mobile_sam_type](checkpoint=args.mobile_sam_ckpt)
    if args.ckpt is not None:
        mobile_sam.image_encoder.load_state_dict(torch.load(args.ckpt))
    mobile_sam.to(device=args.device)
    mobile_sam.eval()

    # mask generator
    sam_mask_generator = SamAutomaticMaskGenerator(sam)
    mobile_sam_mask_generator = SamAutomaticMaskGenerator(mobile_sam)
    
    # predictor
    sam_predictor = SamPredictor(sam)
    mobile_sam_predictor = SamPredictor(mobile_sam)

    # -----start evaluation----- #

    mAP, mIoU, mSSIM = 0, 0, 0

    for i in range(args.data_idx_offset, args.data_idx_offset + args.eval_num):
        test_img_path = os.path.join(args.dataset_path, "sa_" + str(i) + ".jpg")
        test_img = cv2.imread(test_img_path)
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

        # generate masks for sam
        start_time = time.time()
        sam_masks = sam_mask_generator.generate(test_img)
        sam_time = time.time() - start_time
        # generate masks for mobilesam
        start_time = time.time()
        mobile_sam_masks = mobile_sam_mask_generator.generate(test_img)
        mobile_sam_time = time.time() - start_time

        ap, iou = calculate_average_precision(mobile_sam_masks, sam_masks, args.threshold)
        ssim = calculate_gray_scale_ssim(mobile_sam_masks, sam_masks)
        mAP += ap
        mIoU += iou
        mSSIM += ssim

        if args.log is not None:
            with open(os.path.join(args.work_dir, args.log), "a") as f:
                f.write("idx {}: \tAP: {}\tIoU: {}\tSSIM: {}\tmAP: {}\tmIoU: {}\tmSSIM: {}\n".format(i + 1 - args.data_idx_offset, ap, iou, ssim, mAP/(i+1-args.data_idx_offset), mIoU/(i+1-args.data_idx_offset), mSSIM/(i+1-args.data_idx_offset)))

        print("idx {}: \tAP: {}\tIoU: {}\tSSIM: {}\tmAP: {}\tmIoU: {}\tmSSIM: {}".format(i + 1 - args.data_idx_offset, ap, iou, ssim, mAP/(i+1-args.data_idx_offset), mIoU/(i+1-args.data_idx_offset), mSSIM/(i+1-args.data_idx_offset)))
    
    mAP /= args.eval_num
    mIoU /= args.eval_num
    mSSIM /= args.eval_num

    if args.log is not None:
        with open(os.path.join(args.work_dir, args.log), "a") as f:
            f.write("=== summary ===\n")
            f.write("--- test image index from {} to {} ---\n".format(args.data_idx_offset, args.data_idx_offset + args.eval_num))
            f.write("--- mAP: {}\tmIoU: {}\t mSSIM: {} ---\n".format(mAP, mIoU, mSSIM))
    
    print("=== summary ===")
    print("--- test image index from {} to {} ---".format(args.data_idx_offset, args.data_idx_offset + args.eval_num))
    print("--- mAP: {}\tmIoU: {}\t mSSIM: {} ---".format(mAP, mIoU, mSSIM))

        


