import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import json
import torch

from dataset import transform
from train import customized_mseloss
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def parse_option():
    parser = argparse.ArgumentParser('argument for evaluation')

    parser.add_argument('--device', type=str, default='cuda', help='device')

    # eval dataset settings
    parser.add_argument('--dataset_path', type=str, default="/dataset/vyueyu/sa-1b/sa_000020", help='root path of dataset')
    parser.add_argument('--eval_num', type=int, default=20)
    parser.add_argument('--data_idx_offset', type=int, default=223750)

    # our mobile sam model
    parser.add_argument('--ckpt', type=str, default="/dataset/vyueyu/project/MobileSAM/exp_aug/adamw_lr_1e-3_wd_5e-4_v100_aug/ckpt/final.pth")

    # the given mobile sam model
    parser.add_argument('--mobile_sam_type', type=str, default="vit_t")
    parser.add_argument('--mobile_sam_ckpt', type=str, default="/dataset/vyueyu/project/MobileSAM/weights/mobile_sam.pt")

    # sam model 
    parser.add_argument('--sam_type', type=str, default="vit_h")
    parser.add_argument('--sam_ckpt', type=str, default="/dataset/vyueyu/project/MobileSAM/sam_vit_h_4b8939.pth")

    # visualization
    parser.add_argument('--vis', type=bool, default=True, help='whether to visualize the segment results')
    parser.add_argument('--vis_dir', type=str, default="vis", help='root path of dataset')
    # miou
    parser.add_argument('--miou', type=bool, default=True, help='whether to output the miou')
    parser.add_argument('--point_num_h', type=int, default=5)
    parser.add_argument('--point_num_w', type=int, default=5)

    # paths
    parser.add_argument('--work_dir', type=str, default="./work_dir", help='work dir')

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

if __name__ == "__main__":

    args = parse_option()

    if not os.path.exists(os.path.join(args.work_dir, args.vis_dir)) and args.vis:
        os.makedirs(os.path.join(args.work_dir, args.vis_dir))

    # original sam model
    sam = sam_model_registry[args.sam_type](checkpoint=args.sam_ckpt)
    sam.to(device=args.device)
    sam.eval()
    # sam_predictor = SamPredictor(sam)

    # mobile sam provided
    mobile_sam = sam_model_registry[args.mobile_sam_type](checkpoint=args.mobile_sam_ckpt)
    mobile_sam.to(device=args.device)
    mobile_sam.eval()

    # our retrained mobile sam 
    mobile_sam_retrained = sam_model_registry[args.mobile_sam_type](checkpoint=args.mobile_sam_ckpt)
    mobile_sam_retrained.image_encoder.load_state_dict(torch.load(args.ckpt))
    mobile_sam_retrained.to(device=args.device)
    mobile_sam_retrained.eval()

    if args.vis:
        sam_mask_generator = SamAutomaticMaskGenerator(sam)
        mobile_sam_mask_generator = SamAutomaticMaskGenerator(mobile_sam)
        mobile_sam_retrained_mask_generator = SamAutomaticMaskGenerator(mobile_sam_retrained)
    
    if args.miou:
        sam_predictor = SamPredictor(sam)
        mobile_sam_predictor = SamPredictor(mobile_sam)
        mobile_sam_retrained_predictor = SamPredictor(mobile_sam_retrained)

    # -----start evaluation----- #

    mobile_sam_iou = []
    mobile_sam_retrained_iou = []

    for i in range(args.data_idx_offset, args.data_idx_offset + args.eval_num):
        test_img_path = os.path.join(args.dataset_path, "sa_" + str(i) + ".jpg")
        test_img = cv2.imread(test_img_path)
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

        if args.vis:
            # generate masks for sam
            start_time = time.time()
            sam_masks = sam_mask_generator.generate(test_img)
            sam_time = time.time() - start_time
            # generate masks for mobilesam
            start_time = time.time()
            mobile_sam_masks = mobile_sam_mask_generator.generate(test_img)
            mobile_sam_time = time.time() - start_time
            # generate masks for our mobilesam
            start_time = time.time()
            mobile_sam_retrained_masks = mobile_sam_retrained_mask_generator.generate(test_img)
            mobile_sam_retrained_time = time.time() - start_time

            # save vis results
            plt.figure(figsize=(60,20))

            plt.subplot(1,3,1)
            plt.imshow(test_img)
            show_anns(sam_masks)
            plt.axis('off')
            plt.title("SAM")

            plt.subplot(1,3,2)
            plt.imshow(test_img)
            show_anns(mobile_sam_masks)
            plt.axis('off')
            plt.title("MobileSAM (given)")

            plt.subplot(1,3,3)
            plt.imshow(test_img)
            show_anns(mobile_sam_retrained_masks)
            plt.axis('off')
            plt.title("Our MobileSAM (re-trained)")

            plt.savefig(os.path.join(args.work_dir, args.vis_dir, str(i) + ".jpg"))
            plt.clf()
            plt.close()

            tensor_input = transform(test_img)[None, :, :, :].to(args.device)
            # image encoder time and loss
            start_time = time.time()
            pred = mobile_sam.image_encoder(tensor_input)
            mobile_sam_encoder_retrained_time = time.time() - start_time

            start_time = time.time()
            our_pred = mobile_sam_retrained.image_encoder(tensor_input)
            mobile_sam_encoder_time = time.time() - start_time

            target = torch.from_numpy(np.load(os.path.join(args.dataset_path, "sa_" + str(i) + ".npy"))).to(args.device)

            print("Image Index {}:".format(i))
            print("MSE loss: \t\t\t MobileSAM {:.3f} \t Our MobileSAM {:.3f}".format(customized_mseloss(pred, target).item(), customized_mseloss(our_pred, target).item()))
            print("Encoder inference time: \t MobileSAM {:.3f}s \t Our MobileSAM {:.3f}s".format(mobile_sam_encoder_time, mobile_sam_encoder_retrained_time))
            print("Model inference time: \t\t MobileSAM {:.3f}s \t Our MobileSAM {:.3f}s \t SAM {:.3f}s \n".format(mobile_sam_time, mobile_sam_retrained_time, sam_time))

        if args.miou:
            sam_predictor.set_image(test_img)
            mobile_sam_predictor.set_image(test_img)
            mobile_sam_retrained_predictor.set_image(test_img)

            h, w, c = test_img.shape
            point_num_h, point_num_w = args.point_num_h, args.point_num_w
            margin_h, margin_w = h // point_num_h, w // point_num_w
            start_point_pos = (margin_w // 2, margin_h // 2)
            
            input_label = np.array([1])
            for point_h in range(point_num_h):
                for point_w in range(point_num_w):
                    input_point = np.array([[start_point_pos[0] + point_w * margin_w, start_point_pos[1] + point_h * margin_h]])
                    sam_masks, _, _ = sam_predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        multimask_output=True,
                    )

                    mobile_sam_masks, _, _ = mobile_sam_predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        multimask_output=True,
                    )

                    mobile_sam_retrained_masks, _, _ = mobile_sam_retrained_predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        multimask_output=True,
                    )

                    mobile_sam_iou += eval_miou(sam_masks, mobile_sam_masks)
                    mobile_sam_retrained_iou += eval_miou(sam_masks, mobile_sam_retrained_masks)

    if args.miou:
        print("MobileSAM mIoU: {:.3f}, our MobileSAM mIoU: {:.3f}".format(np.array(mobile_sam_iou).mean() * 100, np.array(mobile_sam_retrained_iou).mean() * 100))
    










