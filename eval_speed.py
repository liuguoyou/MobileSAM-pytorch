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

    # paths
    parser.add_argument('--work_dir', type=str, default="./work_dir", help='work dir')
    parser.add_argument('--log', type=str, default=None)

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_option()

    if args.log is not None:
        if not os.path.exists(args.work_dir):
            os.makedirs(args.work_dir)
    
        with open(os.path.join(args.work_dir, args.log), "a") as f:
            f.write("model inference device is [{}].".format(args.device))

    print("model inference device is [{}].".format(args.device))

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

    total_sam_time = 0
    total_mobilesam_time = 0

    total_sam_encoder_time = 0
    total_mobilesam_encoder_time = 0
    
    total_sam_pred_time = 0
    total_mobilesam_pred_time = 0

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

        total_sam_time += sam_time
        total_mobilesam_time += mobile_sam_time

        if args.log is not None:
            with open(os.path.join(args.work_dir, args.log), "a") as f:
                f.write("idx {}: \tsam inference time: {}\tmobile sam inference time: {}\n".format(i + 1 - args.data_idx_offset, sam_time, mobile_sam_time))

        print("idx {}: \tsam inference time: {}\tmobile sam inference time: {}".format(i + 1 - args.data_idx_offset, sam_time, mobile_sam_time))


        # eval encoder inference

        sam_encoder_time = sam_predictor.set_image(test_img)
        mobilesam_encoder_time = mobile_sam_predictor.set_image(test_img)
        
        total_sam_encoder_time += sam_encoder_time
        total_mobilesam_encoder_time += mobilesam_encoder_time

        h, w, c = test_img.shape
        input_point = np.array([[h // 2, h //2]])
        input_label = np.array([1])

        start_time = time.time()
        sam_masks, _, _ = sam_predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        multimask_output=True,
                    )
        sam_pred_time = time.time() - start_time

        start_time = time.time()
        mobile_sam_masks, _, _ = mobile_sam_predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        multimask_output=True,
                    )
        mobilesam_pred_time = time.time() - start_time

        total_sam_pred_time += sam_pred_time
        total_mobilesam_pred_time += mobilesam_pred_time

        if args.log is not None:
            with open(os.path.join(args.work_dir, args.log), "a") as f:
                f.write("idx {}: \tsam encoder time: {}\tmobile sam encoder time: {}\n".format(i + 1 - args.data_idx_offset, sam_encoder_time, mobilesam_encoder_time))
                f.write("idx {}: \tsam pred time: {}\tmobile sam pred time: {}\n".format(i + 1 - args.data_idx_offset, sam_pred_time, mobilesam_pred_time))

        print("idx {}: \tsam encoder time: {}\tmobile sam encoder time: {}".format(i + 1 - args.data_idx_offset, sam_encoder_time, mobilesam_encoder_time))
        print("idx {}: \tsam pred time: {}\tmobile sam pred time: {}".format(i + 1 - args.data_idx_offset, sam_pred_time, mobilesam_pred_time))

    
    if args.log is not None:
        with open(os.path.join(args.work_dir, args.log), "a") as f:
            f.write("===== summary =====\n")
            f.write("average sam inference time: {}\t mobile sam inference time: {}\n".format(total_sam_time / args.eval_num, total_mobilesam_time / args.eval_num))
            f.write("average sam encoder time: {}\t mobile sam encoder time: {}\n".format(total_sam_encoder_time / args.eval_num, total_mobilesam_encoder_time / args.eval_num))
            f.write("average sam pred time: {}\t mobile sam pred time: {}\n".format(total_sam_pred_time / args.eval_num, total_mobilesam_pred_time / args.eval_num))


    print("===== summary =====")
    print("average sam inference time: {}\t mobile sam inference time: {}".format(total_sam_time / args.eval_num, total_mobilesam_time / args.eval_num))
    print("average sam encoder time: {}\t mobile sam encoder time: {}".format(total_sam_encoder_time / args.eval_num, total_mobilesam_encoder_time / args.eval_num))
    print("average sam pred time: {}\t mobile sam pred time: {}".format(total_sam_pred_time / args.eval_num, total_mobilesam_pred_time / args.eval_num))

        

        


