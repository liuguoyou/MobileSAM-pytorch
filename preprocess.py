import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

import torch
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def parse_option():
    parser = argparse.ArgumentParser('argument for pre-processing')

    parser.add_argument('--dataset_path', type=str, default="/dataset/vyueyu/sa-1b", help='root path of dataset')
    parser.add_argument('--dataset_dir', type=str, required=True, help='dir of dataset')

    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--sam_type', type=str, default="vit_h")
    parser.add_argument('--sam_ckpt', type=str, default="/dataset/vyueyu/project/MobileSAM/sam_vit_h_4b8939.pth")

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_option()

    device = args.device

    sam = sam_model_registry[args.sam_type](checkpoint=args.sam_ckpt)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    test_image_dir = os.path.join(args.dataset_path, args.dataset_dir)
    test_image_paths = [os.path.join(test_image_dir, img_name) for img_name in os.listdir(test_image_dir)]
    
    n = len(test_image_paths)
    for i, test_image_path in enumerate(tqdm(test_image_paths)):
        print(i, "/", n)
        if ".jpg" in test_image_path:
            test_image = cv2.imread(test_image_path)
            test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

            predictor.set_image(test_image)
            feature = predictor.features
            np.save(test_image_path.replace(".jpg", ".npy"), feature.cpu().numpy())# .astype(np.float16))
