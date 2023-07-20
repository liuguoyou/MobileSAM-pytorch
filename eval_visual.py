import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import functional as F

from dataset import transform
from train import get_loss_fn
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from mobile_sam.modeling import TinyViT
from mobile_sam.utils.transforms import ResizeLongestSide

def parse_option():
    parser = argparse.ArgumentParser('argument for evaluation')

    parser.add_argument('--dataset_path', type=str, default="/dataset/vyueyu/sa-1b/sa_000010", help='root path of dataset')

    parser.add_argument('--device', type=str, default='cuda', help='device')

    parser.add_argument('--eval_num', type=int, default=20)
    parser.add_argument('--data_idx_offset', type=int, default=111877)

    parser.add_argument('--ckpt', type=str, default="/dataset/vyueyu/project/MobileSAM/exp/adamw_lr_1e-3_v100/ckpt/final.pth")

    parser.add_argument('--mobile_sam_type', type=str, default="vit_t")
    parser.add_argument('--mobile_sam_ckpt', type=str, default="/dataset/vyueyu/project/MobileSAM/weights/mobile_sam.pt")

    parser.add_argument('--save_dir', type=str, default="/dataset/vyueyu/project/MobileSAM/vis", help='root path of dataset')

    args = parser.parse_args()
    return args

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

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    test_img_dir = args.dataset_path
    retrained_mobile_sam_encoder_checkpoint = args.ckpt

    # models
    mobile_sam_type = args.mobile_sam_type
    mobile_sam_checkpoint = args.mobile_sam_ckpt

    device = args.device

    # mobile sam provided
    mobile_sam = sam_model_registry[mobile_sam_type](checkpoint=mobile_sam_checkpoint)
    mobile_sam.to(device=device)
    mobile_sam.eval()
    mobile_sam_mask_generator = SamAutomaticMaskGenerator(mobile_sam)

    # our retrained mobile sam 
    mobile_sam_retrained = sam_model_registry[mobile_sam_type](checkpoint=mobile_sam_checkpoint)
    mobile_sam_retrained.image_encoder.load_state_dict(torch.load(retrained_mobile_sam_encoder_checkpoint))
    mobile_sam_retrained.to(device=device)
    mobile_sam_retrained.eval()
    mobile_sam_retrained_mask_generator = SamAutomaticMaskGenerator(mobile_sam_retrained)

    plt.figure(figsize=(20,20))
    data_idx_offset = args.data_idx_offset
    eval_num = args.eval_num

    for i in tqdm(range(data_idx_offset, data_idx_offset + eval_num)):
        test_img_path = os.path.join(test_img_dir, "sa_" + str(i) + ".jpg")
        test_img = cv2.imread(test_img_path)
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

        mobile_sam_masks = mobile_sam_mask_generator.generate(test_img)
        mobile_sam_retrained_masks = mobile_sam_retrained_mask_generator.generate(test_img)
        # print(our_masks)

        plt.imshow(test_img)
        show_anns(mobile_sam_masks)
        plt.axis('off')
        plt.savefig(os.path.join(args.save_dir, "mobile_sam_" + str(i) + ".jpg"))
        plt.clf()

        plt.imshow(test_img)
        show_anns(mobile_sam_retrained_masks)
        plt.axis('off')
        plt.savefig(os.path.join(args.save_dir, "mobile_sam_retrained_" + str(i) + ".jpg"))
        plt.clf()


        input = transform(test_img)[None, :, :, :].to(device)
        pred = mobile_sam.image_encoder(input)
        our_pred = mobile_sam_retrained.image_encoder(input)
        target = torch.from_numpy(np.load(os.path.join(test_img_dir, "sa_" + str(i) + ".npy"))).to(device)
        loss_fn = get_loss_fn()
        print("image index {}: pred vs target: {:.5f}; our pred vs target: {:.5f}; pred vs our pred: {:.5f}".format(i, loss_fn(pred, target).item(), loss_fn(our_pred, target).item(), loss_fn(pred, our_pred).item()))

        ''' # print distances among three models
        print(pred.min())
        print(our_pred.min())
        print(target.min())
        print(pred.max())
        print(our_pred.max())
        print(target.max())
        '''