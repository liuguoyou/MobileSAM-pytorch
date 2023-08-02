import os
import torch
import argparse
from mobile_sam import sam_model_registry

def parse_option():
    parser = argparse.ArgumentParser('argument for model aggregation')

    parser.add_argument('--ckpt', type=str, required=True)

    parser.add_argument('--mobile_sam_type', type=str, default="vit_t")
    parser.add_argument('--mobile_sam_ckpt', type=str, default="/dataset/vyueyu/project/MobileSAM/weights/mobile_sam.pt")
    
    parser.add_argument('--save_model_path', type=str, default="./weights")
    parser.add_argument('--save_model_name', type=str, default="our_retrained_mobilesam.pth")

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_option()

    # our retrained mobile sam 
    print("load model ...")
    mobile_sam = sam_model_registry[args.mobile_sam_type](checkpoint=args.mobile_sam_ckpt)
    mobile_sam.image_encoder.load_state_dict(torch.load(args.ckpt))

    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)
    torch.save(mobile_sam.state_dict(), os.path.join(args.save_model_path, args.save_model_name))
    print("Completed! The aggregated model is saved as {}".format(os.path.join(args.save_model_path, args.save_model_name)))
