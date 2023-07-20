import os
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter  

from dataset import transform, get_sa1b_dataloaders
from mobile_sam.modeling import TinyViT

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--dataset_path', type=str, default="/dataset/vyueyu/sa-1b", help='root path of dataset')

    parser.add_argument('--epochs', type=int, default=8, help='number of training epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size')

    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--seed', type=int, default=1234, help='seed')
    parser.add_argument('--deterministic', type=bool, default=True, help='deterministic')
    parser.add_argument('--benchmark', type=bool, default=False, help='benchmark')

    parser.add_argument('--optim', type=str, default='sgd', choices=['adam', 'sgd', 'adamw'])
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    parser.add_argument('--print_iters', type=int, default=500, help='print loss iterations')

    parser.add_argument('--eval_nums', type=int, default=200, help='evaluation numbers')
    parser.add_argument('--eval_iters', type=int, default=5000, help='evaluation iterations')

    parser.add_argument('--root_path', type=str, default="/dataset/vyueyu/project/MobileSAM", help='root path')
    parser.add_argument('--work_dir', type=str, default="work_dir", help='work directory')
    parser.add_argument('--save_dir', type=str, default="ckpt", help='save directory')
    parser.add_argument('--log_dir', type=str, default="log", help='save directory')
    parser.add_argument('--save_iters', type=int, default=50000, help='save iterations')

    args = parser.parse_args()
    return args

def build_model():
    model = TinyViT(img_size=1024, in_chans=3, num_classes=1000,
                embed_dims=[64, 128, 160, 320],
                depths=[2, 2, 6, 2],
                num_heads=[2, 4, 5, 10],
                window_sizes=[7, 7, 14, 7],
                mlp_ratio=4.,
                drop_rate=0.,
                drop_path_rate=0.0,
                use_checkpoint=False,
                mbconv_expand_ratio=4.0,
                local_conv_size=3,
                layer_lr_decay=0.8
            )
    return model

def loss_eval(args, model, val_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for idx, (imgs, target_feats) in enumerate(val_loader):
            imgs = imgs.to(device)
            target_feats = target_feats.to(device)
            pred_feats = model(imgs)
            loss_fn = get_loss_fn()
            total_loss += loss_fn(pred_feats, target_feats)
    
    return total_loss / len(val_loader)

def get_optimizer(args, model):
    if args.optim == 'adam':
        return optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        return optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        return optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(args.optim)

def get_scheduler(args, optimizer):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.5)
    
def customized_mseloss(pred_feats, target_feats):
    # return (0.5 * (pred_feats - target_feats) ** 2).sum(1).mean()
    return ((pred_feats - target_feats) ** 2).sum(1).mean().sqrt()


def get_loss_fn():
    return customized_mseloss
    # return nn.MSELoss()
    
def main():
    args = parse_option()

    if not os.path.exists(os.path.join(args.root_path, args.work_dir, args.save_dir)):
        os.makedirs(os.path.join(args.root_path, args.work_dir, args.save_dir))
    
    if not os.path.exists(os.path.join(args.root_path, args.work_dir, args.log_dir)):
        os.makedirs(os.path.join(args.root_path, args.work_dir, args.log_dir))

    device = args.device

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = args.deterministic
        cudnn.benchmark = args.benchmark
    
    train_dir = ["sa_00000" + str(i) for i in range(10)]
    val_dir = ['sa_000010']

    train_loader, val_loader = get_sa1b_dataloaders(transform, args.dataset_path, train_dir, val_dir, args.batch_size, args.num_workers, args.eval_nums)

    writer = SummaryWriter(os.path.join(args.root_path, args.work_dir, args.log_dir))

    model = build_model()
    model.to(device)
    
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)

    iters = 0
    
    for epoch in range(1, args.epochs + 1):
        print("------start epoch {}------".format(epoch))

        for idx, (imgs, target_feats) in enumerate(train_loader):
            model.train()
            imgs = imgs.to(device)
            target_feats = target_feats.to(device)

            optimizer.zero_grad()
            pred_feats = model(imgs)
            loss_fn = get_loss_fn()
            loss = loss_fn(pred_feats, target_feats)
            loss.backward()
            optimizer.step()

            iters += 1

            if iters % args.print_iters == 0:
                print("iter {}: mseloss {}".format(iters, loss.item()))
                writer.add_scalar("mse_loss", loss.item(), iters)

            if iters % args.save_iters == 0:
                save_path = os.path.join(args.root_path, args.work_dir, args.save_dir, str(iters) + ".pth")
                print("save model to {}".format(save_path))
                torch.save(model.state_dict(), save_path)

            if iters % args.eval_iters == 0:
                eval_loss = loss_eval(args, model, val_loader, device)
                writer.add_scalar("eval_mse_loss", eval_loss, iters)
                print("---iter {} eval loss: {}---".format(iters, eval_loss))
        
        scheduler.step()

    torch.save(model.state_dict(), os.path.join(args.root_path, args.work_dir, args.save_dir, "final.pth"))

if __name__ == "__main__":
    main()




    



