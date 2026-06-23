# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Adapted InDuDoNet training script for CTMAR raw data.

Original: MICCAI2021 ``InDuDoNet: An Interpretable Dual Domain Network for CT Metal Artifact Reduction''
paper link: https://arxiv.org/pdf/2109.05298.pdf
"""
from __future__ import print_function
import argparse
import os
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import time
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from math import ceil
import wandb
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
from deeplesion.Dataset import CTMARDataset
from network.indudonet import InDuDoNet

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./data/01_raw/",
                    help='path to data/01_raw/ directory')
parser.add_argument("--index_path", type=str, default="./code/index/",
                    help='path to code/index/ directory with pkl files')
parser.add_argument("--anatomy", type=str, default="body", choices=["body", "head"],
                    help='anatomy type: body or head')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
parser.add_argument('--patchSize', type=int, default=512,
                    help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=100, help='total number of training epochs')
parser.add_argument('--num_channel', type=int, default=32,
                    help='the number of dual channels')
parser.add_argument('--T', type=int, default=4,
                    help='the number of ResBlocks in every ProxNet')
parser.add_argument('--S', type=int, default=10,
                    help='the number of total iterative stages')
parser.add_argument('--resume', type=int, default=0, help='continue to train')
parser.add_argument("--milestone", type=int, default=[40, 80],
                    help="When to decay learning rate")
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--log_dir', default='./logs/', help='tensorboard logs')
parser.add_argument('--model_dir', default='./models/', help='saving model')
parser.add_argument('--eta1', type=float, default=1,
                    help='initialization for stepsize eta1')
parser.add_argument('--eta2', type=float, default=5,
                    help='initialization for stepsize eta2')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='initialization for weight factor')
parser.add_argument('--gamma', type=float, default=1e-1,
                    help='hyper-parameter for balancing different loss items')
opt = parser.parse_args()

# create path
os.makedirs(opt.log_dir, exist_ok=True)
os.makedirs(opt.model_dir, exist_ok=True)

# wandb
wandb.init(
    project="CTMAR",
    name=f"InDuDoNet_{opt.anatomy}_bs{opt.batchSize}",
    config=vars(opt),
    resume="allow",
)
cudnn.benchmark = True


def validate_model(net, val_dataset, writer, step, epoch, anatomy):
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=int(opt.workers), pin_memory=True)
    net.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    psnr_count = 0
    with torch.no_grad():
        for idx, data in enumerate(val_loader):
            Xma, XLI, Xgt, mask, Sma, SLI, Sgt, Tr = [x.cuda() for x in data]
            ListX, ListS, ListYS = net(Xma, XLI, mask, Sma, SLI, Tr)
            loss_l2YS = F.mse_loss(ListYS[-1], Sgt)
            loss_l2X = F.mse_loss(ListX[-1] * (1 - mask), Xgt * (1 - mask))
            loss = opt.gamma * loss_l2YS + loss_l2X
            total_loss += loss.item()

            if idx < 50:
                pred = ListX[-1].squeeze().cpu().numpy()
                gt = Xgt.squeeze().cpu().numpy()
                msk = mask.squeeze().cpu().numpy()
                pred[msk > 0.5] = 0.0
                gt[msk > 0.5] = 0.0
                pred_norm = np.clip(pred / 255.0, 0.0, 1.0)
                gt_norm = np.clip(gt / 255.0, 0.0, 1.0)
                total_psnr += psnr_fn(gt_norm, pred_norm, data_range=1.0)
                total_ssim += ssim_fn(gt_norm, pred_norm, data_range=1.0,
                                      win_size=11, gaussian_weights=True)
                psnr_count += 1

    num_val = len(val_loader)
    avg_loss = total_loss / num_val
    avg_psnr = total_psnr / psnr_count if psnr_count > 0 else 0.0
    avg_ssim = total_ssim / psnr_count if psnr_count > 0 else 0.0

    writer.add_scalar('val/Loss', avg_loss, step)
    writer.add_scalar('val/PSNR', avg_psnr, step)
    writer.add_scalar('val/SSIM', avg_ssim, step)
    wandb.log({'val/Loss': avg_loss, 'val/PSNR': avg_psnr,
               'val/SSIM': avg_ssim, 'epoch': epoch + 1}, step=step)

    print(
        '[Val Epoch:{:>2d}] Loss={:5.2e}, PSNR={:.4f}, SSIM={:.4f}'.format(
            epoch + 1, avg_loss, avg_psnr, avg_ssim))
    return avg_loss


def train_model(net, optimizer, scheduler, datasets, val_dataset):
    data_loader = DataLoader(
        datasets, batch_size=opt.batchSize, shuffle=True,
        num_workers=int(opt.workers), pin_memory=True)
    num_data = len(datasets)
    num_iter_epoch = ceil(num_data / opt.batchSize)
    writer = SummaryWriter(opt.log_dir)
    step = 0
    best_val_loss = float('inf')
    for epoch in range(opt.resume, opt.niter):
        mse_per_epoch = 0
        tic = time.time()
        # train stage
        lr = optimizer.param_groups[0]['lr']
        for ii, data in enumerate(data_loader):
            Xma, XLI, Xgt, mask, Sma, SLI, Sgt, Tr = [x.cuda() for x in data]
            net.train()
            optimizer.zero_grad()
            ListX, ListS, ListYS = net(Xma, XLI, mask, Sma, SLI, Tr)
            loss_l2YSmid = 0.1 * F.mse_loss(ListYS[opt.S - 2], Sgt)
            loss_l2Xmid = 0.1 * F.mse_loss(
                ListX[opt.S - 2] * (1 - mask), Xgt * (1 - mask))
            loss_l2YSf = F.mse_loss(ListYS[-1], Sgt)
            loss_l2Xf = F.mse_loss(ListX[-1] * (1 - mask), Xgt * (1 - mask))
            loss_l2YS = loss_l2YSf + loss_l2YSmid
            loss_l2X = loss_l2Xf + loss_l2Xmid
            loss = opt.gamma * loss_l2YS + loss_l2X
            loss.backward()
            optimizer.step()
            mse_iter = loss.item()
            mse_per_epoch += mse_iter
            if ii % 400 == 0:
                template = (
                    '[Epoch:{:>2d}/{:<2d}] {:0>5d}/{:0>5d}, '
                    'Loss={:5.2e}, Lossl2YS={:5.2e}, Lossl2X={:5.2e}, lr={:.2e}'
                )
                print(template.format(
                    epoch + 1, opt.niter, ii, num_iter_epoch,
                    mse_iter, loss_l2YS, loss_l2X, lr))
            writer.add_scalar('Loss', loss, step)
            writer.add_scalar('Loss_YS', loss_l2YS, step)
            writer.add_scalar('Loss_X', loss_l2X, step)
            wandb.log({'train/Loss': mse_iter, 'train/Loss_YS': loss_l2YS.item(),
                       'train/Loss_X': loss_l2X.item(), 'lr': lr}, step=step)
            step += 1
        mse_per_epoch /= (ii + 1)
        print('Loss={:+.2e}'.format(mse_per_epoch))
        print('-' * 100)
        scheduler.step()
        # validation
        val_loss = validate_model(
            net, val_dataset, writer, step, epoch, opt.anatomy)
        ckpt_dict = {
            'epoch': epoch + 1,
            'step': step + 1,
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt_dict,
                       os.path.join(opt.model_dir, 'InDuDoNet_best.pt'))
            print('Best model saved (val_loss={:5.2e})'.format(best_val_loss))
        # save model
        torch.save(ckpt_dict,
                   os.path.join(opt.model_dir, 'InDuDoNet_latest.pt'))
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'step': step + 1,
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, os.path.join(opt.model_dir, 'InDuDoNet_%d.pt' % (epoch + 1)))
        toc = time.time()
        print('This epoch take time {:.2f}'.format(toc - tic))
    writer.close()
    print('Reach the maximal epochs! Finish training')


if __name__ == '__main__':
    def print_network(name, net):
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print('name={:s}, Total number={:d}'.format(name, num_params))

    net = InDuDoNet(opt).cuda()
    print_network("InDuDoNet:", net)
    optimizer = optim.Adam(net.parameters(), betas=(0.5, 0.999), lr=opt.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=opt.milestone, gamma=0.5)
    # from opt.resume continue to train
    if opt.resume:
        ckpt_path = os.path.join(opt.model_dir, 'InDuDoNet_%d.pt' % (opt.resume))
        ckpt = torch.load(ckpt_path, weights_only=False)
        if isinstance(ckpt, dict) and 'model' in ckpt:
            net.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
        else:
            net.load_state_dict(ckpt)
            for _ in range(opt.resume):
                scheduler.step()
        print('loaded checkpoints, epoch{:d}'.format(opt.resume))

    # load dataset using CTMARDataset
    train_dataset = CTMARDataset(
        data_root=opt.data_path,
        anatomy=opt.anatomy,
        index_path=opt.index_path,
        split='train')
    print(f'Training dataset: {len(train_dataset)} samples '
          f'(anatomy={opt.anatomy})')

    val_dataset = CTMARDataset(
        data_root=opt.data_path,
        anatomy=opt.anatomy,
        index_path=opt.index_path,
        split='val')
    print(f'Validation dataset: {len(val_dataset)} samples')

    # train model
    train_model(net, optimizer, scheduler, train_dataset, val_dataset)
