import torch
import numpy as np
import argparse
import os
import torch.backends.cudnn as cudnn
from architecture import *
from utils import AverageMeter, save_mat, MSE, RMSE, PSNR, SSIM
from Dataset_fusion import ValidDataset
from torch.utils.data import DataLoader

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

parser = argparse.ArgumentParser(description="predict")
parser.add_argument('--data_root', type=str, default='D:/PycharmProjects/pythonProject2/YHY/dataset')
parser.add_argument('--method', type=str, default='SRNet_L_fusion')
parser.add_argument('--pretrained_model_path', type=str, default='C:/Users/1mm/Desktop/SRNet/SRNet-main/SRNet/train_code/exp/P/net_1epoch.pth')
parser.add_argument('--outf', type=str, default='C:/Users/1mm/Desktop/SRNet/SRNet-main/SRNet/predict/exp/')
parser.add_argument("--gpu_id", type=str, default='0')
opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

# load dataset
val_data = ValidDataset(data_root=opt.data_root, bgr2rgb=True)
val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

# loss function
criterion_mse = MSE()
criterion_rmse = RMSE()
criterion_psnr = PSNR()
criterion_ssim = SSIM()

if torch.cuda.is_available():
    criterion_mse.cuda()
    criterion_rmse.cuda()
    criterion_psnr.cuda()
    criterion_ssim.cuda()

# Validate
with open(f'{opt.data_root}/split_txt/valid_list.txt', 'r') as fin:
    hyper_list = [line.replace('\n', '.mat') for line in fin]
hyper_list.sort()
var_name = 'ground_truth'
def validate(val_loader, model):
    model.eval()
    losses_mse = AverageMeter()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()
    losses_ssim = AverageMeter()
    print(f"Validation batches: {len(val_loader)}")
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        with torch.no_grad():
                output = model(input)
                mse = criterion_mse(output, target)
                rmse = criterion_rmse(output, target)
                psnr = criterion_psnr(output, target)
                ssim = criterion_ssim(output, target)

        # record loss
        losses_mse.update(mse.data)
        losses_rmse.update(rmse.data)
        losses_psnr.update(psnr.data)
        losses_ssim.update(ssim.data)

        result = output.cpu().numpy() * 1.0
        result = np.transpose(np.squeeze(result), [1, 2, 0])
        result = np.minimum(result, 1.0)
        result = np.maximum(result, 0)
        mat_name = hyper_list[i]
        mat_dir = os.path.join(opt.outf, mat_name)
        save_mat(mat_dir, var_name, result)
    return losses_mse.avg, losses_rmse.avg, losses_psnr.avg, losses_ssim.avg

if __name__ == '__main__':
    cudnn.benchmark = True
    pretrained_model_path = opt.pretrained_model_path
    method = opt.method
    model = model_select(method, pretrained_model_path).cuda()
    mse, rmse, psnr, ssim = validate(val_loader, model)
    print(f'method:{method}, mse:{mse}, rmse:{rmse}, psnr:{psnr}, ssim:{ssim}')
