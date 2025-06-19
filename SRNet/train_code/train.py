import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from Dataset_fusion import TrainDataset, ValidDataset
from architecture import *
from utils import AverageMeter, save_checkpoint, MRAE, RMSE, PSNR, MSE, SSIM, Structure_Pixel_Loss




parser = argparse.ArgumentParser(description="SRNet")
parser.add_argument('--method', type=str, default='SRNet_L_fusion')  # 选择网络模型
parser.add_argument('--pretrained_model_path', type=str, default=None)
parser.add_argument("--batch_size", type=int, default=8, help="batch size")  # 批量大小
parser.add_argument("--end_epoch", type=int, default=72, help="number of epochs")  # 迭代周期
parser.add_argument("--init_lr", type=float, default=4e-4, help="initial learning rate")  # 学习速率
parser.add_argument("--outf", type=str, default='C:/Users/1mm/Desktop/SRNet/SRNet-main/SRNet/train_code/exp/P/', help='path log files')  # 模型输出路径
parser.add_argument("--data_root", type=str, default='D:/PycharmProjects/pythonProject2/YHY/dataset')  # 选择数据集
parser.add_argument("--patch_size", type=int, default=64, help="patch size")  # 图像分割大小
parser.add_argument("--stride", type=int, default=1, help="stride")  # 图像分割步长
parser.add_argument("--gpu_id", type=str, default='0', help='path log files')
opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

# load dataset
train_data = TrainDataset(data_root=opt.data_root,  bgr2rgb=True, arg=True)
val_data = ValidDataset(data_root=opt.data_root, bgr2rgb=True)


# iterations
per_epoch_iteration = 750
total_iteration = per_epoch_iteration*opt.end_epoch

# loss function
criterion_mrae = MRAE()
criterion_rmse = RMSE()
criterion_psnr = PSNR()
criterion_mse = MSE()
criterion_ssim = SSIM()
criterion = Structure_Pixel_Loss(1,1)


# model
pretrained_model_path = opt.pretrained_model_path
method = opt.method
model = model_select(method, pretrained_model_path).cuda()
print('Parameters number is ', sum(param.numel() for param in model.parameters()))

if torch.cuda.is_available():
    model.cuda()
    criterion_mrae.cuda()
    criterion_rmse.cuda()
    criterion_psnr.cuda()
    criterion_mse.cuda()
    criterion_ssim.cuda()
    criterion.cuda()

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

optimizer = optim.Adam(model.parameters(), lr=opt.init_lr, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iteration, eta_min=1e-6)

# Resume
resume_file = opt.pretrained_model_path
if resume_file is not None:
    if os.path.isfile(resume_file):
        print("=> loading checkpoint '{}'".format(resume_file))
        checkpoint = torch.load(resume_file)
        start_epoch = checkpoint['epoch']
        iteration = checkpoint['iter']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

def main():
    cudnn.benchmark = True
    iteration = 0
    while iteration<total_iteration:
        model.train()
        losses = AverageMeter()
        train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, num_workers=2,
                                  pin_memory=True, drop_last=True)
        val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

        for i, (images, labels) in enumerate(train_loader):
            labels = labels.cuda()
            images = images.cuda()
            images = Variable(images)
            labels = Variable(labels)
            lr = optimizer.param_groups[0]['lr']
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.update(loss.data)
            iteration = iteration + 1
            if iteration % 50 == 0:
                print('[iter:%d/%d],lr=%.9f,train_losses.avg=%.9f'
                      % (iteration, total_iteration, lr, losses.avg))
            if iteration % 1000 == 0:
                print('进入保存循环')
                mse, rmse, psnr, ssim = validate(val_loader, model)
                print(f'MSE:{mse}, RMSE: {rmse}, PNSR:{psnr}, SSIM:{ssim}')
                print(f'Saving to {opt.outf}')
                save_checkpoint(opt.outf, (iteration // 1000), iteration, model, optimizer)
                print('保存成功')
    return 0

# Validate
def validate(val_loader, model):
    model.eval()
    mse = AverageMeter()
    rmse = AverageMeter()
    psnr = AverageMeter()
    ssim = AverageMeter()

    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        with torch.no_grad():
                output = model(input)
                loss_mse = criterion_mse(output, target)
                loss_rmse = criterion_rmse(output, target)
                loss_psnr = criterion_psnr(output, target)
                loss_ssim = criterion_ssim(output, target)

        mse.update(loss_mse.data)
        rmse.update(loss_rmse.data)
        psnr.update(loss_psnr.data)
        ssim.update(loss_ssim.data)
    return mse.avg, rmse.avg, psnr.avg ,ssim.avg

if __name__ == '__main__':
    main()


