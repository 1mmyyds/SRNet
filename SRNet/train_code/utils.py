import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(model_path, epoch, iteration, model, optimizer):
    state = {
        'epoch': epoch,
        'iter': iteration,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    torch.save(state, os.path.join(model_path, 'net_%depoch.pth' % epoch))

class MRAE(nn.Module):
    def __init__(self):
        super(MRAE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = torch.abs(outputs - label) / label
        mrae = torch.mean(error.contiguous().view(-1))
        return mrae

class RMSE(nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs-label
        sqrt_error = torch.pow(error,2)
        rmse = torch.sqrt(torch.mean(sqrt_error.contiguous().view(-1)))
        return rmse
    
class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs-label
        sqrt_error = torch.pow(error,2)
        mse = torch.mean(sqrt_error.contiguous().view(-1))
        return mse

class PSNR(nn.Module):
    def __init__(self):
        super(PSNR, self).__init__()

    def forward(self, im_true, im_fake, data_range=255):
        N = im_true.size()[0]
        C = im_true.size()[1]
        H = im_true.size()[2]
        W = im_true.size()[3]
        Itrue = im_true.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
        Ifake = im_fake.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
        mse = nn.MSELoss(reduce=False)
        err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C * H * W)
        psnr = 10. * torch.log((data_range ** 2) / err) / np.log(10.)
        return torch.mean(psnr)

class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1  # placeholder, will be updated on the fly
        self.window = None

    def _create_window(self, window_size, channel):
        def gauss(window_size, sigma):
            center = window_size // 2
            gauss_vals = torch.tensor([
                ((x - center) ** 2) for x in range(window_size)
            ], dtype=torch.float32)
            gauss_vals = torch.exp(-gauss_vals / (2 * sigma ** 2))
            return gauss_vals / gauss_vals.sum()

        _1D_window = gauss(window_size, 1.5).unsqueeze(1)  # shape [window_size, 1]
        _2D_window = _1D_window @ _1D_window.T  # outer product -> [window_size, window_size]
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if self.window is None or self.channel != channel:
            self.window = self._create_window(self.window_size, channel).to(img1.device)
            self.channel = channel

        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

class Structure_Pixel_Loss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        """
        组合损失函数，结合MSE和SSIM。
        Args:
            alpha (float): MSE损失的权重。
            beta (float): SSIM损失的权重。
        """
        super(Structure_Pixel_Loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()
        self.ssim = SSIM()

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        ssim_loss = 1 - self.ssim(pred, target)  # SSIM越高，损失越低
        return self.alpha * mse_loss + self.beta * ssim_loss
