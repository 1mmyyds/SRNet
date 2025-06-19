import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io
import os


class TrainDataset(Dataset):
    def __init__(self, data_root, arg=True, bgr2rgb=True):
        self.data_root = data_root
        self.arg = arg
        self.file_list = self._load_file_list(f'{data_root}/split_txt/Train_list.txt')
        self.img_num = len(self.file_list)

    def _load_file_list(self, list_path):
        with open(list_path, 'r') as fin:
            return [line.strip() + '.mat' for line in fin if line.strip()]

    def _load_mat_data(self, idx):
        """按需加载单个.mat文件的数据"""
        mat_path = os.path.join(self.data_root, 'Train_data', self.file_list[idx])
        mat = scipy.io.loadmat(mat_path)

        hyper = np.float32(mat['ground_truth'])  # [64,64,32]
        hyper_w = np.float32(mat['ground_truth_w'])  # [64,64,32]
        hyper_s = np.float32(mat['ground_truth_s'])  # [64,64,32]

        # 合并通道并转置
        spec = np.concatenate([hyper, hyper_w], axis=2)  # [64,64,64]
        spec = np.transpose(spec, (2, 0, 1))  # [32,64,64]
        hyper_s = np.transpose(hyper_s, (2, 0, 1))  # [32,64,64]

        return spec, hyper_s

    def __getitem__(self, idx):
        # 按需加载数据
        spec, hyper_s = self._load_mat_data(idx)


        if self.arg:
            rotTimes = np.random.randint(0, 4)
            vFlip = np.random.randint(0, 2)
            hFlip = np.random.randint(0, 2)
            spec = self._augment(spec, rotTimes, vFlip, hFlip)
            hyper_s = self._augment(hyper_s, rotTimes, vFlip, hFlip)

        return (
            torch.from_numpy(np.ascontiguousarray(spec)),
            torch.from_numpy(np.ascontiguousarray(hyper_s))
        )

    def _augment(self, img, rotTimes, vFlip, hFlip):
        """数据增强函数"""
        # Random rotation
        for _ in range(rotTimes):
            img = np.rot90(img, axes=(1, 2))
        # Random vertical Flip
        if vFlip:
            img = img[:, :, ::-1]
        # Random horizontal Flip
        if hFlip:
            img = img[:, ::-1, :]
        return img.copy()  # 确保内存连续

    def __len__(self):
        return self.img_num


class ValidDataset(Dataset):
    def __init__(self, data_root, arg=True, bgr2rgb=True):
        self.data_root = data_root
        self.arg = arg
        self.file_list = self._load_file_list(f'{data_root}/split_txt/Valid_list.txt')
        self.img_num = len(self.file_list)

    def _load_file_list(self, list_path):
        """加载文件列表并过滤无效文件"""
        with open(list_path, 'r') as fin:
            return [line.strip() + '.mat' for line in fin if line.strip()]

    def _load_mat_data(self, idx):
        """按需加载单个.mat文件的数据"""
        mat_path = os.path.join(self.data_root, 'Valid_data', self.file_list[idx])
        mat = scipy.io.loadmat(mat_path)

        # 加载并转换数据
        hyper = np.float32(mat['ground_truth'])  # [64,64,32]
        hyper_w = np.float32(mat['ground_truth_w'])  # [64,64,32]
        hyper_s = np.float32(mat['ground_truth_s'])  # [64,64,32]

        # 合并通道（沿通道维度拼接）
        spec = np.concatenate([hyper, hyper_w], axis=2)  # [64,64,64]
        spec = np.transpose(spec, (2, 0, 1))  # [32,64,64]
        hyper_s = np.transpose(hyper_s, (2, 0, 1))  # [32,64,64]

        return spec, hyper_s

    def __getitem__(self, idx):
        # 按需加载数据
        spec, hyper_s = self._load_mat_data(idx)

        # 转换为Tensor并确保内存连续
        return (
            torch.from_numpy(np.ascontiguousarray(spec)),
            torch.from_numpy(np.ascontiguousarray(hyper_s))
        )

    def __len__(self):
        return self.img_num