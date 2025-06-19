import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io

class TrainDataset(Dataset):
    def __init__(self, data_root,  arg=True, bgr2rgb=True):

        self.hypers = []  # ground_truth
        self.hyper_ws = []  # ground_truth_w
        self.specs = []  # fusion
        self.hyper_ss = []  # ground_truth_s
        self.arg = arg

        hyper_data_path = f'{data_root}/Train_data/'
        rgb_data_path = f'{data_root}/Train_data/'

        with open(f'{data_root}/split_txt/Train_list.txt', 'r') as fin:
            hyper_list = [line.replace('\n','.mat') for line in fin]
            rgb_list = hyper_list
        hyper_list.sort()
        rgb_list.sort()

        for i in range(len(hyper_list)):
            hyper_path = hyper_data_path + hyper_list[i]
            if 'mat' not in hyper_path:
                continue
            mat = scipy.io.loadmat(hyper_path)
            hyper = torch.tensor(np.array(mat['ground_truth']))  # [64,64,32]
            hyper_w = torch.tensor(np.array(mat['ground_truth_w']))  # [64,64,32]

            hyper_s = np.float32(np.array(mat['ground_truth_s']))  # [64,64,32]
            hyper_s = np.transpose(hyper_s, [2, 0, 1])  # [32,64,64]

            spec = torch.cat([hyper, hyper_w], dim=2)  # [64,64,64]
            spec = np.float32(np.array(spec))
            spec = np.transpose(spec, [2, 0, 1])  # [32,64,64]


            self.hypers.append(hyper)
            self.specs.append(spec)
            self.hyper_ws.append(hyper_w)
            self.hyper_ss.append(hyper_s)

        self.img_num = len(self.hypers)


    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):
        spec = self.specs[idx]
        hyper_s = self.hyper_ss[idx]
        return np.ascontiguousarray(spec), np.ascontiguousarray(hyper_s)

    def __len__(self):
        return self.img_num


class ValidDataset(Dataset):
    def __init__(self, data_root,  arg=True, bgr2rgb=True):

        self.hypers = []  # ground_truth
        self.hyper_ws = []  # ground_truth_w
        self.specs = []  # fusion
        self.hyper_ss = []  # ground_truth_s
        self.arg = arg

        hyper_data_path = f'{data_root}/Valid_data/'
        rgb_data_path = f'{data_root}/Valid_data/'

        with open(f'{data_root}/split_txt/Valid_list.txt', 'r') as fin:
            hyper_list = [line.replace('\n','.mat') for line in fin]
            rgb_list = hyper_list
        hyper_list.sort()
        rgb_list.sort()

        for i in range(len(hyper_list)):
            hyper_path = hyper_data_path + hyper_list[i]
            if 'mat' not in hyper_path:
                continue
            mat = scipy.io.loadmat(hyper_path)
            hyper = torch.tensor(np.array(mat['ground_truth']))  # [64,64,32]
            hyper_w = torch.tensor(np.array(mat['ground_truth_w']))  # [64,64,32]

            hyper_s = np.float32(np.array(mat['ground_truth_s']))  # [64,64,32]
            hyper_s = np.transpose(hyper_s, [2, 0, 1])  # [32,64,64]

            spec = torch.cat([hyper, hyper_w], dim=2)  # [64,64,64]
            spec = np.float32(np.array(spec))
            spec = np.transpose(spec, [2, 0, 1])  # [32,64,64]

            self.hypers.append(hyper)
            self.specs.append(spec)
            self.hyper_ws.append(hyper_w)
            self.hyper_ss.append(hyper_s)

        self.img_num = len(self.hypers)


    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):
        spec = self.specs[idx]
        hyper_s = self.hyper_ss[idx]
        return np.ascontiguousarray(spec), np.ascontiguousarray(hyper_s)

    def __len__(self):
        return self.img_num