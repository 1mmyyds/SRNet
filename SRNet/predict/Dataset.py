from torch.utils.data import Dataset
import numpy as np
import cv2
import scipy.io

class TrainDataset(Dataset):
    def __init__(self, data_root,  arg=True, bgr2rgb=True):

        self.hypers = []  
        self.rgbs = []  
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
            hyper =np.float32(np.array(mat['ground_truth']))  # [64,64,32]
            hyper = np.transpose(hyper, [2, 0, 1])  # [32,64,64]
            rgb_path = rgb_data_path + rgb_list[i]
            assert hyper_list[i].split('.')[0] ==rgb_list[i].split('.')[0], 'Hyper and RGB come from different scenes.'


            if 'mat' not in rgb_path:
                continue
            mat = scipy.io.loadmat(rgb_path)
            rgb =np.float32(np.array(mat['rgb']))  # [64,64,3]

            if bgr2rgb:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = np.float32(rgb)
            rgb = (rgb-rgb.min())/(rgb.max()-rgb.min())
            rgb = np.transpose(rgb, [2, 0, 1])  # [3,64,64]

            self.hypers.append(hyper)
            self.rgbs.append(rgb)

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
        hyper = self.hypers[idx]
        rgb = self.rgbs[idx]
        return np.ascontiguousarray(rgb), np.ascontiguousarray(hyper)

    def __len__(self):
        return self.img_num


class ValidDataset(Dataset):
    def __init__(self, data_root, arg=True, bgr2rgb=True):

        self.hypers = []  
        self.rgbs = []  
        self.arg = arg

        hyper_data_path = f'{data_root}/Valid_data/'
        rgb_data_path = f'{data_root}/Valid_data/'

        with open(f'{data_root}/split_txt/Valid_list.txt', 'r') as fin:
            hyper_list = [line.replace('\n', '.mat') for line in fin]
            rgb_list = hyper_list
        hyper_list.sort()
        rgb_list.sort()

        for i in range(len(hyper_list)):
            hyper_path = hyper_data_path + hyper_list[i]
            if 'mat' not in hyper_path:
                continue
            mat = scipy.io.loadmat(hyper_path)
            hyper = np.float32(np.array(mat['ground_truth']))  # [64,64,32]
            hyper = np.transpose(hyper, [2, 0, 1])  # [32,64,64]
            rgb_path = rgb_data_path + rgb_list[i]
            assert hyper_list[i].split('.')[0] == rgb_list[i].split('.')[0], 'Hyper and RGB come from different scenes.'

            if 'mat' not in rgb_path:
                continue
            mat = scipy.io.loadmat(rgb_path)
            rgb = np.float32(np.array(mat['rgb']))  # [64,64,3]

            if bgr2rgb:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = np.float32(rgb)
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
            rgb = np.transpose(rgb, [2, 0, 1])  # [3,64,64]

            self.hypers.append(hyper)
            self.rgbs.append(rgb)

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
        hyper = self.hypers[idx]
        rgb = self.rgbs[idx]
        return np.ascontiguousarray(rgb), np.ascontiguousarray(hyper)

    def __len__(self):
        return self.img_num
