import cv2
import os
import json
import math
import h5py
import scipy
import numpy as np
import scipy.io as sio
import pandas as pd
import torch
from torch.utils.data import Dataset
import sys
import datasets.transforms as transforms
from torchvision.transforms.functional import resize
import torchvision


ubfc_dir = "/data/wuruize/UBFC_process"
pure_dir = "/data/wuruize/pure_process/"
vipl_dir = "/data/wuruize/VIPL"
cohface_dir = "/data/wuruize/cohface"
mmsehr_dir = "/data/wuruize/MMSE-HR_process"

def cal_hr(output : torch.Tensor, Fs : float):
    '''
    args:
        output: (1, T)
        Fs: sampling rate
    return:
        hr: heart rate
    '''
    def compute_complex_absolute_given_k(output : torch.Tensor, k : torch.Tensor, N : int):
        two_pi_n_over_N = 2 * math.pi * torch.arange(0, N, dtype=torch.float) / N
        hanning = torch.from_numpy(np.hanning(N)).type(torch.FloatTensor).view(1, -1)

        k = k.type(torch.FloatTensor)
        two_pi_n_over_N = two_pi_n_over_N
        hanning = hanning

        output = output.view(1, -1) * hanning
        output = output.view(1, 1, -1).type(torch.FloatTensor)
        k = k.view(1, -1, 1)
        two_pi_n_over_N = two_pi_n_over_N.view(1, 1, -1)
        complex_absolute = torch.sum(output * torch.sin(k * two_pi_n_over_N), dim=-1) ** 2 \
                           + torch.sum(output * torch.cos(k * two_pi_n_over_N), dim=-1) ** 2
        return complex_absolute

    output = output.view(1, -1)

    N = output.size()[1]
    bpm_range = torch.arange(40, 180, dtype=torch.float)
    unit_per_hz = Fs / N
    feasible_bpm = bpm_range / 60.0
    k = feasible_bpm / unit_per_hz

    # only calculate feasible PSD range [0.7, 4]Hz
    complex_absolute = compute_complex_absolute_given_k(output, k, N)
    complex_absolute = (1.0 / complex_absolute.sum()) * complex_absolute
    whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0) # max返回（values, indices）
    whole_max_idx = whole_max_idx.type(torch.float) # 功率谱密度的峰值对应频率即为心率

    return whole_max_idx + 40	# Analogous Softmax operator

class BaseDataset(Dataset):
    def __init__(self, data_dir, train=True, T=-1, w=64, h=64, aug='figsc', fold=1, norm_type="reconstruct"):
        """
        :param data_dir: 数据集的根目录
        :param train: 是否是训练集
        :param T: 读取的帧数，-1表示读取整个视频
        :param transform_rate: getitem返回的视频的帧率和ecg的采样率
        """
        self.data_dir = data_dir
        self.train = train
        self.T = T
        self.w = w
        self.h = h
        self.aug = aug
        self.speed_slow = 0.6
        self.speed_fast = 1.4
        self.fold = fold
        self.norm_type = norm_type
        self.data_list = list()
        self.get_data_list() # 返回一个list，每个元素是一个dict，frame_path(一个列表), ecg, frame_rate(原始的，需要更改频率), ecg_rate（原始的需要更改频率）
        self.set_augmentations()

    def get_data_list(self):
        raise NotImplementedError

    def set_augmentations(self):
        self.aug_flip = False
        self.aug_illum = False
        self.aug_gauss = False
        self.aug_speed = False
        self.aug_resizedcrop = False
        if self.train == True:
            self.aug_flip = True if 'f' in self.aug else False
            self.aug_illum = True if 'i' in self.aug else False
            self.aug_gauss = True if 'g' in self.aug else False
            self.aug_speed = True if 's' in self.aug else False
            self.aug_resizedcrop = True if 'c' in self.aug else False
        self.aug_reverse = False ## Don't use this with supervised

    def apply_transformations(self, clip, seg_video, shading_video, idcs, augment=True):
        speed = 1.0
        if augment:
            ## Time resampling
            if self.aug_speed and np.random.rand() > 0.5:
                clip, return_idcs, speed = transforms.augment_speed(clip, idcs, self.T, self.speed_slow, self.speed_fast) # clip: (T, H, W, C) -> (C, T, H, W)
                if self.norm_type == "reconstruct":
                    # print("seg_video.shape: ", seg_video.shape, "shading_video.shape: ", shading_video.shape, speed)
                    seg_video, _, _ = transforms.augment_speed_c(seg_video, idcs, self.T, speed)
                    shading_video, _, _ = transforms.augment_speed_c(shading_video, idcs, self.T, speed)
                    # print("seg_video.shape: ", seg_video.shape, "shading_video.shape: ", shading_video.shape)
            else:
                return_idcs = idcs.copy()
                clip = clip[idcs].transpose(3, 0, 1, 2) # (T, H, W, C) -> (C, T, H, W)
                if self.norm_type == "reconstruct":
                    seg_video = seg_video[idcs].transpose(3, 0, 1, 2)
                    shading_video = shading_video[idcs].transpose(3, 0, 1, 2)

            ## Randomly horizontal flip
            if self.aug_flip:
                clip = transforms.augment_horizontal_flip(clip)
                if self.norm_type == "reconstruct":
                    seg_video = transforms.augment_horizontal_flip(seg_video)
                    shading_video = transforms.augment_horizontal_flip(shading_video)

            ## Randomly reverse time
            if self.aug_reverse and np.random.rand() > 0.5:
                clip = transforms.augment_time_reversal(clip)
                if self.norm_type == "reconstruct":
                    seg_video = transforms.augment_time_reversal(seg_video)
                    shading_video = transforms.augment_time_reversal(shading_video)

            ## Illumination noise
            if self.aug_illum:
                clip = transforms.augment_illumination_noise(clip)

            ## Gaussian noise for every pixel
            if self.aug_gauss:
                clip = transforms.augment_gaussian_noise(clip)

            ## Random resized cropping
            if self.aug_resizedcrop and np.random.rand() > 0.5:
                clip, scale = transforms.random_resized_crop(clip)
                if self.norm_type == "reconstruct":
                    seg_video = transforms.random_resized_crop_crop_scale(seg_video, scale)[0]
                    shading_video = transforms.random_resized_crop_crop_scale(shading_video, scale)[0]


        if self.norm_type == "reconstruct":
            clip = np.clip(clip, 0, 255)
            clip, seg_video, shading_video = torch.from_numpy(clip).float(), torch.from_numpy(seg_video.copy()).float(), torch.from_numpy(shading_video.copy()).float()
            clip = clip.div_(255.0).pow_(2.4)
            seg_video = seg_video.div_(255.0)
            shading_video = shading_video.div_(255.0)
        elif self.norm_type == "kinetics":
            mean = torch.tensor([0.4345, 0.4051, 0.3775]).view(3, 1, 1, 1)
            std = torch.tensor([0.2768, 0.2713, 0.2737]).view(3, 1, 1, 1)
            clip = torch.from_numpy(clip).float()
            clip = clip.div_(255.0)
            clip = clip.sub_(mean).div_(std)
        else:
            clip = np.clip(clip, 0, 255)
            clip = torch.from_numpy(clip).float()
            # clip = clip.add_(-127.5).div_(127.5)
            clip = clip / 255
        return clip, seg_video, shading_video, return_idcs, speed

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        """return: video, ecg, transform_rate, frame_start, frame_end"""
        sample = self.data_list[index]
        start_idx = sample['start_idx']
        video_length = sample['video_length']
        seg_video, shading_video = None, None
        if start_idx + int(self.T * 1.5) > video_length:
            with h5py.File(os.path.join(sample['location'], 'sample.hdf5'), 'r') as f:
                video_x = np.array(f['video_data'][:, start_idx: start_idx + self.T]).transpose(1, 2, 3, 0) # T, H, W, C
                if self.norm_type == "reconstruct":
                    seg_video = np.array(f['seg_video'][start_idx: start_idx + self.T]) # T, H, W, C
                    # seg_video = np.expand_dims(seg_video, axis=-1)
                    shading_video = np.array(f['shading_video'][start_idx: start_idx + self.T]) # T, H, W, C
                ecg = np.array(f['ecg_data'][start_idx: start_idx + self.T]) # T
        else:
            with h5py.File(os.path.join(sample['location'], 'sample.hdf5'), 'r') as f:
                video_x = np.array(f['video_data'][:, start_idx: start_idx + int(self.T * 1.5)]).transpose(1, 2, 3, 0) # T, H, W, C
                if self.norm_type == "reconstruct":
                    seg_video = np.array(f['seg_video'][start_idx: start_idx + int(self.T * 1.5)])
                    shading_video = np.array(f['shading_video'][start_idx: start_idx + int(self.T * 1.5)])
                ecg = np.array(f['ecg_data'][start_idx: start_idx + int(self.T * 1.5)]) # T
        idcs = np.arange(0, self.T, dtype=int) if self.T != -1 else np.arange(len(video_x), dtype=int)
        video_x_aug, seg_video, shading_video, speed_idcs, speed = self.apply_transformations(video_x, seg_video, shading_video, idcs)

        if speed != 1.0:
            min_idx = int(speed_idcs[0])
            max_idx = int(speed_idcs[-1])+1
            orig_x = np.arange(min_idx, max_idx, dtype=int)
            orig_wave = ecg[orig_x]
            wave = np.interp(speed_idcs, orig_x, orig_wave)
        else:
            wave = ecg[idcs]


        wave = (wave - wave.min()) / (wave.max() - wave.min())
        wave = torch.from_numpy(wave).float()

        if self.w != 64 or self.h != 64:
            video_x_aug = video_x_aug.transpose(1,0)
            video_x_aug = resize(video_x_aug, [self.w, self.h], interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
            video_x_aug = video_x_aug.transpose(1,0)
            if self.norm_type == "reconstruct":
                seg_video = seg_video.transpose(1,0)
                seg_video = resize(seg_video, [self.w, self.h], interpolation=torchvision.transforms.InterpolationMode.BILINEAR).transpose(1,0)
                shading_video = shading_video.transpose(1,0)
                shading_video = resize(shading_video, [self.w, self.h], interpolation=torchvision.transforms.InterpolationMode.BILINEAR).transpose(1,0)
        sample_item = {}
        sample_item["location"] = sample['location'].replace(self.data_dir, "")
        sample_item['video'] = video_x_aug
        sample_item['ecg'] = wave
        sample_item['clip_avg_hr'] = cal_hr(wave, 30)
        if self.norm_type == "reconstruct":
            sample_item['seg'] = seg_video.squeeze()
            sample_item['shading'] = shading_video.squeeze()

        return sample_item
        # return video_x_aug, seg_video.squeeze(), shading_video.squeeze(), wave, cal_hr(wave, 30)

    def get_hr(self, index):
        """return: video, ecg, transform_rate, frame_start, frame_end"""
        sample = self.data_list[index]
        start_idx = sample['start_idx']
        video_length = sample['video_length']
        seg_video, shading_video = None, None
        if start_idx + int(self.T * 1.5) > video_length:
            with h5py.File(os.path.join(sample['location'], 'sample.hdf5'), 'r') as f:
                ecg = np.array(f['ecg_data'][start_idx: start_idx + self.T]) # T
        else:
            with h5py.File(os.path.join(sample['location'], 'sample.hdf5'), 'r') as f:
                ecg = np.array(f['ecg_data'][start_idx: start_idx + int(self.T * 1.5)]) # T
        idcs = np.arange(0, self.T, dtype=int) if self.T != -1 else np.arange(len(ecg), dtype=int)
        wave = ecg[idcs]

        wave = (wave - wave.min()) / (wave.max() - wave.min())
        wave = torch.from_numpy(wave).float()

        sample_item = {}
        sample_item["location"] = sample['location'].replace(self.data_dir, "")
        sample_item['ecg'] = wave
        sample_item['clip_avg_hr'] = cal_hr(wave, 30)

        return sample_item

    def remove_data(self, location):
        new_data_list = []
        for sample in self.data_list:
            _name = sample["location"].replace(self.data_dir, "")
            # print(_name, location)
            if _name == location:
                new_data_list.append(sample.copy())
        self.data_list = new_data_list


class UBFC(BaseDataset):
    def __init__(self, data_dir=ubfc_dir, **kwargs):
        super(UBFC, self).__init__(data_dir, **kwargs)

    def get_data_list(self):
        self.name = "UBFC"
        subject_list = os.listdir(self.data_dir)
        subject_list.remove('subject11')    # exist error hr (eq 0) in sample
        subject_list.remove('subject18')    # exist error hr (eq 0) in sample
        subject_list.remove('subject20')    # exist error hr (eq 0) in sample
        subject_list.remove('subject24')    # exist error hr (eq 0) in sample
        subject_list.sort()
        if self.train:
            subject_list = subject_list[:30]
        else:
            subject_list = subject_list[30:]
        for subject in subject_list:
            file_dir = os.path.join(self.data_dir, subject)
            with h5py.File(os.path.join(file_dir, 'sample.hdf5'), 'r') as f:
                video_length = f['video_data'].shape[1]   # C, T, H, W
            sample_num = video_length // self.T if self.T != -1 else 1
            for i in range(sample_num):
                sample = {}
                sample["location"] = file_dir
                sample["start_idx"] = i * self.T
                sample["video_length"] = video_length
                self.data_list.append(sample)

class PURE(BaseDataset):
    def __init__(self, data_dir=pure_dir, **kwargs):
        super().__init__(data_dir, **kwargs)

    def get_data_list(self):
        self.name = "PURE"
        date_list = os.listdir(self.data_dir)
        date_list.sort()
        train_list = ['06-01', '06-03', '06-04', '06-05', '06-06', '08-01', '08-02', '08-03', '08-04', '08-05', '08-06',\
                    '05-01', '05-02', '05-03', '05-04', '05-05', '05-06', '01-01', '01-02', '01-03', '01-04', '01-05', '01-06',\
                    '04-01', '04-02', '04-03', '04-04', '04-05', '04-06', '09-01', '09-02', '09-03', '09-04', '09-05', '09-06',\
                    '07-01', '07-02', '07-03', '07-04', '07-05', '07-06']
        if self.train:
            date_list = [i for i in date_list if i in train_list]
        else:
            date_list = [i for i in date_list if i not in train_list]
        for date in date_list:
            sample_dir = os.path.join(self.data_dir, date)
            with h5py.File(os.path.join(sample_dir, 'sample.hdf5'), 'r') as f:
                video_length = f['video_data'].shape[1]   # C, T, H, W
            sample_num = video_length // self.T if self.T != -1 else 1
            for i in range(sample_num):
                sample = {}
                sample["location"] = sample_dir
                sample["start_idx"] = i * self.T
                sample["video_length"] = video_length
                self.data_list.append(sample)


class VIPL(BaseDataset):
    def __init__(self, data_dir=vipl_dir, **kwargs):
        super().__init__(data_dir, **kwargs)

    def get_data_list(self):
        self.name = "VIPL"
        self.fold_split_dir = os.path.join(self.data_dir, "VIPL_fold")
        self.fold_list = []
        for i in range(1, 6):
            mat_path = os.path.join(self.fold_split_dir, f"fold{i}.mat")
            mat = sio.loadmat(mat_path)
            self.fold_list.append(mat[f"fold{i}"].reshape(-1))

        if self.train:
            # all flod except self.fold
            fold = np.concatenate(self.fold_list[:self.fold - 1] + self.fold_list[self.fold:])
        else:
            fold = self.fold_list[self.fold - 1]

        # print(fold)
        p_lists = [f"p{i}" for i in fold]
        p_lists.sort()
        for p_name in p_lists:
            p_root = os.path.join(self.data_dir, p_name)
            v_lists = os.listdir(p_root)
            v_lists.sort()
            for v_name in v_lists:
                v_root = os.path.join(p_root, v_name)
                source_lists = os.listdir(v_root)
                if "source4" in source_lists:
                    source_lists.remove("source4")
                source_lists.sort()
                for source_name in source_lists:
                    if os.path.join(v_root, source_name) in [f'{self.data_dir}/p32/v7/source3', f'{self.data_dir}/p45/v1/source2', \
                                                            f'{self.data_dir}/p19/v2/source2']: # , f'{self.data_dir}/p43/v5/source2']: # 32-7-3, 45-1-2 lack of wave, 19-2-2 lack of frame
                        continue
                    with h5py.File(os.path.join(v_root, source_name, 'sample.hdf5'), 'r') as f:
                        video_length = f['video_data'].shape[1]   # C, T, H, W
                    sample_num = video_length // self.T if self.T != -1 else 1
                    for i in range(sample_num):
                        sample = {}
                        sample["location"] = os.path.join(v_root, source_name)
                        sample["start_idx"] = i * self.T
                        sample["video_length"] = video_length
                        self.data_list.append(sample)

class COHFACE(BaseDataset):
    def __init__(self, data_dir=cohface_dir, **kwargs):
        super().__init__(data_dir, **kwargs)

    def get_data_list(self):
        self.name = "COHFACE"
        date_list = os.listdir(self.data_dir)
        try:
            date_list.remove("protocols")
            # date_list.remove(".README.rst.swp")
            date_list.remove("README.rst")
        except:
            pass
        date_list.sort()
        if self.train:
            date_list = [subject for subject in date_list if int(subject) not in [25, 29, 31 ,35, 27 ,33 ,1 ,16, 38, 21, 28, 4]]
        else:
            date_list = [str(subject) for subject in [25, 29, 31 ,35, 27 ,33 ,1 ,16, 38, 21, 28, 4]]
        for i in range(len(date_list)):
            i_root = os.path.join(self.data_dir, date_list[i])
            for vd in range(4):
                sample_dir = os.path.join(i_root, str(vd))
                try:
                    with h5py.File(os.path.join(sample_dir, 'sample.hdf5'), 'r') as f:
                        video_length = f['video_data'].shape[1]   # C, T, H, W
                except:
                    continue
                sample_num = video_length // self.T if self.T != -1 else 1
                for i in range(sample_num):
                    sample = {}
                    sample["location"] = sample_dir
                    sample["start_idx"] = i * self.T
                    sample["video_length"] = video_length
                    self.data_list.append(sample)


from scipy.stats import zscore

def remove_outliers(signal, threshold=2.0):
    """
    去除一维信号中的离群值。

    参数:
    - signal: 一维信号的NumPy数组。
    - threshold: 离群值判定阈值，通常为标准差的倍数。

    返回:
    - 清理后的信号，不包含离群值。
    """
    z_scores = zscore(signal)
    outliers = np.abs(z_scores) > threshold

    cleaned_signal = np.where(outliers, np.mean(signal), signal)

    return cleaned_signal
from scipy.signal import savgol_filter


def smooth_signal(signal, window_size=51, order=2):
    """
    对一维信号进行平滑处理。

    参数:
    - signal: 一维信号的NumPy数组。
    - window_size: 平滑窗口的大小，必须为奇数。
    - order: 多项式阶数。

    返回:
    - 平滑后的信号。
    """
    smoothed_signal = savgol_filter(signal, window_size, order)
    return smoothed_signal


class MMSEHR(BaseDataset):
    def __init__(self, data_dir=mmsehr_dir, **kwargs):
        super().__init__(data_dir, **kwargs)

    def __getitem__(self, index):
        """return: video, ecg, transform_rate, frame_start, frame_end"""
        sample = self.data_list[index]
        start_idx = sample['start_idx']
        video_length = sample['video_length']
        seg_video, shading_video = None, None
        if start_idx + int(self.T * 1.5) > video_length:
            with h5py.File(os.path.join(sample['location'], 'sample.hdf5'), 'r') as f:
                video_x = np.array(f['video_data'][:, start_idx: start_idx + self.T]).transpose(1, 2, 3, 0) # T, H, W, C
                ecg = np.array(f['ecg_data'][start_idx: start_idx + self.T]) # T
        else:
            with h5py.File(os.path.join(sample['location'], 'sample.hdf5'), 'r') as f:
                video_x = np.array(f['video_data'][:, start_idx: start_idx + int(self.T * 1.5)]).transpose(1, 2, 3, 0) # T, H, W, C
                ecg = np.array(f['ecg_data'][start_idx: start_idx + int(self.T * 1.5)]) # T

        video_x_aug = video_x.transpose(3, 0, 1, 2) # (T, H, W, C) -> (C, T, H, W)
        # ecg_remove = remove_outliers(ecg, threshold=1.2)
        # wave = smooth_signal(ecg_remove)
        wave = ecg
        wave = torch.from_numpy(wave).float()

        if self.w != 64 or self.h != 64:
            video_x_aug = video_x_aug.transpose(1,0)
            video_x_aug = resize(video_x_aug, [self.w, self.h], interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
            video_x_aug = video_x_aug.transpose(1,0)

        sample_item = {}
        sample_item["location"] = sample['location'].replace(self.data_dir, "")
        sample_item['video'] = video_x_aug
        sample_item['ecg'] = wave
        sample_item['clip_avg_hr'] = cal_hr(wave, 30)

        return sample_item

    def get_data_list(self):
        self.name = "MMSEHR"
        person_list = os.listdir(self.data_dir)
        try:
            person_list.remove("white_name.txt")
        except:
            pass
        person_list.sort()
        for i in range(len(person_list)):
            person_root = os.path.join(self.data_dir, person_list[i])
            task_list = os.listdir(person_root)
            task_list.sort()
            for task in task_list:
                sample_dir = os.path.join(person_root, task)

                if task in ["/F006/T11"]:
                    continue

                try:
                    with h5py.File(os.path.join(sample_dir, 'sample.hdf5'), 'r') as f:
                        video_length = f['video_data'].shape[1]   # C, T, H, W
                except:
                    continue
                sample_num = video_length // self.T if self.T != -1 else 1
                for i in range(sample_num):
                    sample = {}
                    sample["location"] = sample_dir
                    sample["start_idx"] = i * self.T
                    sample["video_length"] = video_length
                    self.data_list.append(sample)


if __name__ == "__main__":
    dataset = VIPL(data_dir="/data/wuruize/VIPL", train=True, norm_type="reconstruct", T=160, w=64, h=64, aug="", fold=1)
    for i in range(len(dataset)):
        sample = dataset[i]
        print(i,sample['video'].shape, sample['ecg'].shape, sample['clip_avg_hr'], sample['seg'].shape, sample['shading'].shape)
        break