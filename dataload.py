import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.io import read_video
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import os
import numpy as np

import os
import random
from tqdm import tqdm
import torchvision.io as io
from torch.utils.data import Dataset


# Load 16 frames from each video
# class KineticsDataset(Dataset):
#     def __init__(self, directory, transform=None):
#         super(KineticsDataset, self).__init__()
#         self.directory = directory
#         self.transform = transform
#         self.videos = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.mp4', '.avi'))]

#     def __len__(self):
#         return len(self.videos)

#     def __getitem__(self, idx):
#         video_path = self.videos[idx]
#         # Read video and audio, torchvision reads videos as (T, H, W, C)
#         video, _, _ = io.read_video(video_path, pts_unit='sec')
        
#         total_frames = video.shape[0]
#         target_frames = 16  # We want exactly 16 frames
        
#         # Handling videos shorter than 16 frames
#         if total_frames < target_frames:
#             repeat, remainder = divmod(target_frames, total_frames)
#             video = video.repeat(repeat, 1, 1, 1)
#             video = torch.cat((video, video[:remainder]), dim=0)
#         elif total_frames > target_frames:
#             # If the video is longer, select 16 frames evenly spaced from the video
#             indices = torch.linspace(0, total_frames - 1, target_frames).long()
#             video = video[indices]
        
#         video = video.permute(0, 3, 1, 2)  # Change shape to (T, C, H, W)

#         if self.transform:
#             video = self.transform(video)
        
#         return video

class KineticsDataset(Dataset):
    def __init__(self, directory, transform=None):
        super(KineticsDataset, self).__init__()
        self.directory = directory
        self.transform = transform
        self.videos = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.mp4', '.avi'))]

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path = self.videos[idx]
        # Read video and audio, torchvision reads videos as (T, H, W, C)
        video, _, _ = io.read_video(video_path, pts_unit='sec')
        
        total_frames = video.shape[0]
        target_frames = 16  # We want exactly 16 frames

        # Skip videos with fewer frames than the target
        if total_frames < target_frames:
            return None  # Or raise ValueError if skipping is not desired

        # Handling videos longer than or equal to 16 frames
        if total_frames > target_frames:
            # Select 16 frames evenly spaced from the video
            indices = torch.linspace(0, total_frames - 1, target_frames).long()
            video = video[indices]
        
        video = video.permute(0, 3, 1, 2)  # Change shape to (T, C, H, W)

        if self.transform:
            video = self.transform(video)
        
        return video


# Load every 16 frames from every video
# class KineticsDataset(Dataset):
#     def __init__(self, directory, transform=None, frame_count=16):
#         super(KineticsDataset, self).__init__()
#         self.directory = directory
#         self.transform = transform
#         self.frame_count = frame_count
#         self.data = []
#         self.videos = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.mp4', '.avi'))]
#         self._prepare_dataset()

#     def _prepare_dataset(self):
#         for video_path in self.videos:
#             video, _, _ = io.read_video(video_path, pts_unit='sec')
#             total_frames = video.shape[0]
#             if total_frames >= self.frame_count:
#                 self.data.append(video_path)  # Only add video if it has enough frames

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         video_path, start_frame = self.data[idx]
#         video, _, _ = io.read_video(video_path, pts_unit='sec')
#         video = video[start_frame:start_frame + self.frame_count]  # Extract the window
#         video = video.permute(0, 3, 1, 2)  # Change shape to (T, C, H, W)

#         if self.transform:
#             video = self.transform(video)
        
#         return video


class MovingMNIST(Dataset):
    """`MovingMNIST` Dataset.

    Args:
        root (string): Root directory of dataset where `mnist_test_seq.npy` exists.
        train (bool, optional): If True, creates dataset from the first part of the dataset,
            otherwise from the last part.
        split (int, optional): Train/test split size. Defines how many samples belong to the test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.Resize``
    """
    def __init__(self, root, train=True, split=1000, transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.split = split
        self.train = train  # training set or test set

        data_path = os.path.join(self.root, 'mnist_test_seq.npy')
        if not os.path.exists(data_path):
            raise RuntimeError('Dataset not found. Please place mnist_test_seq.npy in the directory')

        # Loading data: shape (20, 10000, 64, 64)
        data = np.load(data_path)
        data = torch.from_numpy(data).float()  # shape becomes (20, 10000, 64, 64)
        
        # Swap axes to make it (10000, 20, 64, 64)
        data = data.permute(1, 0, 2, 3)

        if self.train:
            self.data = data[:-self.split]
        else:
            self.data = data[-self.split:]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (seq, target) where target is the next frame in sequence
        """
        seq = self.data[index, :16]  # Get 16 frames

        if self.transform is not None:
            seq = torch.stack([self.transform(Image.fromarray(frame.numpy(), mode='L')) for frame in seq])

        return seq

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train else 'test'
        fmt_str += '    Train/test: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        return fmt_str
