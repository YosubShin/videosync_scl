# coding=utf-8
import os
import math
import cv2
from tqdm import tqdm
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import read_video

import utils.logging as logging
from datasets.data_augment import create_data_augment, create_ssl_data_augment

logger = logging.get_logger(__name__)


class Ntu(torch.utils.data.Dataset):
    def __init__(self, cfg, split, mode="auto", sample_all=False, dataset=None):
        assert split in ["train", "val"]
        self.cfg = cfg
        self.split = split
        if mode == "auto":
            self.mode = "train" if self.split == "train" else "eval"
        else:
            self.mode = mode
        self.sample_all = sample_all
        self.num_contexts = cfg.DATA.NUM_CONTEXTS
        self.train_dataset = os.path.join(
            cfg.PATH_TO_DATASET, f"train.pkl")
        self.val_dataset = os.path.join(
            cfg.PATH_TO_DATASET, f"val.pkl")

        if self.split == "train":
            with open(self.train_dataset, 'rb') as f:
                dataset = pickle.load(f)
        else:
            with open(self.val_dataset, 'rb') as f:
                dataset = pickle.load(f)

        self.dataset = []
        self.error_videos = []
        self.dataset = dataset

        if self.cfg.SSL and self.mode == "train":
            self.data_preprocess = create_ssl_data_augment(cfg, augment=True)
        elif self.mode == "train":
            self.data_preprocess = create_data_augment(cfg, augment=True)
        else:
            self.data_preprocess = create_data_augment(cfg, augment=False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        videos = []
        labels = []
        seq_lens = []
        chosen_steps = []
        video_masks = []
        names = []

        for i, camera_id in enumerate(['001', '002']):
            name = self.dataset[index][f"video_file_{i}"].split("/")[-1].split(".")[0]
            video_file = os.path.join(
                self.cfg.PATH_TO_DATASET, self.dataset[index][f"video_file_{i}"])
            video, _, info = read_video(video_file, pts_unit='sec')
            seq_len = len(video)
            if seq_len == 0:
                print('seq_len is 0', video_file)
            # T H W C -> T C H W, [0,1] tensor
            video = video.permute(0, 3, 1, 2).float() / 255.0

            steps = torch.arange(0, seq_len, self.cfg.DATA.SAMPLE_ALL_STRIDE)
            video = video[steps.long()]
            video = self.data_preprocess(video)
            label = torch.full((1, ), self.dataset[index][f"label_{i}"], dtype=torch.int32)
            seq_len = len(steps)
            chosen_step = steps.clone()
            video_mask = torch.ones(seq_len)

            videos.append(video)
            labels.append(label)
            seq_lens.append(seq_len)
            chosen_steps.append(chosen_step)
            video_masks.append(video_mask)
            names.append(name)

        # videos = torch.stack(videos, dim=0)
        # labels = torch.stack(labels, dim=0)
        # seq_lens = torch.tensor(seq_lens)
        # chosen_steps = torch.stack(chosen_steps, dim=0)
        # video_masks = torch.stack(video_masks, dim=0)

        return videos, labels, seq_lens, chosen_steps, video_masks, names
