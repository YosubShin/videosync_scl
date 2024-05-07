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
        if dataset is None:
            for data in tqdm(dataset, total=len(dataset)):
                try:
                    video_file = os.path.join(
                        self.cfg.PATH_TO_DATASET, data["video_file"])
                    video = cv2.VideoCapture(video_file)
                    seq_len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    assert seq_len > 0
                except:
                    self.error_videos.append(data["video_file"])
                else:
                    self.dataset.append(data)
            print(self.error_videos)

            logger.info(
                f"{len(self.dataset)} {self.split} samples of Finegym dataset have been read.")
            seq_lens = [int(data['seq_len']) for data in self.dataset]
            hist, bins = np.histogram(seq_lens, bins='auto')
            print(list(bins.astype(np.int)))
            print(list(hist))
        else:
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
        name = self.dataset[index]["video_file"].split("/")[-1].split(".")[0]
        video_file = os.path.join(
            self.cfg.PATH_TO_DATASET, self.dataset[index]["video_file"])
        video, _, info = read_video(video_file, pts_unit='sec')
        seq_len = len(video)
        if seq_len == 0:
            print(video_file)
        # T H W C -> T C H W, [0,1] tensor
        video = video.permute(0, 3, 1, 2).float() / 255.0
        frame_label = torch.full(
            (seq_len, ), self.dataset[index]['label'], dtype=torch.int32)

        if self.cfg.SSL and not self.sample_all:
            names = [name, name]
            steps_0, chosen_step_0, video_mask0 = self.sample_frames(
                seq_len, seq_len)
            view_0 = self.data_preprocess(video[steps_0.long()])
            label_0 = frame_label[chosen_step_0.long()]
            steps_1, chosen_step_1, video_mask1 = self.sample_frames(
                seq_len, seq_len, pre_steps=steps_0)
            view_1 = self.data_preprocess(video[steps_1.long()])
            label_1 = frame_label[chosen_step_1.long()]
            videos = torch.stack([view_0, view_1], dim=0)
            labels = torch.stack([label_0, label_1], dim=0)
            seq_lens = torch.tensor([seq_len, seq_len])
            chosen_steps = torch.stack([chosen_step_0, chosen_step_1], dim=0)
            video_mask = torch.stack([video_mask0, video_mask1], dim=0)
            return videos, labels, seq_lens, chosen_steps, video_mask, names

        elif not self.sample_all:
            steps, chosen_steps, video_mask = self.sample_frames(
                seq_len, self.num_frames)
        else:
            steps = torch.arange(0, seq_len, self.cfg.DATA.SAMPLE_ALL_STRIDE)
            seq_len = len(steps)
            chosen_steps = steps.clone()
            video_mask = torch.ones(seq_len)

        # Select data based on steps
        video = video[steps.long()]
        video = self.data_preprocess(video)
        if self.cfg.DATA.FRAME_LABELS:
            label = frame_label[chosen_steps.long()]

        return video, label, torch.tensor(seq_len), chosen_steps, video_mask, name

    def sample_frames(self, seq_len, num_frames, pre_steps=None):
        # When dealing with very long videos we can choose to sub-sample to fit
        # data in memory. But be aware this also evaluates over a subset of frames.
        # Subsampling the validation set videos when reporting performance is not
        # recommended.
        sampling_strategy = self.cfg.DATA.SAMPLING_STRATEGY
        pre_offset = min(pre_steps) if pre_steps is not None else None

        if sampling_strategy == 'offset_uniform':
            # Sample a random offset less than a provided max offset. Among all frames
            # higher than the chosen offset, randomly sample num_frames
            if seq_len >= num_frames:
                # Returns a random permutation of integers from 0 to n - 1.
                steps = torch.randperm(seq_len)
                steps = torch.sort(steps[:num_frames])[0]
            else:
                steps = torch.arange(0, num_frames)
        elif sampling_strategy == 'time_augment':
            num_valid = min(seq_len, num_frames)
            expand_ratio = np.random.uniform(
                low=1.0, high=self.cfg.DATA.SAMPLING_REGION) if self.cfg.DATA.SAMPLING_REGION > 1 else 1.0

            block_size = math.ceil(expand_ratio*seq_len)
            if pre_steps is not None and self.cfg.DATA.CONSISTENT_OFFSET != 0:
                shift = int((1-self.cfg.DATA.CONSISTENT_OFFSET)*num_valid)
                offset = np.random.randint(low=max(0, min(
                    seq_len-block_size, pre_offset-shift)), high=max(1, min(seq_len-block_size+1, pre_offset+shift+1)))
            else:
                offset = np.random.randint(
                    low=0, high=max(seq_len-block_size, 1))
            steps = offset + torch.randperm(block_size)[:num_valid]
            steps = torch.sort(steps)[0]
            if num_valid < num_frames:
                steps = F.pad(steps, (0, num_frames-num_valid),
                              "constant", seq_len)
        else:
            raise ValueError('Sampling strategy %s is unknown. Supported values are '
                             'stride, offset_uniform .' % sampling_strategy)

        video_mask = torch.ones(num_frames)
        video_mask[steps < 0] = 0
        video_mask[steps >= seq_len] = 0
        # Store chosen indices.
        chosen_steps = torch.clamp(steps.clone(), 0, seq_len - 1)
        if self.num_contexts == 1:
            steps = chosen_steps
        else:
            # Get multiple context steps depending on config at selected steps.
            context_stride = self.cfg.DATA.CONTEXT_STRIDE
            steps = steps.view(-1, 1) + context_stride * \
                torch.arange(-(self.num_contexts-1), 1).view(1, -1)
            steps = torch.clamp(steps.view(-1), 0, seq_len - 1)

        return steps, chosen_steps, video_mask
