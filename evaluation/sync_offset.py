# coding=utf-8
import torch
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import kendalltau
import utils.logging as logging
from torch.nn import functional as Fun
import math


logger = logging.get_logger(__name__)


def softmax(w, t=1.0):
    e = np.exp(np.array(w) / t)
    return e / np.sum(e)


class SyncOffset(object):
    """Calculate Synchronization offset."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.downstream_task = False

    def evaluate(self, model, train_loader, val_loader, cur_epoch, summary_writer):
        model.eval()

        abs_frame_errors = []
        with torch.no_grad():
            for videos, labels, seq_lens, chosen_steps, video_masks, names in val_loader:
                embs = []
                for i in [0, 1]:
                    video = videos[i]
                    seq_len = seq_lens[i]
                    chosen_step = chosen_steps[i]
                    video_mask = video_masks[i]
                    name = names[i]

                    emb = self.get_embs(model, video, labels[i], seq_len, chosen_step, video_mask, name)                
                    embs.append(emb)

                abs_frame_error = self.get_sync_offset(embs[0], labels[0], embs[1], labels[1])
                abs_frame_errors.append(abs_frame_error)

                print('names', names, 'labels', labels, 'abs_frame_error', abs_frame_error)

        mean_abs_frame_error = np.mean(abs_frame_errors)
        std_dev = np.std(abs_frame_errors)

        logger.info('epoch[{}/{}] mean abs frame error: {:.4f}'.format(
            cur_epoch, self.cfg.TRAIN.MAX_EPOCHS, mean_abs_frame_error))
        logger.info('epoch[{}/{}] std dev: {:.4f}'.format(
            cur_epoch, self.cfg.TRAIN.MAX_EPOCHS, std_dev))
        logger.info('epoch[{}/{}] len(abs_frame_errors): {}'.format(
            cur_epoch, self.cfg.TRAIN.MAX_EPOCHS, len(abs_frame_errors)))


    def get_sync_offset(self, embs0, label0, embs1, label1):
        return decision_offset(torch.tensor(embs0).cuda(), torch.tensor(embs1).cuda(), label0 - label1)


    def get_embs(self, model, video, frame_label, seq_len, chosen_steps, video_masks, name):
        logger.info(f'name: {name}, video.shape: {video.shape}, frame_label: {frame_label}, seq_len: {seq_len}, chosen_steps.shape: {chosen_steps.shape}, video_masks.shape: {video_masks.shape}')

        assert video.size(0) == 1  # batch_size==1
        assert video.size(1) == int(seq_len.item())

        embs = []
        seq_len = seq_len.item()
        num_batches = 1
        frames_per_batch = int(math.ceil(float(seq_len)/num_batches))

        num_steps = seq_len
        steps = torch.arange(0, num_steps)
        curr_data = video[:, steps]

        if self.cfg.USE_AMP:
            with torch.cuda.amp.autocast():
                emb_feats = model(curr_data, num_steps)
        else:
            emb_feats = model(curr_data, num_steps)
        embs.append(emb_feats[0].cpu())
        embs = torch.cat(embs, dim=0)
        embs = embs.numpy()

        return embs


def get_similarity(view1, view2):
    norm1 = torch.sum(torch.square(view1), dim=1)
    norm1 = norm1.reshape(-1, 1)
    norm2 = torch.sum(torch.square(view2), dim=1)
    norm2 = norm2.reshape(1, -1)
    similarity = norm1 + norm2 - 2.0 * \
        torch.matmul(view1, view2.transpose(1, 0))
    similarity = -1.0 * torch.max(similarity, torch.zeros(1).cuda())

    return similarity


def decision_offset(view1, view2, label):
    sim_12 = get_similarity(view1, view2)

    print('sim_12', sim_12)

    softmaxed_sim_12 = Fun.softmax(sim_12, dim=1)

    print('view1.shape', view1.shape)
    print('view2.shape', view2.shape)
    
    print('softmaxed_sim_12.shape', softmaxed_sim_12.shape)

    print('softmaxed_sim_12', softmaxed_sim_12)

    ground = (torch.tensor([i * 1.0 for i in range(view1.size(0))]).cuda()).reshape(-1, 1)

    predict = softmaxed_sim_12.argmax(dim=1)

    print('predict', predict)

    length1 = ground.size(0)

    frames = []

    for i in range(length1):
        p = predict[i].item()
        g = ground[i][0].item()

        frame_error = (p - g)
        frames.append(frame_error)

    # median_frames = np.median(frames)

    # num_frames = math.floor(median_frames)

    print('frames', frames)
    best_subsequence, moe, subsequence_mean = find_optimal_subsequence(frames)
    print(best_subsequence, moe, subsequence_mean)

    num_frames = best_subsequence[1] if best_subsequence[0] == 0 else -1 * best_subsequence[0]
    print('label', label, 'median', str(math.floor(np.median(frames))), 'subsequence_mean', subsequence_mean, 'num_frames', num_frames)

    result = abs(subsequence_mean - label)

    return result

from scipy.stats import t

class SubsequenceStats:
    """
    To efficiently maintain and update statistics like mean and standard deviation for a dynamically changing subsequence, we'll create a Python class that handles the addition and removal of elements. For this purpose, we'll use Welford's method for updating the mean and standard deviation in a single pass, which is more numerically stable than the naive method.
    """
    def __init__(self):
        self.n = 0
        self.mean = 0
        self.M2 = 0

    def add(self, x):
        """ Add a new element x to the subsequence. """
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def remove(self, x):
        """ Remove an element x from the subsequence. """
        if self.n == 0:
            raise ValueError("Cannot remove from an empty subsequence")
        self.n -= 1
        delta = x - self.mean
        self.mean -= delta / (self.n if self.n > 0 else 1)
        self.M2 -= delta * (x - self.mean)

        if self.n < 1:  # Reset to initial state if no more elements are present
            self.n = 0
            self.mean = 0
            self.M2 = 0

    @property
    def stddev(self):
        """ Return the standard deviation of the subsequence. """
        if self.n < 2:
            return 0
        return (self.M2 / self.n) ** 0.5

    def get_stats(self):
        """ Return the mean and standard deviation of the subsequence. """
        return self.mean, self.stddev

    def margin_of_error(self, confidence_level=0.90):
        if self.n < 2:
            return float('inf')
        t_value = t.ppf((1 + confidence_level) / 2, self.n - 1)
        return t_value * self.stddev / math.sqrt(self.n)


def find_optimal_subsequence(data, ci=0.9, epsilon=5):
    """
      looking for subsequences for which the length is within the confidence interval from the mean
    """

    n = len(data)

    best_margin = float('inf')
    best_subsequence = (0, 0)
    best_mean = float('inf')

    start = 0
    substat = SubsequenceStats()

    for end in range(start, len(data)):
        substat.add(data[end])
        mean = substat.mean
        me = substat.margin_of_error(ci)
        if np.abs(end-start - (n-mean)) <= me + epsilon:
            if me <= best_margin:
                best_margin = me
                best_subsequence = (start, end)
                best_mean = round(mean)

    end = len(data)

    for start in range(len(data)):
        substat.remove(data[start])
        mean = substat.mean
        me = substat.margin_of_error(ci)
        if np.abs(end-start - (n+mean)) <= me + epsilon:
            if me <= best_margin:
                best_margin = me
                best_subsequence = (start, end)
                best_mean = round(mean)

    return best_subsequence, best_margin, best_mean