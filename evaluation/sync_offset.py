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
            count = 0
            for videos, labels, seq_lens, chosen_steps, video_masks, names in val_loader:
                if count > 100:
                    break

                embs = []
                for i in [0, 1]:
                    video = videos[i]
                    seq_len = seq_lens[i]
                    chosen_step = chosen_steps[i]
                    video_mask = video_masks[i]
                    name = names[i]

                    emb = self.get_embs(
                        model, video, labels[i], seq_len, chosen_step, video_mask, name)
                    embs.append(emb)

                abs_frame_error = self.get_sync_offset(
                    embs[0], labels[0], embs[1], labels[1])
                abs_frame_errors.append(abs_frame_error)

                print('names', names, 'labels', labels,
                      'abs_frame_error', abs_frame_error)
                count += 1

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
        logger.info(
            f'name: {name}, video.shape: {video.shape}, seq_len: {seq_len}')

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
    logger.info(f'view1.shape: {view1.shape}')
    logger.info(f'view2.shape: {view2.shape}')

    sim_12 = get_similarity(view1, view2)

    softmaxed_sim_12 = Fun.softmax(sim_12, dim=1)

    logger.info(f'softmaxed_sim_12.shape: {softmaxed_sim_12.shape}')
    logger.info(f'softmaxed_sim_12: {softmaxed_sim_12}')

    ground = (torch.tensor(
        [i * 1.0 for i in range(view1.size(0))]).cuda()).reshape(-1, 1)

    predict = softmaxed_sim_12.argmax(dim=1)

    logger.info(f'predict: {predict}')

    length1 = ground.size(0)

    frames = []

    for i in range(length1):
        p = predict[i].item()
        g = ground[i][0].item()

        frame_error = (p - g)
        frames.append(frame_error)

    logger.info(f'len(frames): {len(frames)}')
    logger.info(f'frames: {frames}')

    median_frames = np.median(frames)

    num_frames = math.floor(median_frames)

    result = abs(num_frames - label)

    return result
