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
        self.downstream_task = True

    def evaluate(self, dataset, cur_epoch, summary_writer):
        """Labeled evaluation."""

        train_embs = dataset['train_dataset']['embs']
        train_names = dataset['train_dataset']['names']
        train_labels = dataset['train_dataset']['labels']

        self.get_sync_offset(
            train_embs,
            train_names,
            train_labels,
            cur_epoch, summary_writer,
            '%s_train' % dataset['name'])

        val_embs = dataset['val_dataset']['embs']
        val_names = dataset['val_dataset']['names']
        val_labels = dataset['val_dataset']['labels']

        sync_offset = self.get_sync_offset(
            val_embs, val_names, val_labels, cur_epoch, summary_writer, '%s_val' % dataset['name'])
        return sync_offset

    def get_sync_offset(self, embs_list, names, labels, cur_epoch, summary_writer, split):
        num_seqs = len(embs_list)

        name_to_idx = {}
        for i in range(num_seqs):
            name = names[i]
            name_to_idx[name] = i

        num_pairs = int(num_seqs / 2)
        frame_errors = np.zeros(num_pairs)
        idx = 0

        frame_error = 0
        for i in range(num_seqs):
            query_feats = embs_list[i]
            name = names[i]
            label = labels[i][0]

            camera = name[5:8]
            if camera != '001':
                continue

            # Get the other angle's filename.
            candidate_name = f'{name[:5]}002{name[8:]}'
            candidate_i = name_to_idx[candidate_name]
            candidate_feats = embs_list[candidate_i]
            candidate_label = labels[candidate_i][0]

            print('name', name, 'candidate_name', candidate_name)
            print('query_feats.shape', query_feats.shape,
                  'candiate_feats.shape', candidate_feats.shape)
            print('label', str(label), 'candidate_label', str(
                candidate_label), 'label - candidate_label', str(label - candidate_label))

            frame_error = decision_offset(torch.tensor(query_feats).cuda(
            ), torch.tensor(candidate_feats).cuda(), label - candidate_label)
            print('frame error', frame_error)

            frame_errors[idx] = frame_error
            idx += 1

        avg_frame_error = np.mean(frame_errors)
        std_dev = np.std(frame_errors)

        logger.info('epoch[{}/{}] {} set avg frame error: {:.4f}'.format(
            cur_epoch, self.cfg.TRAIN.MAX_EPOCHS, split, avg_frame_error))
        logger.info('epoch[{}/{}] {} set std dev: {:.4f}'.format(
            cur_epoch, self.cfg.TRAIN.MAX_EPOCHS, split, std_dev))

        return avg_frame_error


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

    softmaxed_sim_12 = Fun.softmax(sim_12, dim=1)

    ground = (torch.tensor(
        [i * 1.0 for i in range(view1.size(0))]).cuda()).reshape(-1, 1)

    predict = softmaxed_sim_12.argmax(dim=1)

    length1 = ground.size(0)

    frames = []

    for i in range(length1):
        p = predict[i].item()
        g = ground[i][0].item()

        frame_error = (p - g)
        frames.append(frame_error)

    median_frames = np.median(frames)

    num_frames = math.floor(median_frames)

    result = abs(num_frames - label)

    return result
