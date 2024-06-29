# coding=utf-8
import torch
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import kendalltau
import utils.logging as logging
from torch.nn import functional as Fun
import math
import wandb
from scipy.stats import t
from logging import INFO
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import csv


logger = logging.get_logger(__name__)
# logger.disabled = True
logger.setLevel(INFO)

sample_rate = 0.01


def softmax(w, t=1.0):
    e = np.exp(np.array(w) / t)
    return e / np.sum(e)


class SyncOffset(object):
    """Calculate Synchronization offset."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.downstream_task = False
        self.now_str = None
        self.cur_epoch = None
        self.cur_iter = None
        self.sample = None

    def evaluate(self, model, train_loader, val_loader, cur_epoch, summary_writer, sample=False, cur_iter=None):
        model.eval()

        now = datetime.now()
        self.now_str = now.strftime("%Y-%m-%d_%H_%M_%S")
        self.cur_epoch = cur_epoch
        self.cur_iter = cur_iter
        self.sample = sample

        abs_frame_errors_median = []
        abs_frame_errors_mean = []
        frame_errors_median = []
        frame_errors_mean = []

        csv_data = []

        with torch.no_grad():
            count = 0
            for videos, labels, seq_lens, chosen_steps, video_masks, names in val_loader:
                if sample and count / len(val_loader) > sample_rate:
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

                abs_frame_error_dict = self.get_sync_offset(
                    embs[0], labels[0], embs[1], labels[1], names[0][0], names[1][0])

                abs_frame_errors_median.append(
                    abs_frame_error_dict['abs_median'])
                abs_frame_errors_mean.append(
                    abs_frame_error_dict['abs_mean'])
                frame_errors_median.append(
                    abs_frame_error_dict['err_median'])
                frame_errors_mean.append(
                    abs_frame_error_dict['err_mean'])

                csv_data.append([names[0][0], labels[0][0].item(), names[1][0], labels[1][0].item(),
                                abs_frame_error_dict['abs_median'].item(
                ), abs_frame_error_dict['abs_mean'].item(),
                    abs_frame_error_dict['err_median'].item(), abs_frame_error_dict['err_mean'].item()])

                count += 1

        median_abs_frame_error = np.mean(abs_frame_errors_median)
        median_std_dev = np.std(abs_frame_errors_median)
        median_moe = calculate_margin_of_error(abs_frame_errors_median)

        mean_abs_frame_error = np.mean(abs_frame_errors_mean)
        mean_std_dev = np.std(abs_frame_errors_mean)
        mean_moe = calculate_margin_of_error(abs_frame_errors_mean)

        if not sample:
            logger.info('epoch[{}/{}] mean of abs_frame_errors_median: {:.4f}'.format(
                cur_epoch, self.cfg.TRAIN.MAX_EPOCHS, median_abs_frame_error))
            logger.info('epoch[{}/{}] std dev for abs_frame_errors_median: {:.4f}'.format(
                cur_epoch, self.cfg.TRAIN.MAX_EPOCHS, median_std_dev))
            logger.info('epoch[{}/{}] moe(abs_frame_errors_median): {}'.format(
                cur_epoch, self.cfg.TRAIN.MAX_EPOCHS, len(abs_frame_errors_median)))
            logger.info('epoch[{}/{}] len(abs_frame_errors_median): {}'.format(
                cur_epoch, self.cfg.TRAIN.MAX_EPOCHS, len(abs_frame_errors_median)))

            logger.info('epoch[{}/{}] mean of abs_frame_errors_mean: {:.4f}'.format(
                cur_epoch, self.cfg.TRAIN.MAX_EPOCHS, mean_abs_frame_error))
            logger.info('epoch[{}/{}] std dev for abs_frame_errors_mean: {:.4f}'.format(
                cur_epoch, self.cfg.TRAIN.MAX_EPOCHS, mean_std_dev))
            logger.info('epoch[{}/{}] moe(abs_frame_errors_mean): {}'.format(
                cur_epoch, self.cfg.TRAIN.MAX_EPOCHS, len(abs_frame_errors_mean)))
            logger.info('epoch[{}/{}] len(abs_frame_errors_mean): {}'.format(
                cur_epoch, self.cfg.TRAIN.MAX_EPOCHS, len(abs_frame_errors_mean)))

            wandb.log({"median_abs_frame_error": median_abs_frame_error,
                       "median_abs_frame_error_std_dev": median_std_dev,
                       "median_abs_frame_error_moe": median_moe,
                       "mean_abs_frame_error": mean_abs_frame_error,
                       "mean_abs_frame_error_std_dev": mean_std_dev,
                       "mean_abs_frame_error_moe": mean_moe
                       })

        # Write CSV data to file
        csv_filename = f'evaluation_results_epoch_{cur_epoch}_iter_{self.cur_iter}_{self.now_str}.csv'
        csv_filepath = os.path.join(
            self.cfg.LOGDIR, 'eval_logs', csv_filename)

        with open(csv_filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write header
            writer.writerow(['Video 0', 'Label 0', 'Video 1',
                            'Label 1', 'abs_median_error', 'abs_mean_error', 'median_error', 'mean_error'])
            # Write data rows
            writer.writerows(csv_data)

        return {
            'median_abs_frame_error': median_abs_frame_error,
            'median_abs_frame_error_std_dev': median_std_dev,
            "median_abs_frame_error_moe": median_moe,
            'mean_abs_frame_error': mean_abs_frame_error,
            'mean_abs_frame_error_std_dev': mean_std_dev,
            "mean_abs_frame_error_moe": mean_moe
        }

    def get_sync_offset(self, embs0, label0, embs1, label1, name0, name1):
        return decision_offset(self.cfg, torch.tensor(embs0).cuda(), torch.tensor(embs1).cuda(), label0 - label1, name0, name1, self.now_str, self.cur_epoch, self.cur_iter, self.sample)

    def get_embs(self, model, video, frame_label, seq_len, chosen_steps, video_masks, name):
        logger.debug(
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


def decision_offset(cfg, view1, view2, label, name0, name1, now_str, cur_epoch, cur_iter, sample):
    logger.debug(f'view1.shape: {view1.shape}')
    logger.debug(f'view2.shape: {view2.shape}')

    sim_12 = get_similarity(view1, view2)
    softmaxed_sim_12 = Fun.softmax(sim_12, dim=1)

    logger.debug(f'softmaxed_sim_12.shape: {softmaxed_sim_12.shape}')
    logger.debug(f'softmaxed_sim_12: {softmaxed_sim_12}')

    ground = (torch.tensor(
        [i * 1.0 for i in range(view1.size(0))]).cuda()).reshape(-1, 1)

    predict = softmaxed_sim_12.argmax(dim=1)

    # Only plot if we are sampling
    if sample:
        plt.figure(figsize=(10, 8))
        sns.heatmap(softmaxed_sim_12.cpu().numpy(),
                    annot=False, cmap="viridis", cbar=True, square=True)
        plt.plot(predict.cpu(), np.arange(len(predict.cpu())), color='red',
                 marker='o', linestyle='-', linewidth=2, markersize=5)

        k = label.item() * -1
        # Create the points for the line with y-intercept k
        x_line = np.arange(softmaxed_sim_12.shape[1])
        y_line = x_line + k

        valid_indices = (y_line >= 0) & (y_line < softmaxed_sim_12.shape[0])
        x_line = x_line[valid_indices]
        y_line = y_line[valid_indices]

        plt.plot(x_line, y_line, color='blue', linestyle='--',
                 linewidth=2, label=f'Line with y-intercept {k}')

        plt.gca().set_aspect('equal', adjustable='box')

        # Save the heatmap to a PNG file
        plt.title(
            f"{name0}_{name1}_softmaxed_sim_12_epoch_{cur_epoch}_iter_{cur_iter}")
        plt.savefig(os.path.join(
            cfg.LOGDIR, 'eval_logs', f"{name0}_{name1}_softmaxed_sim_12_epoch_{cur_epoch}_iter_{cur_iter}_{now_str}.png"))
        plt.close()

    logger.debug(f'predict: {predict}')

    length1 = ground.size(0)

    frames = []

    for i in range(length1):
        p = predict[i].item()
        g = ground[i][0].item()

        frame_error = (p - g)
        frames.append(frame_error)

    logger.debug(f'len(frames): {len(frames)}')
    logger.debug(f'frames: {frames}')

    median_frames = np.median(frames)
    mean_frames = np.average(frames)

    num_frames_median = math.floor(median_frames)
    num_frames_mean = math.floor(mean_frames)

    abs_median = abs(num_frames_median - label)
    abs_mean = abs(num_frames_mean - label)

    logger.info(
        f'name0: {name0}, name1: {name1}, abs_frame_error (median): {abs_median}, abs_frame_error (mean): {abs_mean}, frame_error (median): {num_frames_median - label}, frame_error (mean): {num_frames_mean - label}')

    return {
        'abs_median': abs_median,
        'abs_mean': abs_mean,
        'err_median': num_frames_median - label,
        'err_mean': num_frames_mean - label,
    }


def calculate_margin_of_error(data, confidence_level=0.95):
    """
    Calculate the margin of error for the mean of a sample data using t-distribution.

    Args:
    data (list): list of sample data.
    confidence_level (float): The confidence level (0 < confidence_level < 1).

    Returns:
    float: The margin of error for the mean of the sample data.
    """
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1")

    # Calculate the sample mean and standard deviation
    std_dev = np.std(data)
    n = len(data)

    # Calculate the degrees of freedom
    df = n - 1

    # Determine the critical t-value for the given confidence level
    # We use two-tailed, hence (1 + confidence_level) / 2
    alpha = (1 - confidence_level) / 2
    t_critical = t.ppf(1 - alpha, df)

    # Calculate the margin of error
    margin_of_error = t_critical * \
        (std_dev / math.sqrt(n))

    return margin_of_error
