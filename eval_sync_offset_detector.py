import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, median_absolute_error
import torch.nn.functional as F
import pickle

# Function to calculate the baseline sync offset using the median approach


def calculate_median_offset(softmaxed_sim_12):
    predict = softmaxed_sim_12.argmax(axis=1)
    ground = np.arange(softmaxed_sim_12.shape[0])

    frames = predict - ground
    median_frames = np.median(frames)
    return median_frames


def pad_matrices(matrices, target_size):
    padded_matrices = []
    for matrix in matrices:
        original_height, original_width = matrix.shape
        pad_height = max(0, target_size - original_height)
        pad_width = max(0, target_size - original_width)

        # Pad and truncate as necessary
        padded_matrix = np.pad(
            matrix, ((0, pad_height), (0, pad_width)), mode='constant')
        padded_matrix = padded_matrix[:target_size,
                                      :target_size]  # Truncate if necessary

        # Print shapes for debugging
        # print(
        #     f'Original matrix shape: {matrix.shape}, Padded matrix shape: {padded_matrix.shape}')

        # Flatten for logistic regression
        padded_matrices.append(padded_matrix.flatten())
        # print(f'Flattened padded matrix shape: {padded_matrices[-1].shape}')

    return np.vstack(padded_matrices)


if __name__ == "__main__":
    # Load prepared data
    # X = np.load(
    #     '/home/yosubs/koa_scratch/tmp/scl_transformer_ntu_logs/val_softmaxed_sim_12.npy', allow_pickle=True)
    # y = np.load('/home/yosubs/koa_scratch/tmp/scl_transformer_ntu_logs/val_softmaxed_sim_12_labels.npy', allow_pickle=True)
    X = np.load(
        '/home/yosubs/videosync_scl/human_pose_ntu_softmaxed_sim_12.npy', allow_pickle=True)
    y = np.load(
        '/home/yosubs/videosync_scl/human_pose_ntu_softmaxed_sim_12_labels.npy', allow_pickle=True)

    print(f'Shape of X: {X.shape}, Shape of y: {y.shape}')

    # Pad matrices to target_size*target_size
    X_padded = pad_matrices(X, target_size=256)
    print(f'Shape of X_padded: {X_padded.shape}')

    X_val = X_padded
    y_val = y

    print(f'Shape of X_val: {X_val.shape}')
    print(f'Shape of y_val: {y_val.shape}')

    with open('logistic_regression_model.pkl', 'rb') as file:
        log_reg = pickle.load(file)

    # Evaluate models
    y_pred_log_reg = log_reg.predict(X_val)

    mae_log_reg = mean_absolute_error(y_val, y_pred_log_reg)
    medae_log_reg = median_absolute_error(y_val, y_pred_log_reg)

    print(f'Logistic Regression - MAE: {mae_log_reg}, MedAE: {medae_log_reg}')

    # Calculate baseline using median approach
    baseline_offsets = []
    for i, softmaxed_sim_12 in enumerate(X):
        # print(f'Shape of softmaxed_sim_12: {softmaxed_sim_12.shape}')
        median_offset = calculate_median_offset(softmaxed_sim_12)
        baseline_offsets.append(median_offset)

    baseline_mae = mean_absolute_error(y, baseline_offsets)
    baseline_medae = median_absolute_error(y, baseline_offsets)

    print(f'Baseline (Median) - MAE: {baseline_mae}, MedAE: {baseline_medae}')
