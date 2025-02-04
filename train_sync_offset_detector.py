import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, median_absolute_error
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import argparse

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


# Set up command line argument parsing
parser = argparse.ArgumentParser(description='Train sync offset detector')
parser.add_argument('--prefix', type=str, required=True,
                    help='Prefix for input/output files (e.g., "ntu")')
args = parser.parse_args()

prefix = args.prefix

# Load prepared data
X_train = np.load(
    f'{prefix}_train_softmaxed_sim_12.npy', allow_pickle=True)
y_train = np.load(
    f'{prefix}_train_softmaxed_sim_12_labels.npy', allow_pickle=True)
X_val = np.load(
    f'{prefix}_val_softmaxed_sim_12.npy', allow_pickle=True)
y_val = np.load(
    f'{prefix}_val_softmaxed_sim_12_labels.npy', allow_pickle=True)

print(f'Shape of X_train: {X_train.shape}, Shape of y_train: {y_train.shape}')
print(f'Shape of X_val: {X_val.shape}, Shape of y_val: {y_val.shape}')

# Pad matrices to target_size*target_size
X_train_padded = pad_matrices(X_train, target_size=256)
X_val_padded = pad_matrices(X_val, target_size=256)

print('after padding X matrices')
print(
    f'Shape of X_train_padded: {X_train_padded.shape}, Shape of X_val_padded: {X_val_padded.shape}')
print(f'Shape of y_train: {y_train.shape}, Shape of y_val: {y_val.shape}')

# Define the range of classes explicitly
min_offset = -30
max_offset = 30
n_classes = max_offset - min_offset + 1  # 61 classes
all_possible_classes = np.arange(min_offset, max_offset + 1)

# Initialize label encoder with all possible classes
label_encoder = LabelEncoder()
# Fit with shifted values (0 to 60)
label_encoder.fit(all_possible_classes + 30)

# Convert y values to class labels (0 to 60)
y_train_shifted = y_train + 30
y_val_shifted = y_val + 30
y_train_encoded = label_encoder.transform(y_train_shifted)
y_val_encoded = label_encoder.transform(y_val_shifted)

# Configuration dictionary
config = {
    'train_log_reg': False,
    'train_svm': False,
    'train_mlp': True,
    'mlp_config': {
        'hidden_layer_sizes': (2048, 1024, 512),
        'max_iter': 1000,
        'activation': 'relu',
        'solver': 'adam',
        'random_state': 42,
        'learning_rate_init': 0.001,
        'batch_size': 'auto',
        'early_stopping': True,
        'validation_fraction': 0.2,
        'n_iter_no_change': 20,
        'tol': 1e-4,
    }
}

# Print configuration and prefix for logging
print("\nRunning with configuration:")
print(f"Prefix: {prefix}")
print(json.dumps(config, indent=4))
print("\n")

# Initialize models based on config
models = {}
if config['train_log_reg']:
    models['log_reg'] = LogisticRegression(
        n_jobs=-1, verbose=True, max_iter=1000)

if config['train_svm']:
    models['svm'] = SVR()

if config['train_mlp']:
    models['mlp'] = MLPClassifier(
        verbose=True,
        **config['mlp_config']
    )

# Train and evaluate models
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")

    # Handle MLP differently due to label encoding
    if name == 'mlp':
        model.fit(X_train_padded, y_train_encoded)
        y_pred = model.predict(X_val_padded)
        y_pred = y_pred - 30  # Convert back to original scale
    else:
        model.fit(X_train_padded, y_train)
        y_pred = model.predict(X_val_padded)

    # Calculate metrics
    mae = mean_absolute_error(y_val, y_pred)
    medae = median_absolute_error(y_val, y_pred)
    results[name] = {'mae': mae, 'medae': medae}

    print(f'{name.upper()} - MAE: {mae}, MedAE: {medae}')

    # Save model
    with open(f'{prefix}_{name}_model.pkl', 'wb') as file:
        pickle.dump(model, file)

# Calculate baseline using median approach
baseline_offsets = []
for i, softmaxed_sim_12 in enumerate(X_val):
    # print(f'Shape of softmaxed_sim_12: {softmaxed_sim_12.shape}')
    median_offset = calculate_median_offset(softmaxed_sim_12)
    baseline_offsets.append(median_offset)

baseline_mae = mean_absolute_error(y_val, baseline_offsets)
baseline_medae = median_absolute_error(y_val, baseline_offsets)

print(f'Baseline (Median) - MAE: {baseline_mae}, MedAE: {baseline_medae}')
