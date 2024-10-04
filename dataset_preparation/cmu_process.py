import os
import cv2
import pickle
import itertools
import numpy as np
from tqdm import tqdm
import multiprocessing
import signal
import sys
import random
import subprocess
import logging
from skimage.metrics import structural_similarity as ssim

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Global pool reference to be used in signal handler
pool = None

# Set up logging for retries and errors
logging.basicConfig(filename="video_processing.log", level=logging.DEBUG)


def log_failed_task(video1_output_path, video2_output_path, reason):
    """Log a failed task with a reason for failure."""
    with open(FAILURE_LOG_FILE, 'a') as f:
        f.write(f"{video1_output_path},{video2_output_path},{reason}\n")
    logging.warning(f"Logged failed task: {video1_output_path}, {video2_output_path}, Reason: {reason}")


def is_video_valid(video_path):
    """Check if a video file is valid and has non-zero duration using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=duration",
                "-of",
                "csv=p=0",
                video_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        duration = float(result.stdout.strip())
        return duration > 0
    except Exception as e:
        logging.error(f"ffprobe failed on {video_path}: {e}")
        return False


FAILURE_LOG_FILE = "failed_tasks.log"


def load_failed_tasks():
    if os.path.exists(FAILURE_LOG_FILE):
        with open(FAILURE_LOG_FILE, "r") as f:
            failed_tasks = set(line.strip() for line in f.readlines())
    else:
        failed_tasks = set()
    return failed_tasks


def load_failed_tasks():
    """Load failed tasks and their reasons from the log file."""
    failed_tasks = {}
    if os.path.exists(FAILURE_LOG_FILE):
        with open(FAILURE_LOG_FILE, "r") as f:
            for line in f.readlines():
                video1_output_path, video2_output_path, reason = line.strip().split(",")
                failed_tasks[(video1_output_path, video2_output_path)] = reason
    return failed_tasks


def cleanup_videos(video1_output_path, video2_output_path):
    """Remove corrupted video files."""
    if os.path.exists(video1_output_path):
        os.remove(video1_output_path)
        logging.info(f"Deleted {video1_output_path}")
    if os.path.exists(video2_output_path):
        os.remove(video2_output_path)
        logging.info(f"Deleted {video2_output_path}")


def is_green_screen_frame(frame, green_threshold=0.8):
    """Detect if a frame is mostly green, indicating a green screen."""
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)
    green_ratio = np.sum(green_mask > 0) / (frame.shape[0] * frame.shape[1])
    return green_ratio > green_threshold


def is_black_screen_frame(frame, black_threshold=0.8):
    """Detect if a frame is mostly black, indicating a black screen."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    black_mask = gray_frame < 50  # Pixel intensity threshold for black
    black_ratio = np.sum(black_mask) / (frame.shape[0] * frame.shape[1])
    return black_ratio > black_threshold


def are_frames_similar_ssim(frame1, frame2, similarity_threshold):
    """Compare two frames using Structural Similarity Index (SSIM)."""
    gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray_frame1, gray_frame2, full=True)
    return score > similarity_threshold


def process_video_pair(args, failed_tasks):
    video1_path, video2_path, start_frame, end_frame, offset, video1_output_path, video2_output_path = args

    # Check if this pair has already failed with a specific reason
    if (video1_output_path, video2_output_path) in failed_tasks:
        reason = failed_tasks[(video1_output_path, video2_output_path)]
        logging.info(f"Skipping previously failed task: {video1_output_path}, {video2_output_path}, Reason: {reason}")
        return False

    # Skip processing if output files already exist
    if os.path.exists(video1_output_path) and os.path.exists(video2_output_path):
        logging.info(f"Skipping already existing files: {video1_output_path} and {video2_output_path}")
        return True

    # Process the video pair without retries
    success, reason = sync_and_save_video_pair(
        video1_path, video2_path, start_frame, end_frame, offset, 
        video1_output_path, video2_output_path
    )

    if success:
        return True  # Processing succeeded
    else:
        # Clean up and log the failure reason
        cleanup_videos(video1_output_path, video2_output_path)
        log_failed_task(video1_output_path, video2_output_path, reason)
        logging.error(f"Processing failed for {video1_output_path} and {video2_output_path}. Reason: {reason}")
        return False


def process_event(event_dir, output_dir, event_id, k, max_offset):
    event_name = os.path.basename(event_dir)

    if os.path.exists(os.path.join(event_dir, "hdVideos")):
        video_dir = os.path.join(event_dir, "hdVideos")
    elif os.path.exists(os.path.join(event_dir, "kinectVideos")):
        video_dir = os.path.join(event_dir, "kinectVideos")
    else:
        return [], []

    video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
    video_pairs = list(itertools.combinations(video_files, 2))
    dataset = []

    tasks = []
    for pair in video_pairs:
        video1_path = os.path.join(video_dir, pair[0])
        video2_path = os.path.join(video_dir, pair[1])

        # Check FPS
        video1 = cv2.VideoCapture(video1_path)
        video2 = cv2.VideoCapture(video2_path)
        fps1 = video1.get(cv2.CAP_PROP_FPS)
        fps2 = video2.get(cv2.CAP_PROP_FPS)

        assert round(fps1) == 30, f"FPS of {pair[0]} is not 30: {fps1}"
        assert round(fps2) == 30, f"FPS of {pair[1]} is not 30: {fps2}"

        total_frames1 = int(video1.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames2 = int(video2.get(cv2.CAP_PROP_FRAME_COUNT))

        min_total_frames = min(total_frames1, total_frames2)
        sample_frame_size = 480

        safe_margin = max_offset

        for _ in range(k):
            # Randomly select a start frame
            start_frame = random.randint(
                safe_margin, min_total_frames - sample_frame_size - safe_margin
            )
            end_frame = start_frame + sample_frame_size

            # Random offset
            offset = random.randint(-max_offset, max_offset)

            adjusted_start_frame_video1 = start_frame + offset
            adjusted_end_frame_video1 = end_frame + offset

            # Ensure the adjusted frames are within bounds
            if (
                adjusted_start_frame_video1 < safe_margin
                or adjusted_end_frame_video1 > min_total_frames - safe_margin
            ):
                continue  # Skip if the frames go out of bounds

            video1_output_file = f"{event_name}_{pair[0].split('.')[0]}_{adjusted_start_frame_video1:06}_{adjusted_end_frame_video1:06}.mp4"
            video2_output_file = f"{event_name}_{pair[1].split('.')[0]}_{start_frame:06}_{end_frame:06}.mp4"

            video1_output_path = os.path.join(output_dir, video1_output_file)
            video2_output_path = os.path.join(output_dir, video2_output_file)

            task = (
                video1_path,
                video2_path,
                start_frame,
                end_frame,
                offset,
                video1_output_path,
                video2_output_path,
            )

            # Add the task and the associated metadata entry
            tasks.append(task)
            data_entry = {
                "id": event_id,
                "name": event_name,
                f"video_file_0": video1_output_path,
                f"video_file_1": video2_output_path,
                f"seq_len_0": sample_frame_size,
                f"seq_len_1": sample_frame_size,
                f"label_0": offset,
                f"label_1": 0,  # Sync offset as label for the second video
            }
            dataset.append(data_entry)

    return tasks, dataset


def sync_and_save_video_pair(video1_path, video2_path, start_frame, end_frame, offset, video1_output_path, video2_output_path,
                             frame_skip=12, green_threshold=0.8, black_threshold=0.8, 
                             similarity_threshold=0.9, similar_frames_ratio_threshold=0.8, min_frames_for_similarity_check=10):
    """Open video files, synchronize, detect bad frames, and save to output videos."""
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)

    # Initialize VideoWriter for output videos
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out1 = cv2.VideoWriter(video1_output_path, fourcc, fps1, (224, 224))
    out2 = cv2.VideoWriter(video2_output_path, fourcc, fps2, (224, 224))

    try:
        cap1.set(cv2.CAP_PROP_POS_FRAMES, start_frame + offset)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        previous_frame1 = None
        previous_frame2 = None

        similar_frame_count1 = 0
        similar_frame_count2 = 0
        total_frames_checked = 0

        for frame_num in range(end_frame - start_frame):
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 or not ret2:
                # Return with reason for corrupted file or early termination
                return False, "file_corrupted"

            # Check for green or black frames
            if is_green_screen_frame(frame1, green_threshold=green_threshold):
                return False, "green_frame"
            if is_green_screen_frame(frame2, green_threshold=green_threshold):
                return False, "green_frame"
            if is_black_screen_frame(frame1, black_threshold=black_threshold):
                return False, "black_frame"
            if is_black_screen_frame(frame2, black_threshold=black_threshold):
                return False, "black_frame"

            # Perform SSIM-based static frame check every `frame_skip` frames
            if frame_num % frame_skip == 0:
                if previous_frame1 is not None:
                    if are_frames_similar_ssim(previous_frame1, frame1, similarity_threshold):
                        similar_frame_count1 += 1
                if previous_frame2 is not None:
                    if are_frames_similar_ssim(previous_frame2, frame2, similarity_threshold):
                        similar_frame_count2 += 1

                total_frames_checked += 1

                # Only check for similarity ratio after checking at least `min_frames_for_similarity_check` frames
                if total_frames_checked >= min_frames_for_similarity_check:
                    if (similar_frame_count1 / total_frames_checked > similar_frames_ratio_threshold or
                            similar_frame_count2 / total_frames_checked > similar_frames_ratio_threshold):
                        return False, "static_video"

                previous_frame1 = frame1
                previous_frame2 = frame2

            # Resize frames to 224x224
            frame1_resized = cv2.resize(frame1, (224, 224))
            frame2_resized = cv2.resize(frame2, (224, 224))

            # Write frames to output videos
            out1.write(frame1_resized)
            out2.write(frame2_resized)

        return True, None  # Indicate success

    finally:
        # Ensure resources are released in all cases
        cap1.release()
        cap2.release()
        out1.release()
        out2.release()


def signal_handler(sig, frame):
    global pool
    print("\nInterrupt received, terminating processes...")
    if pool:
        pool.terminate()  # Forcefully stop all worker processes
        pool.join()  # Wait for termination
    sys.exit(0)  # Exit the main process


def main(data_root, output_dir, k=5, max_offset=30):
    global pool
    os.makedirs(output_dir, exist_ok=True)
    processed_videos_dir = os.path.join(output_dir, "processed_videos")
    os.makedirs(processed_videos_dir, exist_ok=True)
    processed_data = []

    event_id = 0
    task_list = []

    # Set up a signal handler for interrupt (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)

    # Use a multiprocessing pool with a fixed number of CPUs
    pool = multiprocessing.Pool(processes=4)

    pbar = tqdm(total=0, desc="Processing all events", dynamic_ncols=True)

    def update_progress(_):
        pbar.update(1)

    # Iterate over events and create tasks
    for event_dir in os.listdir(data_root):
        full_event_dir = os.path.join(data_root, event_dir)
        if not os.path.isdir(full_event_dir):
            continue

        tasks, event_dataset = process_event(
            full_event_dir, processed_videos_dir, event_id, k, max_offset
        )

        logging.info(f"Event {event_id}: {len(tasks)} tasks created.")
        event_id += 1
        processed_data.extend(event_dataset)

        # Add each task with its corresponding dataset entry
        for task, dataset_entry in zip(tasks, event_dataset):
            # Link each task to its corresponding entry
            task_list.append((task, dataset_entry))

    pbar.total = len(task_list)

    logging.info(f"Total tasks: {len(task_list)}")
    logging.info(
        f"Total processed data before processing: {len(processed_data)} entries"
    )

    failed_tasks = load_failed_tasks()
    # Process tasks
    for task, dataset_entry in task_list:
        result = pool.apply_async(
            process_video_pair, args=(task, failed_tasks), callback=update_progress
        )

        if not result.get():  # If the task fails
            logging.warning(f"Processing failed for entry: {dataset_entry}")
            if dataset_entry in processed_data:
                # Remove only the failed task's entry
                processed_data.remove(dataset_entry)

    pool.close()
    pool.join()

    pbar.close()

    logging.info(f"Final processed data length: {len(processed_data)} entries")

    # Shuffle the processed_data before splitting
    random.shuffle(processed_data)

    # Split the data into 80% train and 20% val
    split_index = len(processed_data) // 5
    val_data = processed_data[:split_index]
    train_data = processed_data[split_index:]

    # Save train and val data
    with open(os.path.join(output_dir, "train.pkl"), "wb") as f:
        pickle.dump(train_data, f)
    logging.info(f"Training data saved to train.pkl with {len(train_data)} entries")

    with open(os.path.join(output_dir, "val.pkl"), "wb") as f:
        pickle.dump(val_data, f)
    logging.info(f"Validation data saved to val.pkl with {len(val_data)} entries")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python cmu_process.py <data_root>")
        sys.exit(1)
    data_root = sys.argv[1]
    main(data_root, data_root, k=5, max_offset=30)
