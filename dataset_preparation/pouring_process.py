# coding=utf-8
import os
import json
import cv2
from tqdm import tqdm
import pickle
import numpy as np
import subprocess


def get_video_frame_count(video_file_path):
    # Command to get the frame count using ffprobe
    command = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-count_frames', '-show_entries', 'stream=nb_read_frames',
        '-of', 'json', video_file_path
    ]

    # Execute the command and capture the output
    result = subprocess.run(command, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)

    # Parse the JSON output
    result_json = json.loads(result.stdout)

    # Extract the number of frames
    frame_count = int(result_json['streams'][0]['nb_read_frames'])

    return frame_count


def main(split="train"):
    data_root = "/home/yosubs/koa_scratch/pouring"
    if split == "train":
        save_file = os.path.join(data_root, f"train.pkl")
    else:
        save_file = os.path.join(data_root, f"val.pkl")

    video_dir = os.path.join(data_root, "videos")

    labels = [{}, {}]
    num_frames = [{}, {}]

    zero_or_one = 0 if split == "train" else 1
    video_name_prefixes = set()
    for root, _, files in os.walk(video_dir):
        for i, file in enumerate(files):
            if not file.endswith('.mp4'):
                continue
            elif (i % 4 == 0 or i % 4 == 1 or i % 4 == 2) and split == "val":
                continue
            elif (i % 4 == 3) and split == "train":
                continue

            # "creamsoda_to_clear5_real_view_1.mp4" => "creamsoda_to_clear5_real_view_"
            prefix = file.split('.')[0][:-1]
            video_name_prefixes.add(prefix)

    dataset = []

    for i, video_name_prefix in tqdm(enumerate(video_name_prefixes), total=len(video_name_prefixes)):
        data_dict = {"id": i, "name": video_name_prefix}
        # positive offset k means the first video starts at k'th frame and the second video starts at 0.
        # negative offset -k means the first video starts at 0'th frame and the second video starts at k'th frame
        frame_offset = np.random.randint(-30, 30)

        for j, camera_id in enumerate(range(2)):
            video_id = f'{video_name_prefix}{camera_id}'
            source_video_path = os.path.join(video_dir, video_id + ".mp4")

            local_frame_offset = 0
            if frame_offset > 0:
                if j == 0:
                    local_frame_offset = frame_offset
                else:
                    local_frame_offset = 0
            elif frame_offset < 0:
                if j == 0:
                    local_frame_offset = 0
                else:
                    local_frame_offset = -1 * frame_offset

            if not os.path.exists(source_video_path):
                print(
                    f'source video file does not exist: {source_video_path}')
                continue

            video = cv2.VideoCapture(source_video_path)
            fps = int(video.get(cv2.CAP_PROP_FPS))
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

            data_dict[f"video_file_{j}"] = source_video_path
            data_dict[f"seq_len_{j}"] = int(
                video.get(cv2.CAP_PROP_FRAME_COUNT))
            data_dict[f"label_{j}"] = local_frame_offset

        if "video_file_0" not in data_dict or "video_file_1" not in data_dict:
            print(
                f"Skipping the entry because we are missing some keys. name: {video_name_prefix}, data_dict.keys(): {data_dict.keys()}")
            continue

        dataset.append(data_dict)
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"{len(dataset)} {split} samples of Pouring dataset have been writen.")


if __name__ == '__main__':
    main(split="train")
    main(split="val")
