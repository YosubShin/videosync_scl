# coding=utf-8
import os
import json
import cv2
from tqdm import tqdm
import pickle


def main(split="train"):
    data_root = "/home/yosubs/koa_scratch/ntu"
    output_dir = os.path.join(data_root, "processed_videos")
    os.makedirs(output_dir, exist_ok=True)
    if split == "train":
        save_file = os.path.join(data_root, f"train.pkl")
    else:
        save_file = os.path.join(data_root, f"val.pkl")

    video_dir = os.path.join(data_root, "raw_videos")
    ntu_syn_dir = os.path.join(data_root, "NTU-SYN/pose/test")

    labels = {}
    event_ids = set()

    video_pair_ids = set()

    for root, _, files in os.walk(ntu_syn_dir):
        for i, file in enumerate(files):
            if not file.endswith('.json'):
                continue

            camera = file[5:8]
            if camera != '001':
                continue

            # Sample 1/16 only.
            if i % 16 != 0:
                continue

            # Sample file name:
            # S001C002P001R002A007_rgb.avi.json
            # S: scene, C: camera, P: performer, R: replication, A: action

            for camera_id in ['001', '002']:
                file = f'{file[0:5]}{camera_id}{file[8:]}'
                event_id = file.split("_rgb")[0]
                event_ids.add(event_id)

                with open(os.path.join(root, file), 'r') as f:
                    data = json.load(f)
                    labels[event_id] = data['category_id']

    dataset = []

    for i, event_id in tqdm(enumerate(event_ids), total=len(event_ids)):
        output_file = os.path.join(output_dir, event_id) + ".mp4"
        video_id = event_id

        if not os.path.exists(output_file):
            video_path = os.path.join(video_dir, video_id)
            suffix = "_rgb.avi"

            if os.path.exists(video_path+suffix):
                video_file = video_path+suffix
            else:
                continue

            frame_offset = labels[event_id]
            frame_select_filter = '' if frame_offset == 0 else f'select=gte(n\,{str(frame_offset)}),'

            cmd = f'ffmpeg -hide_banner -loglevel panic -y -i {video_file} -strict -2 -vf "{frame_select_filter}scale=640:360" {output_file}'
            os.system(cmd)

        video = cv2.VideoCapture(output_file)
        fps = int(video.get(cv2.CAP_PROP_FPS))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        data_dict = {"id": i, "name": event_id, "video_file": os.path.join("processed_videos", event_id+".mp4"),
                     "seq_len": num_frames, "label": labels[event_id]}
        dataset.append(data_dict)
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"{len(dataset)} {split} samples of NTU dataset have been writen.")


if __name__ == '__main__':
    main(split="train")
    main(split="val")
