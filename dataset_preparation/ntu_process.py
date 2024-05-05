# coding=utf-8
import os
import json
import cv2
from tqdm import tqdm
import pickle
# import torch


def main(split="train"):
    data_root = "/home/yosubs/koa_scratch/ntu"
    output_dir = os.path.join(data_root, "processed_videos")
    os.makedirs(output_dir, exist_ok=True)
    if split == "train":
        save_file = os.path.join(data_root, f"train.pkl")
    else:
        save_file = os.path.join(data_root, f"val.pkl")

    # annotation_file = os.path.join(data_root, f"finegym_annotation_info_{version}.json")
    # train_file = os.path.join(data_root, f"{classes}_train_element_{version}.txt")
    # val_file = os.path.join(data_root, f"{classes}_val_element.txt")
    video_dir = os.path.join(data_root, "raw_videos")

    ntu_syn_dir = os.path.join(data_root, "NTU-SYN/pose/test")

    # with open(annotation_file, 'r') as f:
    #     data=json.load(f)
    # with open(train_file, 'r') as f:
    #     train_lines = f.readlines()
    # with open(val_file, 'r') as f:
    #     val_lines = f.readlines()
    # if split == "train":
    #     lines = train_lines
    # else:
    #     lines = val_lines
    labels = {}
    video_ids = set()
    event_ids = set()
    # for line in lines:
    #     full_id = line.split(" ")[0]
    #     label = int(line.split(" ")[1])
    #     labels[full_id] = label
    #     video_id = full_id.split("_E_")[0]
    #     video_ids.add(video_id)
    #     event_id = full_id.split("_A_")[0]
    #     event_ids.add(event_id)

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
            # video_ids.add(file.split("_rgb")[0])

            for camera_id in ['001', '002']:
                file = f'{file[0:5]}{camera_id}{file[8:]}'
                event_id = file.split("_rgb")[0]
                event_ids.add(event_id)

                with open(os.path.join(root, file), 'r') as f:
                    data = json.load(f)
                    labels[event_id] = data['category_id']

            # scene = file[1:4]
            
            # performer = file[9:12]
            # replication = file[13:16]
            # action = file[17:20]

    # event_ids = set([
    #     'S001C001P001R002A007',
    #     'S001C002P001R002A007'
    # ])
    # for event_id in event_ids:
    #     labels[event_id] = 0
    # start_times = [
    #     0,
    #     1
    # ]

    dataset = []

    # print('ntu_process.py:74')

    for i, event_id in tqdm(enumerate(event_ids), total=len(event_ids)):
        # print('event_id' + event_id)

        output_file = os.path.join(output_dir, event_id) + ".mp4"
        video_id = event_id

        if not os.path.exists(output_file):
            video_path = os.path.join(video_dir, video_id)
            suffix = "_rgb.avi"

            # print('video_path' + video_path)
            if os.path.exists(video_path+suffix):
                video_file = video_path+suffix
            else:
                continue

            # print('line 88')

            # start_time = start_times[i]

            # temp_output_file = os.path.join(output_dir, event_id) + "_temp.mp4"
            # cmd = f'ffmpeg -hide_banner -loglevel panic -y -i {video_file} -c:v copy -c:a copy {output_file}'
            # os.system(cmd)

            frame_offset = labels[event_id]
            frame_select_filter = '' if frame_offset == 0 else f'select=gte(n\,{str(frame_offset)}),'

            # print(video_file)
            # print('frame_offset', frame_offset, 'frame_select_filter', frame_select_filter)

            # print('line 94')
            cmd = f'ffmpeg -hide_banner -loglevel panic -y -i {video_file} -strict -2 -vf "{frame_select_filter}scale=640:360" {output_file}'
            os.system(cmd)
            # cmd = f'ffmpeg -hide_banner -loglevel panic -y -i {temp_output_file} -filter:v fps=25 {output_file}'
            # os.system(cmd)
            # os.remove(temp_output_file)

        # print('line105')
        video = cv2.VideoCapture(output_file)
        fps = int(video.get(cv2.CAP_PROP_FPS))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # print(video_file, "\n", output_file)
        # print(event_id, num_frames, fps, width, height)

        data_dict = {"id": i, "name": event_id, "video_file": os.path.join("processed_videos", event_id+".mp4"),
                     "seq_len": num_frames, "label": labels[event_id]}
        dataset.append(data_dict)
        # print(data_dict)
        # print('line 114')
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"{len(dataset)} {split} samples of NTU dataset have been writen.")


if __name__ == '__main__':
    print('ntu_process.py')
    main(split="train")
    main(split="val")
