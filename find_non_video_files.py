import csv
import os
import subprocess
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def extract_target_filename(row):
    """
    Extract the target video file name based on the row values.
    """
    return f'{row[1]}_{row[2].zfill(6)}_{row[3].zfill(6)}.mp4'


def read_csv_and_generate_filenames(csv_path):
    """
    Read the CSV file and generate a list of target video file names.
    """
    target_filenames = []
    with open(csv_path, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header row if it exists
        for row in tqdm(csv_reader, desc="Reading CSV"):
            target_filename = extract_target_filename(row)
            target_filenames.append(target_filename)
    return target_filenames


def check_video_stream(file_path):
    """
    Use ffprobe to check if the video file has a video stream.
    Returns True if the file lacks a video stream, False otherwise.
    """
    try:
        # Check if video stream is present
        stream_info = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v',
                '-show_entries', 'stream=index', '-of', 'csv=p=0', file_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if not stream_info.stdout.strip():
            return True
        return False
    except Exception as e:
        print(f"Error checking file {file_path}: {e}")
        return True


def find_files_missing_video_stream(target_dir, target_filenames, max_workers=4):
    """
    Check for files missing video streams in the target directory using multiple threads.
    """
    files_missing_video_stream = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_filename = {executor.submit(check_video_stream, os.path.join(
            target_dir, filename)): filename for filename in target_filenames}

        for future in tqdm(as_completed(future_to_filename), total=len(future_to_filename), desc="Checking for files missing video streams"):
            filename = future_to_filename[future]
            try:
                if future.result():
                    files_missing_video_stream.append(filename)
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
                files_missing_video_stream.append(filename)
    return files_missing_video_stream


def main(csv_path, target_dir, max_workers=4):
    """
    Main function to read CSV, generate filenames, and find files missing video streams.
    """
    target_filenames = read_csv_and_generate_filenames(csv_path)
    files_missing_video_stream = find_files_missing_video_stream(
        target_dir, target_filenames, max_workers)

    print("Files missing video streams:")
    print('Total files missing video streams:',
          len(files_missing_video_stream))
    print('Percentage of files missing video streams:', len(
        files_missing_video_stream) / len(target_filenames) * 100)

    # Save files missing video streams list to csv
    with open('files_missing_video_stream.csv', mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(['filename'])
        for filename in files_missing_video_stream:
            writer.writerow([filename])


if __name__ == "__main__":
    csv_path = '/home/yosubs/koa_scratch/kinetics-dataset/k400/annotations/train.csv'
    target_dir = '/home/yosubs/koa_scratch/kinetics-dataset/k400/train'
    max_workers = 8  # Replace with the number of threads you want to use
    main(csv_path, target_dir, max_workers)
