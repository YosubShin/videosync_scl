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


def check_video_duration(file_path):
    """
    Use ffprobe to check if the video file has a duration of 0.
    Returns True if the file has a duration of 0, False otherwise.
    """
    try:
        result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'format=duration',
                                '-of', 'default=noprint_wrappers=1:nokey=1', file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        duration = result.stdout.strip()
        if duration == 'N/A' or duration == "" or float(duration) == 0 :
            return True
        return False
    except Exception as e:
        print(f"Error checking file {file_path}: {e} {duration}")
        return True


def find_corrupted_files(target_dir, target_filenames, max_workers=4):
    """
    Check for corrupted files in the target directory using multiple threads.
    """
    corrupted_files = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_filename = {executor.submit(check_video_duration, os.path.join(
            target_dir, filename)): filename for filename in target_filenames}

        for future in tqdm(as_completed(future_to_filename), total=len(future_to_filename), desc="Checking for corrupted files"):
            filename = future_to_filename[future]
            try:
                if future.result():
                    corrupted_files.append(filename)
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
                corrupted_files.append(filename)
    return corrupted_files


def main(csv_path, target_dir, max_workers=4):
    """
    Main function to read CSV, generate filenames, and find corrupted files.
    """
    target_filenames = read_csv_and_generate_filenames(csv_path)
    corrupted_files = find_corrupted_files(
        target_dir, target_filenames, max_workers)

    print("Corrupted files:")
    print('Total corrupted files:', len(corrupted_files))
    print('Percentage of corrupted files:', len(
        corrupted_files) / len(target_filenames) * 100)

    # Save corrupted files list to csv
    with open('corrupted_files.csv', mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(['filename'])
        for filename in corrupted_files:
            writer.writerow([filename])


if __name__ == "__main__":
    csv_path = '/home/yosubs/koa_scratch/kinetics-dataset/k400/annotations/train.csv'
    target_dir = '/home/yosubs/koa_scratch/kinetics-dataset/k400/train'
    max_workers = 8  # Replace with the number of threads you want to use
    main(csv_path, target_dir, max_workers)
