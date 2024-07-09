import csv
import os
from tqdm import tqdm


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


def get_existing_files(target_dir):
    """
    Get a set of all existing files in the target directory.
    """
    existing_files = set()
    for root, _, files in os.walk(target_dir):
        for file in files:
            existing_files.add(file)
    return existing_files


def find_missing_files(target_dir, target_filenames):
    """
    Check for missing files in the target directory.
    """
    existing_files = get_existing_files(target_dir)
    missing_files = [filename for filename in tqdm(
        target_filenames, desc="Checking files") if filename not in existing_files]
    return missing_files


def main(csv_path, target_dir):
    """
    Main function to read CSV, generate filenames, and find missing files.
    """
    target_filenames = read_csv_and_generate_filenames(csv_path)
    missing_files = find_missing_files(target_dir, target_filenames)

    print("Missing files:")
    # for file in missing_files:
    #     print(file)
    print('total missing files:', len(missing_files))
    print('total number of files:', len(target_filenames))
    print('percentage of missing files:', len(
        missing_files) / len(target_filenames) * 100)

    # Save missing files list to csv
    with open('missing_files.csv', mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(['filename'])
        for filename in missing_files:
            writer.writerow([filename])


if __name__ == "__main__":
    csv_path = '/home/yosubs/koa_scratch/kinetics-dataset/k400/annotations/train.csv'
    target_dir = '/home/yosubs/koa_scratch/kinetics-dataset/k400/train'
    main(csv_path, target_dir)
