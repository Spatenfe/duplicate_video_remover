import torch
import os
import csv
import argparse
from tqdm import tqdm
from utils import load_video

parser = argparse.ArgumentParser(description='Video Duplicate Finder')

parser.add_argument('-t', '--similarity_threshold', type=float,
                    help='Lowest similarity score that the combination is still saved.',
                  default=0.55)
parser.add_argument('-d', '--video_dir_path',
                    help='Directory path that should be checked for duplicates.')
parser.add_argument('--device',
                    help='Device to run at.',
                  default="cpu")
parser.add_argument('-f', '--check_csv_file_path',
                    help='Path of the csv file that should be checked. If this flag is not provided all the videos in the directory are used.')
parser.add_argument('-o', '--out_dir', default='./',
                    help='Output directory of the csv file')
parser.add_argument('-l', '--file_limit', type=int,
                    help='Output directory of the csv file', default=999_999_999)

def print_run_config(args):
    print("------------------------------------------")
    print("Running with:")
    print("similarity_threshold = " + str(args.similarity_threshold))
    print("device = " + args.device)
    print("video_dir_path = " + str(args.video_dir_path))
    print("check_csv_file_path = " + str(args.check_csv_file_path))
    print("file_limit = " + str(args.file_limit))
    print("out_dir = " + str(args.out_dir))
    print("------------------------------------------")

    if args.video_dir_path is None:
        print("Please provide a --video_dir_path")
        exit()

def get_file_paths_from_csv(csv_path):
    file_names = []
    with open(csv_path, mode='r') as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
            file_names.append(lines[0])

    return file_names

if __name__ == '__main__':
    args = parser.parse_args()

    print_run_config(args)

    # set args
    device = args.device
    threshold_similarity = args.similarity_threshold
    dir_path = args.video_dir_path
    csv_file_path = args.check_csv_file_path
    file_limit = args.file_limit
    out_dir = args.out_dir

    #disable grad for efficient VRAM usage
    with torch.no_grad():
        #init models
        feat_extractor = torch.hub.load('gkordo/s2vs:main', 'resnet50_LiMAC')
        s2vs_dns = torch.hub.load('gkordo/s2vs:main', 's2vs_dns')
        #s2vs_vcdb = torch.hub.load('gkordo/s2vs:main', 's2vs_vcdb')

        feat_extractor.eval()
        s2vs_dns.eval()

        feat_extractor = feat_extractor.to(device)

        #init save architectures
        video_path_list = []
        if csv_file_path is None:
            video_path_list = os.listdir(dir_path)
        else:
            video_path_list = get_file_paths_from_csv(csv_file_path)

        print("Found " + str(len(video_path_list)) + " video files...")

        feature_array = []

        #generate all feature embeddings
        print("------------------------------------------")
        print("feature extraction started!")
        print("------------------------------------------")

        num_of_videos = min(file_limit, len(video_path_list))
        for i in tqdm(range(num_of_videos)):
            file_name = video_path_list[i]
            path = os.path.join(dir_path, file_name)
            video = torch.from_numpy(load_video(path)).to(device)
            video_features = feat_extractor(video)

            if len(video_features.shape) != 3 or video_features.shape[0] < 1 or video_features.shape[1] != 9 or video_features.shape[2] != 512:
                print("Skipped " + str(file_name) + " with shape: " + video_features.shape)
                del video
                del video_features
                torch.cuda.empty_cache()
                continue

            feature_array.append((file_name, video_features))

            del video
            torch.cuda.empty_cache()

        #calculate all similarity scores with O(n*n/2)
        s2vs_dns = s2vs_dns.to(device)
        print("------------------------------------------")
        print("Comparison started!")
        print("------------------------------------------")

        #init csv save file
        with open(os.path.join(out_dir, 'duplication_analysis_result.csv'), 'w', newline='') as csvfile:
            fieldnames = ['name_a', 'name_b', 'score']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            #run for all video combinations
            for step in tqdm(range(len(feature_array))):
                query_feature = feature_array[step]
                for target_num in range(step + 1, len(feature_array)):
                    #compare two videos
                    target_feature = feature_array[target_num]
                    similarity = s2vs_dns.calculate_video_similarity(query_feature[1], target_feature[1])

                    #log combination if similar score threshold is reached
                    if similarity[0] > threshold_similarity:
                        writer.writerow({'name_a': query_feature[0], 'name_b': target_feature[0],
                                         'score': round(float(similarity[0][0]), 2)})
