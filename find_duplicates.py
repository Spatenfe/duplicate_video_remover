import time

import torch
import os
import numpy as np
import csv
from tqdm import tqdm

from utils import load_video

if __name__ == '__main__':
    #disable grad for efficient VRAM usage
    with torch.no_grad():
        #define settings
        device = 'cuda'
        dir_path = '/home/stud/foef/Hiwi/Video-Sim-VPDQ/example_videos/'
        threshold_similarity = 0.6

        #init models
        feat_extractor = torch.hub.load('gkordo/s2vs:main', 'resnet50_LiMAC')
        s2vs_dns = torch.hub.load('gkordo/s2vs:main', 's2vs_dns')
        #s2vs_vcdb = torch.hub.load('gkordo/s2vs:main', 's2vs_vcdb')

        feat_extractor.eval()
        s2vs_dns.eval()

        feat_extractor = feat_extractor.to(device)

        #init save architectures
        video_path_list = os.listdir(dir_path)
        feature_array = []

        #generate all feature embeddings
        print("------------------------------------------")
        print("feature extraction started!")
        print("------------------------------------------")
        for i in tqdm(range(len(video_path_list))):
            file_name = video_path_list[i]
            path = dir_path + file_name
            video = torch.from_numpy(load_video(path)).to(device)
            video_features = feat_extractor(video)

            feature_array.append((file_name, video_features))

            del video
            torch.cuda.empty_cache()

        #calculate all similarity scores with O(n*n/2)
        s2vs_dns = s2vs_dns.to(device)
        print("------------------------------------------")
        print("Comparison started!")
        print("------------------------------------------")

        #init csv save file
        with open('logs.csv', 'w', newline='') as csvfile:
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
