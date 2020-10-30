import os
from tqdm import tqdm
import numpy as np
import joblib
import torch
import torch.nn as nn
import yaml
from easydict import EasyDict as edict
import shutil
from preprocessing import collect_files
import utils
from torch.utils import data
from itertools import groupby
import json
from Levenshtein import distance as levenshtein_distance
import multiprocessing
import concurrent.futures
import random
#import model
from dataset import Dataset
import model_hash
import torch.nn.functional as F
from scipy.spatial import distance

config = edict(yaml.load(open('config.yml'), Loader=yaml.SafeLoader))

def analyze():
    if not os.path.exists('hash_pairs_by_speaker.pkl'):
        files = collect_files(config.directories.hashed_embeddings)
        """Sort by speaker"""
        files_by_speaker = {}
        for file in files:
            metadata = utils.get_metadata(file)
            if metadata['speaker'] not in files_by_speaker:
                files_by_speaker[metadata['speaker']] = [file]
            else:
                files_by_speaker[metadata['speaker']].append(file)
        """Now make pairs and then find unique ones using set"""
        pairs_by_speaker = {}
        for speaker, filelist in tqdm(files_by_speaker.items()):
            pairs = []
            for file_outer in filelist:
                for file_inner in filelist:
                    pairs.append((file_inner, file_outer))
            pairs = set(pairs)
            pairs = list(pairs)
            random.shuffle(pairs)
            pairs = pairs[0:5000]  # just take a random small subset of all pairs
            pairs_by_speaker[speaker] = pairs
        joblib.dump(pairs_by_speaker, 'hash_pairs_by_speaker.pkl')
    else:
        pairs_by_speaker = joblib.load('hash_pairs_by_speaker.pkl')
    if not os.path.exists('intra_speaker_distances.pkl'):
        speaker_distances = {}
        for speaker, pairs in tqdm(pairs_by_speaker.items()):
            for pair in pairs:
                embedding1 = joblib.load(pair[0])
                embedding2 = joblib.load(pair[1])
                dist = distance.hamming(embedding1, embedding2)
                if speaker not in speaker_distances:
                    speaker_distances[speaker] = [dist]
                else:
                    speaker_distances[speaker].append(dist)
        for speaker, distance_list in speaker_distances.items():
            mean_distance = np.mean(np.asarray(distance_list))
            speaker_distances[speaker] = mean_distance
        joblib.dump(speaker_distances, 'intra_speaker_distances.pkl')
    else:
        speaker_distances = joblib.load('intra_speaker_distances.pkl')
    if not os.path.exists('inter_speaker_distances.pkl'):
        inter_speaker_distances = {}
        for speaker_outer, pairs_outer in tqdm(pairs_by_speaker.items()):
            new_pairs = []
            distances = []
            for speaker_inner, pairs_inner in pairs_by_speaker.items():
                if speaker_inner != speaker_outer:
                    i = 0
                    for outer, inner in zip(pairs_outer, pairs_inner):
                        if i < 20:
                            new_pairs.append((outer[0], inner[0]))
                            embedding1 = joblib.load(outer[0])
                            embedding2 = joblib.load(outer[1])
                            dist = distance.hamming(embedding1, embedding2)
                            distances.append(dist)
                        i += 1
            mean_distance = np.mean(np.asarray(distances))
            inter_speaker_distances[speaker_outer] = mean_distance
            stop = None
        joblib.dump(inter_speaker_distances, 'inter_speaker_distances.pkl')
    else:
        inter_speaker_distances = joblib.load('inter_speaker_distances.pkl')
    total = 0
    for speaker, value in speaker_distances.items():
        total += value
    total = total/109
    print('Intra-speaker differences: ' + str(total))
    total = 0
    for speaker, value in inter_speaker_distances.items():
        total += value
    total = total / 109
    print('Inter-speaker differences: ' + str(total))
    stop = None



def main():
    analyze()

if __name__ == "__main__":
    main()