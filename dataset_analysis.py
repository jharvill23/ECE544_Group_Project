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
import matplotlib.pyplot as plt

config = edict(yaml.load(open('config.yml'), Loader=yaml.SafeLoader))

all_data = collect_files(config.directories.features)

if not os.path.exists('dataset_frame_lengths.pkl'):
    lengths = []
    for file in tqdm(all_data):
        data = joblib.load(file)
        lengths.append(data.shape[0])
    lengths = np.asarray(lengths)
    joblib.dump(lengths, 'dataset_frame_lengths.pkl')
else:
    lengths = joblib.load('dataset_frame_lengths.pkl')
hist = np.histogram(lengths, bins=20)
stop = None