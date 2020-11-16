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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import itertools
from cycler import cycler

config = edict(yaml.load(open('config.yml'), Loader=yaml.SafeLoader))

def cov_matrix(directory, dim):
    files = collect_files(directory)
    random.shuffle(files)
    embeddings = []
    for i, file in tqdm(enumerate(files)):
        if i < 89000:
            data = joblib.load(file)
            embeddings.append(data)
        i += 1
    embeddings = np.asarray(embeddings)
    embeddings = embeddings.T
    cov = np.cov(embeddings)
    plt.imshow(cov)
    plt.savefig('cov_matrix_dim_' + str(dim) + '.png')
    plt.close()

    """Add mean of diagonal value and mean of non-diagonal values"""
    diag = np.diagonal(cov)
    non_diag = []
    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            if i != j:
                non_diag.append(cov[i, j])
    non_diag = np.asarray(non_diag)
    diag_mean = np.mean(diag)
    # non_diag_mean = (np.sum(cov) - np.sum(np.diagonal(cov)))/(cov.shape[0]**2 - cov.shape[0])
    non_diag_mean = np.mean(non_diag)
    diag_var = np.var(diag)
    non_diag_var = np.var(non_diag)
    stats = {'diag_mean': diag_mean, 'non_diag_mean': non_diag_mean,
             'diag_var': diag_var, 'non_diag_var': non_diag_var}
    with open('cov_matrix_dim_' + str(dim) + '_stats.json', 'w') as fp:
        json.dump(stats, fp)
    stop = None

def get_cov():
    """Get cov_matrix for binary speaker embeddings and save to disk"""
    if not os.path.exists('binary_cov_matrix.pkl'):
        # cov = cov_matrix('/home/john/Documents/School/Fall_2020/ECE544/hashed_256_32000G', dim=256)
        # cov = cov_matrix('/home/john/Documents/School/Fall_2020/ECE544/ECE544_Group_Project/hashed_128_dim', dim=128)
        cov = cov_matrix(config.directories.hashed_embeddings, dim=16)
    else:
        cov = joblib.load('binary_cov_matrix.pkl')

def main():
    """"""
    get_cov()


if __name__ == '__main__':
    main()