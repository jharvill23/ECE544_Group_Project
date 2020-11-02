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

def save_tsne_plot(tsne_embeddings, type):
    """Now group embeddings by speaker and plot"""
    files = tsne_embeddings['filelist']
    embeddings = tsne_embeddings['tsne']
    grouped_embeddings = {}
    for file, embed in zip(files, embeddings):
        metadata = utils.get_metadata(file)
        speaker = metadata['speaker']
        if speaker not in grouped_embeddings:
            grouped_embeddings[speaker] = [embed]
        else:
            grouped_embeddings[speaker].append(embed)
    """Now plot the grouped embeddings"""

    fig = plt.figure()
    ax = fig.add_subplot(111)
    marker = ['o', '+', 'x', '*', '.', 'X']
    for speaker, embeddings in tqdm(grouped_embeddings.items()):
        new_marker = marker[random.randint(0, 5)]
        # new_marker = 'o'
        color = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]
        embeddings = np.asarray(embeddings)

        x = embeddings[:, 0]
        y = embeddings[:, 1]
        for q, p in zip(x, y):
            ax.plot(q, p, linestyle='', marker=new_marker, color=color)
    if type == 'continuous':
        plt.title('t-SNE of Continuous Speaker Embeddings')
        plt.savefig('continuous_tsne_embeddings.png')
    elif type == 'binary':
        plt.title('t-SNE of Binary Speaker Embeddings')
        plt.savefig('binary_tsne_embeddings.png')
    # plt.show()
    plt.close()

def tsne_data(directory, type):
    files = collect_files(directory)
    random.shuffle(files)
    embeddings = []
    for i, file in tqdm(enumerate(files)):
        if i < 2000:
            data = joblib.load(file)
            embeddings.append(data)
        i += 1
    embeddings = np.asarray(embeddings)
    tsne_embeddings = TSNE(n_components=2, verbose=10, perplexity=50).fit_transform(embeddings)
    data_to_save = {'filelist': files, 'tsne': tsne_embeddings}  # file in same row as embedding in numpy array
    if type == 'continuous':
        joblib.dump(data_to_save, 'continuous_tsne_embeddings.pkl')
    elif type == 'binary':
        joblib.dump(data_to_save, 'binary_tsne_embeddings.pkl')
    tsne_embeddings = data_to_save
    return tsne_embeddings

def get_tsne(continuous, binary):
    if continuous == True:
        """Get tsne for continuous speaker embeddings and save to disk"""
        if not os.path.exists('continuous_tsne_embeddings.pkl'):
            tsne_embeddings = tsne_data(config.directories.continuous_embeddings, type='continuous')
            # files = collect_files(config.directories.continuous_embeddings)
            # random.shuffle(files)
            # #embeddings = np.zeros(shape=(len(files), 256))
            # embeddings = []
            # for i, file in tqdm(enumerate(files)):
            #     if i < 2000:
            #         data = joblib.load(file)
            #         embeddings.append(data)
            #     #embeddings[i] = data
            #     i += 1
            # embeddings = np.asarray(embeddings)
            # tsne_embeddings = TSNE(n_components=2, verbose=10, perplexity=5).fit_transform(embeddings)
            # data_to_save = {'filelist': files, 'tsne': tsne_embeddings}  # file in same row as embedding in numpy array
            # joblib.dump(data_to_save, 'continuous_tsne_embeddings.pkl')
            # tsne_embeddings = data_to_save
        else:
            tsne_embeddings = joblib.load('continuous_tsne_embeddings.pkl')
        save_tsne_plot(tsne_embeddings, type='continuous')

    if binary == True:
        """Get tsne for binary speaker embeddings and save to disk"""
        if not os.path.exists('binary_tsne_embeddings.pkl'):
            tsne_embeddings = tsne_data(config.directories.hashed_embeddings, type='binary')
            # files = collect_files(config.directories.hashed_embeddings)
            # random.shuffle(files)
            # #embeddings = np.zeros(shape=(len(files), 256))
            # embeddings = []
            # for i, file in tqdm(enumerate(files)):
            #     if i < 200:
            #         data = joblib.load(file)
            #         embeddings.append(data)
            #     #embeddings[i] = data
            #     i += 1
            # embeddings = np.asarray(embeddings)
            # tsne_embeddings = TSNE(n_components=2, verbose=10, perplexity=5).fit_transform(embeddings)
            # data_to_save = {'filelist': files, 'tsne': tsne_embeddings}  # file in same row as embedding in numpy array
            # joblib.dump(data_to_save, 'binary_tsne_embeddings.pkl')
            # tsne_embeddings = data_to_save
        else:
            tsne_embeddings = joblib.load('binary_tsne_embeddings.pkl')
        save_tsne_plot(tsne_embeddings, type='binary')

def main():
    """"""
    get_tsne(continuous=False, binary=True)


if __name__ == '__main__':
    main()