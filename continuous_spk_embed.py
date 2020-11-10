import librosa
import numpy as np
import os
import joblib
import multiprocessing
import concurrent.futures
from tqdm import tqdm
from preprocessing import collect_files
import yaml
from easydict import EasyDict as edict
# import pysptk
# from pysptk.synthesis import Synthesizer, MLSADF
import matplotlib.pyplot as plt
import random
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import sys

"""Librosa issues a warning for every flac file so we need below if statement"""
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

MAX_PROCESSES = 8
NUM_PROCESSES = min(MAX_PROCESSES, multiprocessing.cpu_count())

config = edict(yaml.load(open('config.yml'), Loader=yaml.SafeLoader))

def split_list(x, num_chunks):
    """"""
    step_size = int(len(x)/num_chunks)
    chunk_list = []
    for i in range(num_chunks):
        if i != num_chunks-1:
            next_list = x[i*step_size:(i+1)*step_size]
        else:
            next_list = x[i*step_size:]
        chunk_list.append({'list': next_list, 'number': i})  # only have the number so we can use for tqdm
        # for one of the num_cpu processes
    return chunk_list

def process_list(data):
    encoder = VoiceEncoder()
    number = data['number']
    filelist = data['list']
    if number == 0:  # we have tqdm here to show progress of first process
        for file in tqdm(filelist):
            try:
                fpath = Path(file)
                wav = preprocess_wav(fpath)
                # plt.plot(wav)
                # plt.show()
                embed = encoder.embed_utterance(wav)
                dump_path = os.path.join(config.directories.continuous_embeddings, file.split('/')[-1][:-4] + '.pkl')
                joblib.dump(embed, dump_path)
            except:
                print("Had trouble processing file " + file + " ...")
        print('Process 0 finished, others should finish soon if not done already...')
    else:  # no tqdm for other processes
        for file in filelist:
            try:
                fpath = Path(file)
                wav = preprocess_wav(fpath)
                # plt.plot(wav)
                # plt.show()
                embed = encoder.embed_utterance(wav)
                dump_path = os.path.join(config.directories.continuous_embeddings, file.split('/')[-1][:-4] + '.pkl')
                joblib.dump(embed, dump_path)
            except:
                print("Had trouble processing file " + file + " ...")


def main():
    if not os.path.isdir(config.directories.continuous_embeddings):
        os.mkdir(config.directories.continuous_embeddings)
    files = collect_files(config.directories.silence_removed)

    "Split into num_cpu lists"
    random.shuffle(files)
    chunks = split_list(files, num_chunks=NUM_PROCESSES)

    # process_list(chunks[0])
    print("Calculating embeddings...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
        for _ in executor.map(process_list, chunks):
            """"""

if __name__ == "__main__":
    main()