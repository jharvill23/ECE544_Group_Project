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
import pysptk
from pysptk.synthesis import Synthesizer, MLSADF
import matplotlib.pyplot as plt
import random
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path

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
        chunk_list.append(next_list)
    return chunk_list

def process_list(filelist):
    encoder = VoiceEncoder()
    for file in filelist:
        try:
            fpath = Path(file)
            wav = preprocess_wav(fpath)
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
    chunks = split_list(files, num_chunks=multiprocessing.cpu_count())

    # process_list(chunks[0])
    print("Calculating embeddings... (sorry for no good progress indicator, check output directory)")
    with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        for _ in executor.map(process_list, chunks):
            """"""

if __name__ == "__main__":
    main()