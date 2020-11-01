import os
from tqdm import tqdm
import joblib
import silence_removal
import yaml
from easydict import EasyDict as edict
import shutil
import multiprocessing
import concurrent.futures
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

config = edict(yaml.load(open('config.yml'), Loader=yaml.SafeLoader))

def process(file):
    audio, sr = librosa.core.load(file, sr=config.data.sr)
    if sr == config.data.sr:
        """Save the data as a wav file instead of flac"""
        name = file.split('/')[-1][:-5] + '.wav'
        dump_path = os.path.join(config.directories.all_audio, name)
        x = np.round(audio * 32767)
        x = x.astype('int16')
        # plt.plot(x)
        # plt.show()
        sf.write(dump_path, x, sr, subtype='PCM_16')

def main(files):
    """"""
    #process(files[0])
    with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        for _ in tqdm(executor.map(process, files)):
            """"""


if __name__ == "__main__":
    """"""