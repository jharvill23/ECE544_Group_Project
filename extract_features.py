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
import sys

"""Librosa issues a warning for every flac file so we need below if statement"""
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

config = edict(yaml.load(open('config.yml'), Loader=yaml.SafeLoader))

class Mel_log_spect(object):
    def __init__(self):
        self.nfft = config.data.fftl
        self.num_mels = config.data.num_mels
        self.hop_length = config.data.hop_length
        self.top_db = config.data.top_db
        self.sr = config.data.sr

    def feature_normalize(self, x):
        log_min = np.min(x)
        x = x - log_min
        x = x / self.top_db
        x = x.T
        return x

    def get_Mel_log_spect(self, y):
        y = librosa.util.normalize(S=y)
        spect = librosa.feature.melspectrogram(y=y, sr=self.sr, n_fft=self.nfft,
                                               hop_length=self.hop_length, n_mels=self.num_mels)
        log_spect = librosa.core.amplitude_to_db(spect, ref=1.0, top_db=self.top_db)
        log_spect = self.feature_normalize(log_spect)
        return log_spect

    def norm_Mel_log_spect_to_amplitude(self, feature):
        feature = feature * self.top_db
        spect = librosa.core.db_to_amplitude(feature, ref=1.0)
        return spect

    def audio_from_spect(self, feature):
        spect = self.norm_Mel_log_spect_to_amplitude(feature)
        audio = librosa.feature.inverse.mel_to_audio(spect.T, sr=self.sr, n_fft=self.nfft, hop_length=self.hop_length)
        return audio

    def convert_and_write(self, load_path, write_path):
        y, sr = librosa.core.load(path=load_path, sr=self.sr)
        feature = self.get_Mel_log_spect(y, n_mels=self.num_mels)
        audio = self.audio_from_spect(feature)
        librosa.output.write_wav(write_path, y=audio, sr=self.sr, norm=True)

def process(file):
    try:
        audio, sr = librosa.core.load(file, sr=config.data.sr)
        if sr == config.data.sr:
            feature_processor = Mel_log_spect()
            features = feature_processor.get_Mel_log_spect(audio)
            mel_log_dump_path = os.path.join(config.directories.features, file.split('/')[-1][:-4] + '.pkl')

            # plt.subplot(211)
            # plt.imshow(features.T)
            # plt.subplot(212)
            # plt.imshow(features.T)
            # plt.show()

            joblib.dump(features, mel_log_dump_path)

    except:
        print("Had trouble processing file " + file + " ...")

def main():
    if not os.path.isdir(config.directories.features):
        os.mkdir(config.directories.features)
    files = collect_files(config.directories.silence_removed)
    # process(files[0])
    with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        for _ in tqdm(executor.map(process, files)):
            """"""

if __name__ == "__main__":
    main()