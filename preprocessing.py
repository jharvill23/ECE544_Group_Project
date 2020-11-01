import os
from tqdm import tqdm
import joblib
import silence_removal
import yaml
from easydict import EasyDict as edict
import shutil
import flac_to_wav
import sys

"""Librosa issues a warning for every flac file so we need below if statement"""
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

config = edict(yaml.load(open('config.yml'), Loader=yaml.SafeLoader))

STAGE = '012'
"""Stage 0 collects all files"""
"""Stage 1 is moving all recordings to common directory as wav files instead of flac"""
"""Stage 2 is removing silence"""

def collect_files(directory):
    all_files = []
    for path, subdirs, files in tqdm(os.walk(directory)):
        for name in files:
            filename = os.path.join(path, name)
            all_files.append(filename)
    return all_files

def keep_flac(files):
    new_files = []
    for f in files:
        if '.flac' in f:
            new_files.append(f)
    return new_files

def main():
    if '0' in STAGE:
        """Collect files to preprocess by speaker"""
        speakers = [(f.path).split('/')[-1] for f in os.scandir(config.directories.root) if f.is_dir()]
        for speaker in speakers:
            folder = os.path.join(config.directories.root, speaker)
            files = collect_files(folder)
            joblib.dump(files, os.path.join(config.directories.speakers, speaker + '.pkl'))

        """Combine into one list of files to process using multiprocessing"""
        for i, speaker in enumerate(collect_files(config.directories.speakers)):
            if i == 0:
                files = joblib.load(speaker)
            else:
                files.extend(joblib.load(speaker))
        files = keep_flac(files)  # make sure all files are .flac files

    if '1' in STAGE:
        if not os.path.isdir(config.directories.all_audio):
            os.mkdir(config.directories.all_audio)
        """Move the recordings"""
        flac_to_wav.main(files)
        # for file in tqdm(files):
        #     name = file.split('/')[-1]
        #     dump_path = os.path.join(config.directories.all_audio, name)
        #     shutil.copy(file, dump_path)
    if '2' in STAGE:
        # if not os.path.isdir(config.directories.silence_removed):
        #     os.mkdir(config.directories.silence_removed)
        files = collect_files(config.directories.all_audio)
        silence_removal.main(files)

if __name__ == "__main__":
    if not os.path.isdir(config.directories.speakers):
        os.mkdir(config.directories.speakers)
    main()