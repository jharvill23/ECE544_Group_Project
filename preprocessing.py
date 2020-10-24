import os
from tqdm import tqdm
import joblib
import silence_removal
import yaml
from easydict import EasyDict as edict

config = edict(yaml.load(open('config.yml'), Loader=yaml.SafeLoader))

STAGE = '01'
"""Stage 0 collects all files and moves to common directory audio/"""
"""Stage 1 is removing silence from recordings"""

def collect_files(directory):
    all_files = []
    for path, subdirs, files in tqdm(os.walk(directory)):
        for name in files:
            filename = os.path.join(path, name)
            all_files.append(filename)
    return all_files

def keep_wav(files):
    new_files = []
    for f in files:
        if '.wav' in f:
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
        files = keep_wav(files)  # make sure all files are .wav files

    if '1' in STAGE:
        if not os.path.isdir(config.directories.silence_removed):
            os.mkdir(config.directories.silence_removed)
        """Remove silence from recordings"""
        silence_removal.main(files)

if __name__ == "__main__":
    if not os.path.isdir(config.directories.speakers):
        os.mkdir(config.directories.speakers)
    main()