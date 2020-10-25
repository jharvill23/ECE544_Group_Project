import os
import pandas as pd
import joblib
from preprocessing import collect_files
import shutil
from tqdm import tqdm
import yaml
from easydict import EasyDict as edict
import random
import re
import numpy as np

config = edict(yaml.load(open('config.yml'), Loader=yaml.SafeLoader))

def get_speaker_info(file='speaker-info.txt'):
    """Note: Speaker 280 doesn't have info for some reason"""
    speaker_info = {}
    with open(file) as f:
        for i, l in enumerate(f):
            if i > 0:
                l = re.sub(' +', ' ', l)
                info = l.split(' ')
                speaker = info[0]
                age = info[1]
                gender = info[2]
                accent = info[3]
                if speaker not in speaker_info:
                    speaker_info[speaker] = {'age': age, 'gender': gender, 'accent': accent}
    joblib.dump(speaker_info, 'speaker-info.pkl')
    return speaker_info

def get_text_data(dir=config.directories.text):
    speakers = [(f.path).split('/')[-1] for f in os.scandir(dir) if f.is_dir()]
    file_text = {}
    for speaker in speakers:
        for file in collect_files(os.path.join(dir, speaker)):
            filename = file.split('/')[-1][:-4]
            text = []
            with open(file) as f:
                for l in f:
                    l = l.replace('\n', '')
                    text.append(l)
            file_text[filename] = text
    joblib.dump(file_text, 'text_data.pkl')
    return file_text

def get_speaker2class_and_class2speaker():
    s2c = {}
    c2s = {}
    for i, speaker in enumerate(collect_files(config.directories.speakers)):
        speaker = speaker.split('/')[-1][1:4]
        s2c[speaker] = i
        c2s[i] = speaker
    joblib.dump(s2c, 'speaker2class.pkl')
    joblib.dump(c2s, 'class2speaker.pkl')
    return s2c, c2s

def check_weird_files():
    files = collect_files(config.directories.features)
    weird_files = []
    for file in files:
        speaker_number = file.split('/')[-1][1:4]
        if speaker_number not in speaker2class:
            weird_files.append(file)
    return weird_files

"""Build basic pickle files for organized data"""
if not os.path.exists('speaker-info.pkl'):
    speaker_info = get_speaker_info()
else:
    speaker_info = joblib.load('speaker-info.pkl')
if not os.path.exists('text_data.pkl'):
    text_data = get_text_data()
else:
    text_data = joblib.load('text_data.pkl')
if not os.path.exists('speaker2class.pkl') or not os.path.exists('class2speaker.pkl'):
    speaker2class, class2speaker = get_speaker2class_and_class2speaker()
else:
    speaker2class = joblib.load('speaker2class.pkl')
    class2speaker = joblib.load('class2speaker.pkl')

def get_metadata(file):
    utterance = file.split('/')[-1][:-4]
    speaker = utterance.split('_')[0]
    speaker = speaker.replace('p', '')
    utt_number = utterance.split('_')[1]
    try:
        additional_data = speaker_info[speaker]
    except:
        """This is for speaker 280, data isn't provided"""
        additional_data = {'age': None, 'gender': None, 'accent': None}
    try:
        text = text_data[utterance]
    except:
        text = None
    """Get one-hot vector for speaker as well"""
    one_hot = np.zeros(shape=(len(speaker2class),))
    one_hot[speaker2class[speaker]] = 1
    return_data = {'speaker': speaker, 'utt_number': utt_number, 'text': text,
                   'speaker_age': additional_data['age'], 'speaker_gender': additional_data['gender'],
                   'speaker_accent': additional_data['accent'], 'one_hot': one_hot}
    return return_data

def get_partition():
    all_files = collect_files(config.directories.features)
    random.shuffle(all_files)
    split_index = int(config.hash.train_val_split*len(all_files))
    train = all_files[0:split_index]
    val = all_files[split_index:]
    partition = {'train': train, 'val': val}
    joblib.dump(partition, 'partition.pkl')


def main():
    """"""
    # weird_files = check_weird_files()
    # file = './features/p280_001.pkl'
    # metadata = get_metadata(file)
    # stop = None

if __name__ == '__main__':
    main()