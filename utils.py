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

config = edict(yaml.load(open('config.yml'), Loader=yaml.SafeLoader))

def get_speaker_info(file='speaker-info.txt'):
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

"""Build basic pickle files for organized data"""
if not os.path.exists('speaker-info.pkl'):
    get_speaker_info()
else:
    speaker_info = joblib.load('speaker-info.pkl')
if not os.path.exists('text_data.pkl'):
    get_text_data()
else:
    text_data = joblib.load('text_data.pkl')

def get_metadata(file):
    utterance = file.split('/')[-1][:-4]
    speaker = utterance.split('_')[0]
    speaker = speaker.replace('p', '')
    utt_number = utterance.split('_')[1]
    additional_data = speaker_info[speaker]
    text = text_data[utterance]
    return_data = {'speaker': speaker, 'utt_number': utt_number, 'text': text,
                   'speaker_age': additional_data['age'], 'speaker_gender': additional_data['gender'],
                   'speaker_accent': additional_data['accent']}
    return return_data

def main():
    """"""

if __name__ == '__main__':
    main()