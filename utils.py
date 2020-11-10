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

def get_speaker_info(file=config.directories.speaker_info):
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
        speaker = speaker.split('/')[-1]
        speaker = speaker.split('.')[0]
        s2c[speaker] = i
        c2s[i] = speaker
    joblib.dump(s2c, 'speaker2class.pkl')
    joblib.dump(c2s, 'class2speaker.pkl')
    return s2c, c2s

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
    utt_number = utterance.split('_')[1]
    mic = utterance.split('_')[2]
    try:
        additional_data = speaker_info[speaker]
    except:
        """This is for speaker 280 in old dataset, data wasn't provided"""
        additional_data = {'age': None, 'gender': None, 'accent': None}
    try:
        text = text_data[speaker + '_' + utt_number]
    except:
        text = None
    """Get one-hot vector for speaker as well"""
    one_hot = np.zeros(shape=(len(speaker2class),))
    one_hot[speaker2class[speaker]] = 1
    return_data = {'speaker': speaker, 'utt_number': utt_number, 'text': text,
                   'speaker_age': additional_data['age'], 'speaker_gender': additional_data['gender'],
                   'speaker_accent': additional_data['accent'], 'one_hot': one_hot, 'mic': mic,
                   'speaker_class': speaker2class[speaker]}
    return return_data

def get_partition():
    all_files = collect_files(config.directories.features)
    if not os.path.exists('partition.pkl'):
        if not os.path.exists('time_limited_files.pkl'):
            """Limit audios only to those longer than 300 frames (3 seconds)"""
            allowable_files = []
            for file in tqdm(all_files):
                data = joblib.load(file)
                # plt.imshow(data.T)
                # plt.show()
                if data.shape[0] >= 200 and data.shape[0] <= 350:
                    allowable_files.append(file)
            joblib.dump(allowable_files, 'time_limited_files.pkl')
            # return allowable_files
        else:
            allowable_files = joblib.load('time_limited_files.pkl')
            # return allowable_files


        random.shuffle(allowable_files)
        split_index = int(config.hash.train_val_split*len(allowable_files))
        train = allowable_files[0:split_index]
        val = allowable_files[split_index:]
        partition = {'train': train, 'val': val}
        joblib.dump(partition, 'partition.pkl')
        return partition
    else:
        partition = joblib.load('partition.pkl')
        return partition

def main():
    """"""
    # metadata = get_metadata('./features/p225_004_mic2.pkl')
    partition = get_partition()

if __name__ == '__main__':
    main()