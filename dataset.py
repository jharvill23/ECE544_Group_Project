from torch.utils import data
import joblib
from torch.nn.utils.rnn import pad_sequence
import torch
from utils import get_metadata
import yaml
from easydict import EasyDict as edict
import numpy as np
import matplotlib.pyplot as plt

config = edict(yaml.load(open('config.yml'), Loader=yaml.SafeLoader))

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  # taken from https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
  def __init__(self, params):
        'Initialization'
        self.list_IDs = params['files']

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        """Get the data item"""
        file = self.list_IDs[index]
        metadata = get_metadata(file)
        """Load spectrogram from disk"""
        spectrogram = joblib.load(file)

        """Let's limit spectrograms to 120 to 300 frames"""
        # if spectrogram.shape[0] > 100 and spectrogram.shape[0] < 300:

        """Convert spectrogram and one-hot to tensors"""
        spectrogram = torch.from_numpy(spectrogram)
        one_hot = torch.from_numpy(metadata['one_hot'])
        """Get speaker index"""
        speaker_index = int(np.squeeze(np.nonzero(metadata['one_hot'])))
        return spectrogram, one_hot, metadata, speaker_index

        # else:
        #     return None, None, None, None

  def fix_tensor(self, x):
      x.requires_grad = True
      x = x.cuda()
      return x

  def keep_non_None_elements(self, list):
      new_list = []
      for element in list:
          if element != None:
              new_list.append(element)
      return new_list

  def collate(self, batch):
      spectrograms = [item[0] for item in batch]
      one_hots = [item[1] for item in batch]
      metadata = [item[2] for item in batch]
      speaker_indices = [item[3] for item in batch]

      # """Only keep elements that aren't None"""
      # new_spects = self.keep_non_None_elements(spectrograms)
      # new_one_hots = self.keep_non_None_elements(one_hots)
      # new_metadata = self.keep_non_None_elements(metadata)
      # new_speaker_indices = self.keep_non_None_elements(speaker_indices)


      # try:


      """Batch the spectrograms and one_hots"""
      spectrograms = pad_sequence(spectrograms, batch_first=True, padding_value=0)
      spectrograms = self.fix_tensor(spectrograms)
      spectrograms = torch.unsqueeze(spectrograms, 1)  # for conv input
      one_hots = pad_sequence(one_hots, batch_first=True, padding_value=0)  # not actually padding,
      # just using to take list of tensors and make tensor (all one-hots same length)
      one_hots = self.fix_tensor(one_hots)

      return {"spectrograms": spectrograms, "one_hots": one_hots,
              "metadata": metadata, "speaker_indices": speaker_indices}


      # except:
      #     return None




