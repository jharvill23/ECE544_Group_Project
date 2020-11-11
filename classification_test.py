import os
from tqdm import tqdm
import numpy as np
import joblib
import torch
import torch.nn as nn
import yaml
from easydict import EasyDict as edict
import shutil
from preprocessing import collect_files
import utils
from torch.utils import data
from itertools import groupby
import json
from Levenshtein import distance as levenshtein_distance
import multiprocessing
import concurrent.futures
import random
#import model
from dataset import Dataset
import model_hash
import torch.nn.functional as F
import matplotlib.pyplot as plt

config = edict(yaml.load(open('config.yml'), Loader=yaml.SafeLoader))

print(torch.__version__)
# print(torchvision.__version__)



if not os.path.exists(config.directories.exps):
    os.mkdir(config.directories.exps)

trial = 'trial_13_CLASSIFICATION'
exp_dir = os.path.join(config.directories.exps, trial)
if not os.path.isdir(exp_dir):
    os.mkdir(exp_dir)

TRAIN = False
LOAD_MODEL = True
RESUME_TRAINING = False
if RESUME_TRAINING:
    LOAD_MODEL = True

EVAL = True
RESNET18 = False

class Classifier(nn.Module):
    def __init__(self, config):
        super(Classifier, self).__init__()
        self.config = config
        self.vocab_size = 80  # need this for dimension of output of BLSTM
        self.hidden_size = 300
        self.batch_first = True
        self.dropout = 0.1
        self.bidirectional = True
        self.adaptive_length = 32
        self.longest_phone_sequence_len = 15  # THIS IS FOR UASPEECH DATASET
        self.lstm = nn.LSTM(input_size=self.vocab_size, hidden_size=self.hidden_size,
                            num_layers=2, batch_first=self.batch_first,
                            dropout=self.dropout, bidirectional=self.bidirectional)
        # self.pool = nn.AdaptiveAvgPool2d(output_size=(self.adaptive_length, self.vocab_size))
        # self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(in_features=self.hidden_size*2,
                            out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=500)
        self.fc3 = nn.Linear(in_features=500, out_features=config.vctk.num_speakers)
    def forward(self, x):
        """Classification"""
        x, _ = self.lstm(x)
        seq_len = x.size()[1]
        batch_size = x.size()[0]
        if self.bidirectional:
            num_directions = 2
        else:
            num_directions = 1
        # x = x.view(seq_len, batch, num_directions, hidden_size)
        x = x.view(batch_size, seq_len, num_directions, self.hidden_size)
        # x_numpy = x.detach().cpu().numpy()
        # first = x_numpy[0]
        # forward_numpy = x_numpy[:, -1, 0, :]
        # backward_numpy = x_numpy[:, 0, 1, :]

        # forward_summary = x[:, -1, 0, :]
        forward_summary = x[:, seq_len-1, 0, :]
        backward_summary = x[:, 0, 1, :]
        summaries = torch.cat((forward_summary, backward_summary), dim=1)




        # x = self.pool(x)
        # x = self.flatten(x)
        """Concatenate the phones with the flattened input"""

        x = self.fc1(summaries)
        x = F.relu_(x)
        x = self.fc2(x)
        x = F.relu_(x)
        x = self.fc3(x)
        # x = F.sigmoid(x)  # make it a probability
        return x

class Solver(object):
    """Solver"""

    def __init__(self):
        """Initialize configurations."""

        # Training configurations.
        self.g_lr = 0.001
        self.torch_type = torch.float32

        # Miscellaneous.
        self.use_tensorboard = True
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:{}'.format(0) if self.use_cuda else 'cpu')

        # Directories.
        self.log_dir = os.path.join(exp_dir, 'logs')
        self.model_save_dir = os.path.join(exp_dir, 'models')
        self.train_data_dir = config.directories.features

        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)
        if not os.path.isdir(self.model_save_dir):
            os.mkdir(self.model_save_dir)

        self.partition = 'partition.pkl'
        """Partition file"""
        if TRAIN:  # only copy these when running a training session, not eval session
            if not os.path.exists('partition.pkl'):
                utils.get_partition()
            shutil.copy(src='partition.pkl',
                        dst=os.path.join(exp_dir, 'partition.pkl'))
            # copy config as well
            shutil.copy(src='config.yml', dst=os.path.join(exp_dir, 'config.yml'))


        # Step size.
        self.log_step = config.train.log_step
        self.model_save_step = 1000

        # Build the model SKIP FOR NOW SO YOU CAN GET DATALOADING DONE
        self.build_model()
        if LOAD_MODEL:
            self.restore_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Build the model"""
        if not RESNET18:
            self.G = Classifier(config)
        else:
            self.G = model_hash.ResNet18DAMH(config)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr)
        self.print_network(self.G, 'G')
        self.G.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def print_optimizer(self, opt, name):
        print(opt)
        print(name)

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def _load(self, checkpoint_path):
        if self.use_cuda:
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path,
                                    map_location=lambda storage, loc: storage)
        return checkpoint

    def restore_model(self):
        """Restore the model"""
        print('Loading the trained models... ')
        # G_path = './exps/trial_1_hash_training/models/260000-G.ckpt'
        # G_path = './exps/trial_8_hash_training_resnet18/models/6000-G.ckpt'
        # G_path = './exps/trial_9_hash_training_resnet18/models/2000-G.ckpt'
        G_path = './exps/trial_13_CLASSIFICATION/models/84000-G.ckpt'
        g_checkpoint = self._load(G_path)
        self.G.load_state_dict(g_checkpoint['model'])
        self.g_optimizer.load_state_dict(g_checkpoint['optimizer'])
        self.g_lr = self.g_optimizer.param_groups[0]['lr']

    def update_lr(self, g_lr):
        """Decay learning rates of g"""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()

    def val_loss(self, val, iterations):
        """Time to write this function"""
        self.val_history = {}
        val_loss_value = 0
        for batch_number, features in tqdm(enumerate(val)):
            if batch_number < 200:
                spectrograms = features['spectrograms']

                """Debugging, want to check spectrograms first"""
                # spect_np = np.squeeze(spectrograms.detach().cpu().numpy())
                # for spect in spect_np:
                #     plt.imshow(spect.T)
                #     plt.show()
                #     stop = None

                one_hots = features['one_hots']
                metadata = features["metadata"]
                speaker_indices = features["speaker_indices"]

                """Pass spectrogram through ResNet"""
                self.G = self.G.eval()  # we have batch normalization layers so this is necessary
                spectrograms = spectrograms.to(self.torch_type)
                spectrograms = torch.squeeze(spectrograms)
                if RESNET18:
                    spectrograms = spectrograms.repeat(1, 3, 1, 1)
                classification_outputs = self.G(spectrograms)

                """Get the index from one-hots for cross-entropy loss"""
                indices = []
                for row in one_hots:
                    indices.append(torch.nonzero(row))

                indices_ = torch.nn.utils.rnn.pad_sequence(indices, padding_value=0)
                indices = torch.squeeze(indices_)

                """Take loss"""
                loss = self.loss_function(input=classification_outputs, target=indices)
                val_loss_value += loss.detach().cpu().numpy()

        return val_loss_value

    def get_train_val_split(self):
        partition = joblib.load('partition.pkl')
        train = partition['train']
        val = partition['val']
        return train, val

    def get_s2c_and_c2s(self):
        self.speaker2class = joblib.load('speaker2class.pkl')
        self.class2speaker = joblib.load('class2speaker.pkl')

    def train(self):
        self.get_s2c_and_c2s()
        iterations = 0
        """Get train/val"""
        train, val = self.get_train_val_split()
        m = 0
        for epoch in range(config.train.num_epochs):
            # """Just trying setting m large"""
            m = 0.02

            # if m < 0.35:
            #     m += 0.02
            #     if m > 0.35:
            #         m = 0.35

            """Make dataloader"""
            train_data = Dataset({'files': train})
            train_gen = data.DataLoader(train_data, batch_size=4,
                                        shuffle=True, collate_fn=train_data.collate, drop_last=True)
            val_data = Dataset({'files': val})
            val_gen = data.DataLoader(val_data, batch_size=32,
                                        shuffle=True, collate_fn=val_data.collate, drop_last=True)
            self.loss_function = torch.nn.CrossEntropyLoss()
            for batch_number, features in enumerate(train_gen):
                # try:
                    if features != None:
                        spectrograms = features['spectrograms']

                        """Debugging, want to check spectrograms first"""
                        # spect_np = np.squeeze(spectrograms.detach().cpu().numpy())
                        # for spect in spect_np:
                        #     plt.imshow(spect.T)
                        #     plt.show()
                        #     stop = None

                        one_hots = features['one_hots']
                        metadata = features["metadata"]
                        speaker_indices = features["speaker_indices"]

                        """Pass spectrogram through ResNet"""
                        self.G = self.G.train()  # we have batch normalization layers so this is necessary
                        spectrograms = spectrograms.to(self.torch_type)
                        spectrograms = torch.squeeze(spectrograms)
                        if RESNET18:
                            spectrograms = spectrograms.repeat(1, 3, 1, 1)
                        classification_outputs = self.G(spectrograms)

                        # """Get the index from one-hots for cross-entropy loss"""
                        # indices = []
                        # for row in one_hots:
                        #     indices.append(torch.nonzero(row))

                        speaker_indices = np.asarray(speaker_indices)
                        indices_ = torch.from_numpy(speaker_indices)
                        indices = torch.squeeze(indices_)
                        indices = self.to_gpu(indices)
                        indices = indices.to(torch.long)

                        """Take loss"""
                        loss = self.loss_function(input=classification_outputs, target=indices)

                        """Backward and optimize"""
                        self.reset_grad()
                        loss.backward()
                        self.g_optimizer.step()

                        if iterations % self.log_step == 0:
                            # print('speaker: ' + metadata['speaker'])
                            print(str(iterations) + ', loss: ' + str(loss.item()))
                            if self.use_tensorboard:
                                self.logger.scalar_summary('loss', loss.item(), iterations)

                        if iterations % self.model_save_step == 0:
                            """Calculate validation loss"""
                            val_loss = self.val_loss(val=val_gen, iterations=iterations)
                            print(str(iterations) + ', val_loss: ' + str(val_loss))
                            if self.use_tensorboard:
                                self.logger.scalar_summary('val_loss', val_loss, iterations)
                        """Save model checkpoints."""
                        if iterations % self.model_save_step == 0:
                            G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(iterations))
                            torch.save({'model': self.G.state_dict(),
                                        'optimizer': self.g_optimizer.state_dict()}, G_path)
                            print('Saved model checkpoints into {}...'.format(self.model_save_dir))

                        iterations += 1
                # except:
                #     """GPU ran out of memory, batch too big"""

    def eval(self):
        if not os.path.isdir(config.directories.hashed_embeddings):
            os.mkdir(config.directories.hashed_embeddings)
        train, val = self.get_train_val_split()
        train_data = Dataset({'files': train})
        train_gen = data.DataLoader(train_data, batch_size=32,
                                    shuffle=True, collate_fn=train_data.collate, drop_last=True)
        val_data = Dataset({'files': val})
        val_gen = data.DataLoader(val_data, batch_size=32,
                                  shuffle=True, collate_fn=val_data.collate, drop_last=True)
        incorrect_count = 0
        total_count = 0
        for batch_number, features in tqdm(enumerate(val_gen)):
            if features != None:
                spectrograms = features['spectrograms']

                one_hots = features['one_hots']
                metadata = features["metadata"]
                speaker_indices = features["speaker_indices"]

                """Pass spectrogram through ResNet"""
                self.G = self.G.train()  # we have batch normalization layers so this is necessary
                spectrograms = spectrograms.to(self.torch_type)
                spectrograms = torch.squeeze(spectrograms)
                if RESNET18:
                    spectrograms = spectrograms.repeat(1, 3, 1, 1)
                classification_outputs = self.G(spectrograms)
                classification_outputs = classification_outputs.detach().cpu().numpy()
                classification_outputs = np.squeeze(classification_outputs)
                predicted_indices = np.argmax(classification_outputs, axis=1)
                speaker_indices = np.asarray(speaker_indices)
                diff = predicted_indices - speaker_indices
                incorrect_count += np.count_nonzero(diff)
                total_count += val_gen.batch_size

        accuracy = (total_count - incorrect_count)/total_count
        print('Accuracy: ' + str(accuracy))



    def to_gpu(self, tensor):
        tensor = tensor.to(self.torch_type)
        tensor = tensor.to(self.device)
        return tensor

    def fix_tensor(self, x):
        x.requires_grad = True
        x = x.to(self.torch_type)
        x = x.cuda()
        return x

    def dump_json(self, dict, path):
        a_file = open(path, "w")
        json.dump(dict, a_file, indent=2)
        a_file.close()

def main():
    solver = Solver()
    if TRAIN:
        solver.train()
    if EVAL:
        solver.eval()



if __name__ == "__main__":
    main()