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

config = edict(yaml.load(open('config.yml'), Loader=yaml.SafeLoader))

if not os.path.exists(config.directories.exps):
    os.mkdir(config.directories.exps)

trial = 'trial_8_hash_training_resnet18'
exp_dir = os.path.join(config.directories.exps, trial)
if not os.path.isdir(exp_dir):
    os.mkdir(exp_dir)

TRAIN = False
LOAD_MODEL = True
RESUME_TRAINING = False
if RESUME_TRAINING:
    LOAD_MODEL = True

EVAL = True
RESNET18 = True

class Solver(object):
    """Solver"""

    def __init__(self):
        """Initialize configurations."""

        # Training configurations.
        self.g_lr = config.model.lr
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
        self.model_save_step = config.train.model_save_step

        # Build the model SKIP FOR NOW SO YOU CAN GET DATALOADING DONE
        self.build_model()
        if LOAD_MODEL:
            self.restore_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Build the model"""
        if not RESNET18:
            self.G = model_hash.DAMH(config)
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
        G_path = './exps/trial_8_hash_training_resnet18/models/6000-G.ckpt'
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
            spectrograms = features['spectrograms']
            phones = features['phones']
            input_lengths = features['input_lengths']
            target_lengths = features['target_lengths']
            metadata = features["metadata"]
            # batch_speakers = [x['speaker'] for x in metadata]
            self.G = self.G.eval()

            """Make input_lengths and target_lengths torch ints"""
            input_lengths = input_lengths.to(torch.int32)
            target_lengths = target_lengths.to(torch.int32)
            phones = phones.to(torch.int32)

            spectrograms = spectrograms.to(self.torch_type)

            outputs = self.G(spectrograms)

            outputs = outputs.permute(1, 0, 2)  # swap batch and sequence length dimension for CTC loss

            loss = self.ctc_loss(log_probs=outputs, targets=phones,
                                 input_lengths=input_lengths, target_lengths=target_lengths)

            val_loss_value += loss.item()
            """Update the loss history MUST BE SEPARATE FROM TRAINING"""
            # self.update_history_val(loss, batch_speakers)
        """We have the history, now do something with it"""
        # val_loss_means = {}
        # for key, value in self.val_history.items():
        #     val_loss_means[key] = np.mean(np.asarray(value))
        # val_loss_means_sorted = {k: v for k, v in sorted(val_loss_means.items(), key=lambda item: item[1])}
        # weights = {}
        # counter = 1
        # val_loss_value = 0
        # for key, value in val_loss_means_sorted.items():
        #     val_loss_value += (config.train.fairness_lambda * counter + (1-config.train.fairness_lambda) * 1) * value
        #     counter += 1

        return val_loss_value

    def get_train_val_split(self):
        partition = joblib.load('partition.pkl')
        train = partition['train']
        val = partition['val']
        return train, val

    def custom_loss(self, classification_outputs, hash_outputs, binary_outputs, W, one_hots, speaker_indices, m):
        s = config.hash.s
        lambda_ = 0.1*(1/config.model.binary_embedding_length)
        num_examples = classification_outputs.shape[0]  # could use config.train.batch_size too
        margin_loss = 0
        hash_loss = 0
        for i in range(num_examples):  # num_examples is N in paper
            """Margin loss"""
            index = speaker_indices[i]
            sub_sum = 0
            for j in range(1, one_hots.shape[0]):
                if j != index:
                    W_j = torch.squeeze(W[j, :])
                    h_i = torch.squeeze(hash_outputs[i])
                    theta_ji = F.cosine_similarity(W_j, h_i, dim=0)
                    term = torch.exp(s*theta_ji)
                    sub_sum += term
            W_yi = torch.squeeze(W[index, :])
            h_yi = torch.squeeze(hash_outputs[i])
            theta_yi_i = F.cosine_similarity(W_yi, h_yi, dim=0)
            main_term = torch.exp(s*(theta_yi_i-m))
            example_term = torch.log(main_term/(main_term + sub_sum))
            margin_loss += example_term
            """Hash loss"""
            hash_loss += torch.sum(torch.square(hash_outputs[i] - binary_outputs[i]))
        margin_loss = (-1/num_examples)*margin_loss
        hash_loss = (lambda_/num_examples)*hash_loss
        loss = margin_loss + hash_loss
        return loss, margin_loss, hash_loss

    def train(self):
        iterations = 0
        """Get train/val"""
        train, val = self.get_train_val_split()
        m = 0
        for epoch in range(config.train.num_epochs):
            if m < 0.35:
                m += 0.02
                if m > 0.35:
                    m = 0.35
            """Make dataloader"""
            train_data = Dataset({'files': train})
            train_gen = data.DataLoader(train_data, batch_size=config.train.batch_size,
                                        shuffle=True, collate_fn=train_data.collate, drop_last=True)
            val_data = Dataset({'files': val})
            val_gen = data.DataLoader(val_data, batch_size=config.train.batch_size,
                                        shuffle=True, collate_fn=val_data.collate, drop_last=True)

            for batch_number, features in enumerate(train_gen):
                try:
                    spectrograms = features['spectrograms']
                    one_hots = features['one_hots']
                    metadata = features["metadata"]
                    speaker_indices = features["speaker_indices"]

                    """Pass spectrogram through ResNet"""
                    self.G = self.G.train()  # we have batch normalization layers so this is necessary
                    spectrograms = spectrograms.to(self.torch_type)
                    if RESNET18:
                        spectrograms = spectrograms.repeat(1, 3, 1, 1)
                    classification_outputs, hash_outputs, binary_outputs, W = self.G(spectrograms)

                    """Take loss"""
                    loss, margin_loss, hash_loss = self.custom_loss(classification_outputs=classification_outputs,
                                            hash_outputs=hash_outputs,
                                            binary_outputs=binary_outputs,
                                            W=W,
                                            one_hots=one_hots,
                                            speaker_indices=speaker_indices, m=m)

                    """Backward and optimize"""
                    self.reset_grad()
                    loss.backward()
                    self.g_optimizer.step()

                    if iterations % self.log_step == 0:
                        # print('speaker: ' + metadata['speaker'])
                        print(str(iterations) + ', loss: ' + str(loss.item()) + ', margin: ' + str(margin_loss.item()) + ', hash: ' + str(hash_loss.item()))
                        if self.use_tensorboard:
                            self.logger.scalar_summary('loss', loss.item(), iterations)
                            self.logger.scalar_summary('m', m, iterations)
                            self.logger.scalar_summary('margin_loss', margin_loss.item(), iterations)
                            self.logger.scalar_summary('hash_loss', hash_loss.item(), iterations)

                    if iterations % self.model_save_step == 0:
                        """Calculate validation loss"""
                        # val_loss = self.val_loss(val=val_gen, iterations=iterations)
                        # print(str(iterations) + ', val_loss: ' + str(val_loss))
                        # if self.use_tensorboard:
                        #     self.logger.scalar_summary('val_loss', val_loss, iterations)
                    """Save model checkpoints."""
                    if iterations % self.model_save_step == 0:
                        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(iterations))
                        torch.save({'model': self.G.state_dict(),
                                    'optimizer': self.g_optimizer.state_dict()}, G_path)
                        print('Saved model checkpoints into {}...'.format(self.model_save_dir))

                    iterations += 1
                except:
                    """GPU ran out of memory, batch too big"""

    def eval(self):
        if not os.path.isdir(config.directories.hashed_embeddings):
            os.mkdir(config.directories.hashed_embeddings)
        train, val = self.get_train_val_split()
        train_data = Dataset({'files': train})
        train_gen = data.DataLoader(train_data, batch_size=1,
                                    shuffle=True, collate_fn=train_data.collate, drop_last=True)
        val_data = Dataset({'files': val})
        val_gen = data.DataLoader(val_data, batch_size=1,
                                  shuffle=True, collate_fn=val_data.collate, drop_last=True)
        for batch_number, features in tqdm(enumerate(train_gen)):
            spectrograms = features['spectrograms']
            one_hots = features['one_hots']
            metadata = features["metadata"]
            speaker_indices = features["speaker_indices"]

            """Pass spectrogram through ResNet"""
            try:
                self.G = self.G.eval()  # we have batch normalization layers so this is necessary
                # Keep in mind, ^^^ could be messing up predictions (try .train() too, had problems
                # with this in the past
                spectrograms = spectrograms.to(self.torch_type)
                if RESNET18:
                    spectrograms = spectrograms.repeat(1, 3, 1, 1)
                classification_outputs, hash_outputs, binary_outputs, W = self.G(spectrograms)

                binary_outputs = binary_outputs.detach().cpu().numpy()
                binary_outputs = np.squeeze(binary_outputs)

                utterance_name = metadata[0]['speaker'] + '_' + metadata[0]['utt_number'] + '_' + metadata[0]['mic'] + '.pkl'
                dump_path = os.path.join(config.directories.hashed_embeddings, utterance_name)
                joblib.dump(binary_outputs, dump_path)
            except:
                print('Audio too short...')

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