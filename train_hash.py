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
import model
from dataset import Dataset

config = edict(yaml.load(open('config.yml'), Loader=yaml.SafeLoader))

if not os.path.exists(config.directories.exps):
    os.mkdir(config.directories.exps)

trial = 'trial_1_hash_training'
exp_dir = os.path.join(config.directories.exps, trial)
if not os.path.isdir(exp_dir):
    os.mkdir(exp_dir)

TRAIN = True
LOAD_MODEL = False
RESUME_TRAINING = False
if RESUME_TRAINING:
    LOAD_MODEL = True

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
        self.predict_dir = os.path.join(exp_dir, 'predictions')

        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)
        if not os.path.isdir(self.model_save_dir):
            os.mkdir(self.model_save_dir)
        if not os.path.isdir(self.predict_dir):
            os.mkdir(self.predict_dir)

        self.partition = 'full_filenames_data_partition.pkl'
        """Partition file"""
        if TRAIN:  # only copy these when running a training session, not eval session
            # if not os.path.exists('ctc_partition.pkl'):
            #     utils.get_ctc_partition_offline_augmentation()
            # if not os.path.exists('baseline_ctc_partition.pkl'):
            #     utils.get_baseline_ctc_partition_offline_augmentation()
            # if not os.path.exists('dcgan_ctc_partition.pkl'):
            #     utils.get_dcgan_ctc_partition_offline_augmentation()
            # if not os.path.exists('original_unseen_ctc_partition.pkl'):
            #     utils.get_original_unseen_ctc_partition_offline_augmentation()
            # if not os.path.exists('unseen_normal_ctc_partition.pkl'):
            #     utils.get_unseen_normal_ctc_partition_offline_augmentation()
            # copy partition to exp_dir then use that for trial (just in case you change partition for other trials)
            # if BASELINE:
            #     shutil.copy(src='baseline_ctc_partition.pkl', dst=os.path.join(exp_dir, 'baseline_ctc_partition.pkl'))
            #     self.partition = os.path.join(exp_dir, 'baseline_ctc_partition.pkl')
            # elif ATTENTION:
            #     shutil.copy(src='ctc_partition.pkl', dst=os.path.join(exp_dir, 'ctc_partition.pkl'))
            #     self.partition = os.path.join(exp_dir, 'ctc_partition.pkl')
            # elif DCGAN:
            #     shutil.copy(src='dcgan_ctc_partition.pkl', dst=os.path.join(exp_dir, 'dcgan_ctc_partition.pkl'))
            #     self.partition = os.path.join(exp_dir, 'dcgan_ctc_partition.pkl')
            # elif ORIGINAL_UNSEEN_BASELINE:
            #     shutil.copy(src='original_unseen_ctc_partition.pkl', dst=os.path.join(exp_dir, 'original_unseen_ctc_partition.pkl'))
            #     self.partition = os.path.join(exp_dir, 'original_unseen_ctc_partition.pkl')
            # elif UNSEEN_NORMAL_BASELINE:
            #     shutil.copy(src='unseen_normal_ctc_partition.pkl',
            #                 dst=os.path.join(exp_dir, 'unseen_normal_ctc_partition.pkl'))
            #     self.partition = os.path.join(exp_dir, 'unseen_normal_ctc_partition.pkl')
            shutil.copy(src='full_filenames_data_partition.pkl',
                        dst=os.path.join(exp_dir, 'full_filenames_data_partition.pkl'))
            # copy config as well
            shutil.copy(src='config.yml', dst=os.path.join(exp_dir, 'config.yml'))
            # copy dict
            shutil.copy(src='dict.pkl', dst=os.path.join(exp_dir, 'dict.pkl'))
            # copy phones
            shutil.copy(src='phones.pkl', dst=os.path.join(exp_dir, 'phones.pkl'))
            # copy wordlist
            shutil.copy(src='wordlist.pkl', dst=os.path.join(exp_dir, 'wordlist.pkl'))

        # Step size.
        self.log_step = config.train.log_step
        self.model_save_step = config.train.model_save_step

        # Build the model
        self.build_model()
        if LOAD_MODEL:
            self.restore_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Build the model"""
        self.G = model.CTCmodel(config)
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
        # G_path = './exps/PARTITION_1_trial_1_attention_vc_CTC_TRAINING/models/310000-G.ckpt'
        if GLOBAL_PARTITION == 1:
            if ATTENTION:
                G_path = './exps/PARTITION_1_trial_2_attention_vc_CTC_TRAINING/models/460000-G.ckpt'
            if DCGAN:
                G_path = './exps/PARTITION_1_trial_2_dcgan_vc_CTC_TRAINING/models/585000-G.ckpt'
            if BASELINE:
                G_path = './exps/PARTITION_1_trial_2_Limited_Baseline_CTC_TRAINING/models/260000-G.ckpt'
            if ORIGINAL_UNSEEN_BASELINE:
                G_path = './exps/PARTITION_1_trial_2_Oracle_Baseline_CTC_TRAINING/models/305000-G.ckpt'
                # G_path = './exps/PARTITION_1_trial_3_Oracle_Baseline_CTC_TRAINING/models/230000-G.ckpt'
            if UNSEEN_NORMAL_BASELINE:
                G_path = './exps/PARTITION_1_trial_2_Lack_Baseline_CTC_TRAINING/models/420000-G.ckpt'
        elif GLOBAL_PARTITION == 2:
            if ATTENTION:
                G_path = './exps/PARTITION_2_trial_1_attention_vc_CTC_training/models/370000-G.ckpt'
            if DCGAN:
                G_path = './exps/PARTITION_2_trial_1_dcgan_vc_CTC_training/models/470000-G.ckpt'
            if ORIGINAL_UNSEEN_BASELINE:
                G_path = './exps/PARTITION_2_trial_1_Oracle_Baseline_CTC_training/models/240000-G.ckpt'
            if BASELINE:
                G_path = './exps/PARTITION_2_trial_1_Limited_Baseline_CTC_training/models/260000-G.ckpt'
            if UNSEEN_NORMAL_BASELINE:
                G_path = './exps/PARTITION_2_trial_1_Lack_Baseline_CTC_training/models/415000-G.ckpt'
            if RESUME_TRAINING:
                if UNSEEN_NORMAL_BASELINE:
                    G_path = './exps/PARTITION_2_trial_1_Lack_Baseline_CTC_training/models/170000-G.ckpt'
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

    def train(self):
        """Initialize history matrix"""
        self.history = np.random.normal(loc=0, scale=0.1, size=(len(self.s2i), config.train.class_history))
        """"""
        """"""
        iterations = 0
        if RESUME_TRAINING:
            iterations = 170000
        """Get train/test"""
        if WORDSPLIT:
            train, test = self.get_train_test_wordsplit()
        elif UTTERANCE_SPLIT:
            train, val = self.get_train_test_utterance_split()
        wordlist = joblib.load('wordlist.pkl')
        dictionary = joblib.load('dict.pkl')
        phones = joblib.load('phones.pkl')
        metadata_help = {'wordlist': wordlist, 'dictionary': dictionary, 'phones': phones}
        p2c = utils.phone2class(phones)
        c2p = utils.class2phone(phones)
        """CTC loss"""
        self.ctc_loss = nn.CTCLoss(blank=p2c[config.data.PAD_token], reduction='mean')
        # self.ctc_loss = nn.CTCLoss(blank=p2c[config.data.PAD_token], reduction='none')
        for epoch in range(config.train.num_epochs):
            """Make dataloader"""
            train_data = Dataset({'files': train, 'mode': 'train', 'metadata_help': metadata_help})
            train_gen = data.DataLoader(train_data, batch_size=config.train.batch_size,
                                        shuffle=True, collate_fn=train_data.collate, drop_last=True)
            val_data = Dataset({'files': val, 'mode': 'train', 'metadata_help': metadata_help})
            val_gen = data.DataLoader(val_data, batch_size=config.train.batch_size,
                                        shuffle=True, collate_fn=val_data.collate, drop_last=True)

            for batch_number, features in enumerate(train_gen):
                spectrograms = features['spectrograms']
                phones = features['phones']
                input_lengths = features['input_lengths']
                target_lengths = features['target_lengths']
                metadata = features["metadata"]
                # batch_speakers = [x['speaker'] for x in metadata]
                self.G = self.G.train()

                """Make input_lengths and target_lengths torch ints"""
                input_lengths = input_lengths.to(torch.int32)
                target_lengths = target_lengths.to(torch.int32)
                phones = phones.to(torch.int32)

                spectrograms = spectrograms.to(self.torch_type)

                outputs = self.G(spectrograms)

                outputs = outputs.permute(1, 0, 2)  # swap batch and sequence length dimension for CTC loss

                loss = self.ctc_loss(log_probs=outputs, targets=phones,
                                     input_lengths=input_lengths, target_lengths=target_lengths)

                """Update the loss history"""
                # self.update_history(loss, batch_speakers)
                # if epoch >= config.train.regular_epochs:
                #     loss_weights = self.get_loss_weights(batch_speakers, type='fair')
                # else:
                #     loss_weights = self.get_loss_weights(batch_speakers, type='unfair')
                # loss = loss * loss_weights

                # Backward and optimize.
                self.reset_grad()
                loss.backward()
                # loss.sum().backward()
                self.g_optimizer.step()

                if iterations % self.log_step == 0:
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



if __name__ == "__main__":
    main()