import torch.nn as nn
import torch.nn.functional as F
import torch

class DAMH(nn.Module):
    def __init__(self, config):
        super(DAMH, self).__init__()
        self.config = config
        self.num_mels = config.data.num_mels
        self.batch_first = config.model.batch_first
        """Layers"""
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64,
                               kernel_size=(7, 7), stride=2, padding=3)
        self.max_pooling = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        self.conv2_x = []
        self.conv2_x = nn.Sequential(block(num_filters=64, prev_out_channels=64),
                                     block(num_filters=64, prev_out_channels=64),
                                     block(num_filters=64, prev_out_channels=64))
        self.conv3_x = nn.Sequential(block(num_filters=128, prev_out_channels=64),
                                     block(num_filters=128, prev_out_channels=128),
                                     block(num_filters=128, prev_out_channels=128),
                                     block(num_filters=128, prev_out_channels=128))
        self.conv4_x = nn.Sequential(block(num_filters=256, prev_out_channels=128),
                                     block(num_filters=256, prev_out_channels=256),
                                     block(num_filters=256, prev_out_channels=256),
                                     block(num_filters=256, prev_out_channels=256),
                                     block(num_filters=256, prev_out_channels=256),
                                     block(num_filters=256, prev_out_channels=256))
        self.conv5_x = nn.Sequential(block(num_filters=512, prev_out_channels=256),
                                     block(num_filters=512, prev_out_channels=512),
                                     block(num_filters=512, prev_out_channels=512))
        self.conv6 = nn.Conv2d(kernel_size=(16, 1), in_channels=512,
                               out_channels=512, stride=1)
        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.hash_layer = nn.Linear(in_features=512, out_features=config.model.binary_embedding_length)
        self.classification_layer = nn.Linear(in_features=config.model.binary_embedding_length,
                                              out_features=config.vctk.num_speakers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pooling(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.conv6(x)
        x = self.avg_pooling(x)
        x = x.squeeze()
        x = self.hash_layer(x)
        x = F.tanh(x)
        hash_outputs = x
        binary_x = torch.sign(x)
        x = self.classification_layer(x)
        return x, hash_outputs, binary_x, self.classification_layer.weight

class block(nn.Module):
    def __init__(self, num_filters, prev_out_channels):
        """As this is, these parameters don't make it to the GPU which is of course causing problems"""
        super(block, self).__init__()
        self.num_filters = num_filters
        self.prev_out_channels = prev_out_channels
        self.layer = nn.Sequential(nn.Conv2d(kernel_size=(3, 3), out_channels=num_filters,
                                   in_channels=prev_out_channels, stride=1, padding=1),
                                   nn.BatchNorm2d(num_features=num_filters),
                                   nn.ReLU(),
                                   nn.Conv2d(kernel_size=(3, 3), out_channels=num_filters,
                                             in_channels=num_filters, stride=1, padding=1),
                                   nn.BatchNorm2d(num_features=num_filters))
        # self.layer1 = nn.Conv2d(kernel_size=(3, 3), out_channels=num_filters,
        #                         in_channels=prev_out_channels, stride=1, padding=1)
        # self.batchnorm1 = nn.BatchNorm2d(num_features=num_filters)
        # self.layer2 = nn.Conv2d(kernel_size=(3, 3), out_channels=num_filters,
        #                         in_channels=num_filters, stride=1, padding=1)
        # self.batchnorm2 = nn.BatchNorm2d(num_features=num_filters)
    def forward(self, x):
        if self.num_filters == self.prev_out_channels:
            x = self.layer(x) + x  # needs to be a residual block
        else:
            x = self.layer(x)
        # x = self.layer1(x)
        # x = self.batchnorm1(x)
        # x = F.relu_(x)
        # x = self.layer2(x)
        # x = self.batchnorm2(x)
        return x