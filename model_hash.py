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

class ResNet18DAMH(nn.Module):
    def __init__(self, config):
        super(ResNet18DAMH, self).__init__()
        self.config = config
        self.num_mels = config.data.num_mels
        self.batch_first = config.model.batch_first

        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=False)

        # self.avg_pooling = nn.AdaptiveAvgPool1d(output_size=(512))
        self.hash_layer = nn.Linear(in_features=1000, out_features=config.model.binary_embedding_length)
        self.classification_layer = nn.Linear(in_features=config.model.binary_embedding_length,
                                              out_features=config.vctk.num_speakers)

    def forward(self, x):
        x = self.model(x)
        # x = self.avg_pooling(x)
        # x = x.squeeze()
        x = self.hash_layer(x)
        x = F.tanh(x)
        hash_outputs = x
        binary_x = torch.sign(x)
        x = self.classification_layer(x)
        return x, hash_outputs, binary_x, self.classification_layer.weight

class LSTM_hasher(nn.Module):
    def __init__(self, config):
        super(LSTM_hasher, self).__init__()
        self.config = config
        self.feature_size = 80  # need this for dimension of output of BLSTM
        self.hidden_size = 400
        self.batch_first = True
        self.dropout = 0.1
        self.bidirectional = True
        self.longest_phone_sequence_len = 15  # THIS IS FOR UASPEECH DATASET
        self.lstm = nn.LSTM(input_size=self.feature_size, hidden_size=self.hidden_size,
                            num_layers=2, batch_first=self.batch_first,
                            dropout=self.dropout, bidirectional=self.bidirectional)
        self.fc1 = nn.Linear(in_features=self.hidden_size*2,
                            out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=500)
        self.fc3 = nn.Linear(in_features=500, out_features=500)
        self.hash_layer = nn.Linear(in_features=500, out_features=config.model.binary_embedding_length)
        self.classification_layer = nn.Linear(in_features=config.model.binary_embedding_length,
                                              out_features=config.vctk.num_speakers)
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
        # forward_summary = x[:, -1, 0, :]
        forward_summary = x[:, seq_len-1, 0, :]
        backward_summary = x[:, 0, 1, :]
        summaries = torch.cat((forward_summary, backward_summary), dim=1)

        x = self.fc1(summaries)
        x = F.relu_(x)
        x = self.fc2(x)
        x = F.relu_(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.hash_layer(x)
        x = F.tanh(x)
        hash_outputs = x
        binary_x = torch.sign(x)
        x = self.classification_layer(x)
        return x, hash_outputs, binary_x, self.classification_layer.weight