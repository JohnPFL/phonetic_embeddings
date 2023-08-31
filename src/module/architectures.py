

import torch
from torch import nn
import torch.nn as nn
import torch.nn.functional as F

class CNNlayer(nn.Module):

    def __init__(self, out_channels=128, n_class=1, embedding_dim=32, kernel_heights=[3, 5, 6], stride=1, padding=0,
                 drop_prob=0.5):
        super(CNNlayer, self).__init__()

        in_channels = 1

        # implementing weight normalization on convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, (kh, embedding_dim), stride, padding) for kh in kernel_heights
        ])
        self.dropout = nn.Dropout(drop_prob)
        self.out_dimensions = len(kernel_heights) * out_channels

        self.linear = nn.Linear(self.out_dimensions, n_class)
        self.xavier_uniform()


    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)  # conv_out.size() = (batch_size, out_channels, dim, 1)
        #activation = F.relu(conv_out.squeeze(3))  # activation.size() = (batch_size, out_channels, dim1)
        lr = nn.LeakyReLU()
        activation = lr(conv_out.squeeze(3))
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)  # maxpool_out.size() = (batch_size, out_channels)
        return max_out

    def forward(self, input):   # input.size() = (batch_size, num_seq, embedding_dim)
        input = input.unsqueeze(1)  # input.size() = (batch_size, 1, num_seq, embedding_length)

        max_outs = [self.conv_block(input, conv) for conv in self.convs]

        all_out = torch.cat(max_outs, 1)  # all_out.size() = (batch_size, num_kernels*out_channels)
        fc_in = self.dropout(all_out)  # fc_in.size()) = (batch_size, num_kernels*out_channels)
        out = self.linear(fc_in).view(-1)
        return out

    def xavier_uniform(self):
        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad:
                nn.init.xavier_uniform_(p)
        return self

    def dim(self):
        return self.out_dimensions

class CNNbase(nn.Module):

    def __init__(self, vocab_size, embedding_dim=32, out_channels=128, 
                 n_class=1, kernel_heights=[3, 6, 12], stride=1,
                 padding=0, drop_prob=0.7, tmp_path = '', def_path = ''):

        super(CNNbase, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.cnn_layer = CNNlayer(
            out_channels=out_channels,
            n_class=n_class,
            embedding_dim=embedding_dim,
            kernel_heights=kernel_heights,
            stride=stride,
            padding=padding,
            drop_prob=drop_prob)
        
        self.xavier_uniform()
        
    def forward(self, input):   # input.size() = (batch_size, num_seq)
        input = self.embedding(input)  # input.size() = (batch_size, num_seq, embedding_dim)
        out = self.cnn_layer(input)
        return out

    def xavier_uniform(self):
        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad:
                nn.init.xavier_uniform_(p)
        return self

class PhoneticCNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim=32, 
                 out_channels=128, n_class=1, kernel_heights=[3, 5, 6], stride=1,
                 padding=0, drop_prob=0.5, tmp_path = '', def_path = ''):

        super(PhoneticCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.cnn_layer = CNNlayer(
            out_channels=out_channels,
            n_class=n_class,
            embedding_dim=embedding_dim,
            kernel_heights=kernel_heights,
            stride=stride,
            padding=padding,
            drop_prob=drop_prob)

    def forward(self, input):   # input.size() = (batch_size, num_seq)
        input = self.embedding(input)  # input.size() = (batch_size, num_seq, embedding_dim)
        out = self.cnn_layer(input)
        return out

    def xavier_uniform(self):
        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad:
                nn.init.xavier_uniform_(p)
        return self
    

