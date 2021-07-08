import torch
import torch.nn as nn
from model.modules import *
import math

class Snail(nn.Module):
    def __init__(self, input_size, seq_length):
        super().__init__()

        self.input_size = input_size
        self.seq_length = seq_length

        in_filters = input_size

        add_filters = 32
        self.attn1 = AttentionBlock(input_size=in_filters, key_size=64, value_size=add_filters)
        in_filters += add_filters

        add_filters = 128
        self.tc1 = TCBlock(seq_len=self.seq_length, in_filters=in_filters, filters=add_filters)
        in_filters += self.tc1.no_layer*add_filters

        add_filters = 128
        self.attn2 = AttentionBlock(input_size=in_filters, key_size=256, value_size=add_filters)
        in_filters += add_filters

        add_filters = 128
        self.tc2 = TCBlock(seq_len=self.seq_length, in_filters=in_filters, filters=add_filters)
        in_filters += self.tc2.no_layer*add_filters

        add_filters = 256
        self.attn3 = AttentionBlock(input_size=in_filters, key_size=512, value_size=add_filters)
        in_filters += add_filters

        self.out_filters = in_filters

    def forward(self, input):
        # input should be (time x filters)
        assert(len(input.shape) == 2)

        x = self.attn1(input)
        x = self.tc1(x)
        x = self.attn2(x)
        x = self.tc2(x)
        x = self.attn3(x)
        return x

class Model(nn.Module):
    def __init__(self, snail_kwargs, action_size):
        super().__init__()

        self.snail = Snail(**snail_kwargs)

        add_filters = 256
        self.behav_repr = AttentionBlock(input_size=self.snail.out_filters, key_size=512, value_size=add_filters)
        self.task_repr = AttentionBlock(input_size=self.snail.out_filters, key_size=512, value_size=add_filters)
        in_filters = self.snail.out_filters + add_filters

        hidden_size = 256

        self.value_net = nn.Sequential(
                nn.Linear(in_filters*2, hidden_size), 
                nn.ReLU(),
                nn.Linear(hidden_size, action_size), 
                )
        self.policy_net = nn.Sequential(
                nn.Linear(in_filters*2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, action_size), 
                nn.Softmax()
                )

        self.token_types = ['demo_end']

    def create_special_token(self, type):
        assert type in self.token_types

        if type == 'demo_end':
            z = torch.zeros(self.snail.input_size,dtype=torch.float32)
            z[-1] = 1.0
            return z

    def forward(self, demo, obs):
        demo_seq_len = demo.shape[0]
        obs_seq_len = obs.shape[0]

        demo_end_token = self.create_special_token('demo_end')
        demo = torch.cat((demo, demo_end_token.unsqueeze(0)), dim = 0)
        demo_obs = torch.cat((demo,obs), dim=0)

        snail_output = self.snail(demo_obs)

        behav_repr = self.behav_repr(snail_output[:demo_seq_len+1]) # +1 because of the added demo_end_token
        behav_repr = behav_repr[-1,:] # get the output of demo_end_token

        task_repr = self.task_repr(snail_output)
        task_repr = task_repr[-obs_seq_len:,:]

        behav_repr_tile = behav_repr.unsqueeze(0).tile((obs_seq_len, 1))
        context_vectors = torch.cat((task_repr,behav_repr_tile), dim = 1)

        actions = self.policy_net(context_vectors)
        values = self.value_net(context_vectors)
        
        return actions, values

class VAE(nn.Module):
    def __init__(self, latent_size, input_size):
        super().__init__()

        stride = 2
        kernel_size = 4
        conv1_channels = 32
        conv2_channels = 64
        conv3_channels = 64
        conv4_channels = 64
        padding = 1

        # Encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d( in_channels=3, out_channels=conv1_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(conv1_channels),
            nn.LeakyReLU(0.2, inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv2d( in_channels=conv1_channels, out_channels=conv2_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(conv2_channels),
            nn.LeakyReLU(0.2, inplace=True))

        self.conv3 = nn.Sequential(
            nn.Conv2d( in_channels=conv2_channels, out_channels=conv3_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(conv3_channels),
            nn.LeakyReLU(0.2, inplace=True))

        self.conv4 = nn.Sequential(
            nn.Conv2d( in_channels=conv3_channels, out_channels=conv4_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(conv4_channels),
            nn.LeakyReLU(0.2, inplace=True))

        conv_out = self.conv_layers(torch.rand(size=input_size))
        self.conv_shape = conv_out.shape
        self.flat_size = conv_out.flatten().shape[0]

        self.mean = nn.Sequential(
            nn.Linear(self.flat_size, latent_size),
            nn.Sigmoid(),)

        self.var = nn.Sequential(
            nn.Linear(self.flat_size, latent_size),
            nn.Sigmoid(),)

        # for decoder
        self.linear = nn.Linear(latent_size, self.flat_size)

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d( in_channels=conv4_channels, out_channels=conv3_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(conv3_channels),
            nn.ReLU(inplace=True))

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d( in_channels=conv3_channels, out_channels=conv2_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(conv2_channels),
            nn.ReLU(inplace=True))

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d( in_channels=conv2_channels, out_channels=conv1_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(conv1_channels),
            nn.ReLU(inplace=True))

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d( in_channels=conv1_channels, out_channels=3, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))

    def conv_layers(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x
    
    def deconv_layer(self, x):
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        return x

    def encode(self, x):
        x = self.conv_layers(x)
        # Flatten and apply sigmoid
        x = x.reshape(-1, self.flat_size)
        return self.mean(x), self.var(x)

    def reparameterize(self, mu, logvar):
        # 0.5 for square root (variance to standard deviation)
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        # Project and reshape
        x = self.linear(z)
        x = x.view(x.shape[0], self.conv_shape[1], self.conv_shape[2], self.conv_shape[3])
        x = self.deconv_layer(x)
        return torch.sigmoid(x)

    def forward(self, x, mode='train'):
        if mode == 'train':
            self.train()
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            recon = self.decode(z)
            return recon, mu, logvar
        elif mode == 'inference':
            self.eval()
            mu, logvar = self.encode(x)
            return mu
