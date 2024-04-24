# A Standard VAE

"""NICE model
"""

import torch
import torch.nn as nn

import params


class M1_VAE(nn.Module):
    def __init__(self,
                 latent_dim,
                 device,
                 convolutional_layers_encoder,
                 convolutional_layers_decoder,
                 sample_space_flatten,
                 sample_space):
        """Initialize a VAE.

        Args:
            latent_dim: dimension of embedding
            device: run on cpu or gpu
        """
        super(M1_VAE, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.sample_space_flatten = sample_space_flatten
        self.sample_space = sample_space
        self.encoder = nn.Sequential(
            nn.Conv2d(*convolutional_layers_encoder[0:4]),  # B,  32, 28, 28
            nn.ReLU(True),
            nn.Conv2d(*convolutional_layers_encoder[5:9]),  # B,  32, 14, 14
            nn.ReLU(True),
            nn.Conv2d(*convolutional_layers_encoder[10:14]),  # B,  64,  7, 7
        )

        self.mu = nn.Linear(sample_space_flatten, latent_dim)
        self.logvar = nn.Linear(sample_space_flatten, latent_dim)

        self.upsample = nn.Linear(latent_dim, sample_space_flatten)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(*convolutional_layers_decoder[0:4]),  # B,  64,  14,  14
            nn.ReLU(True),
            nn.ConvTranspose2d(*convolutional_layers_decoder[5:10]),  # B,  32, 28, 28
            nn.ReLU(True),
            nn.ConvTranspose2d(*convolutional_layers_decoder[11:15]),  # B, 1, 28, 28
            nn.Sigmoid()
        )

    def sample(self, sample_size, mu=None, logvar=None):
        '''
        :param sample_size: Number of samples
        :param mu: z mean, None for prior (init with zeros)
        :param logvar: z logstd, None for prior (init with zeros)
        :return:
        '''
        if mu is None:
            mu = torch.zeros((sample_size, self.latent_dim)).to(self.device)
        if logvar is None:
            logvar = torch.zeros((sample_size, self.latent_dim)).to(self.device)

        up_sampled = self.z_sample(mu, logvar)
        decoded_images = self.decoder(up_sampled)
        return decoded_images

    def z_sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        latent_result = mu + eps * std
        upsampled = self.upsample(latent_result)
        upsampled = upsampled.view(-1, self.image_sapce)
        return upsampled

    def loss(self, x, recon, mu, logvar):
        reproduction_loss = nn.functional.binary_cross_entropy(recon, x, reduction='sum')
        KLD = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return reproduction_loss + KLD

    def forward(self, x):
        encoded_image = self.encoder(x)

        mu = self.mu(encoded_image.view(-1, self.sample_space))
        logvar = self.logvar(encoded_image.view(-1, self.sample_space))
        z = self.z_sample(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
