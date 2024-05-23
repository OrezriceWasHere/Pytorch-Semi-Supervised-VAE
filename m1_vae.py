# A Standard VAE

"""NICE model
"""

import torch
import torch.nn as nn

class M1_VAE_Classifier(nn.Module):


    def __init__(self, latent_space, num_of_classes):
        super(M1_VAE_Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_space, num_of_classes),
            nn.Softmax(dim=None)
        )

    def forward(self, x):
        return self.classifier(x)

    def loss(self, preds, labels):
        return nn.functional.cross_entropy(input=preds, target=labels)

class M1_VAE(nn.Module):
    def __init__(self,
                 latent_space,
                 device,
                 convolutional_layers_encoder,
                 convolutional_layers_decoder,
                 encoder_decoder_z_space):
        """Initialize a VAE.

        Args:
            latent_space: dimension of embedding
            device: run on cpu or gpu
        """

        super(M1_VAE, self).__init__()
        self.device = device
        self.latent_dim = latent_space
        self.encoder_decoder_z_space = encoder_decoder_z_space

        C, H, W = encoder_decoder_z_space
        encoder_decoder_z_space_flatten = C * H * W
        self.encoder_decoder_z_space_flatten = encoder_decoder_z_space_flatten

        self.encoder = nn.Sequential(
            nn.Conv2d(*convolutional_layers_encoder[:5]),
            nn.ReLU(True),
            nn.Conv2d(*convolutional_layers_encoder[5:10]),
            nn.ReLU(True),
            nn.Conv2d(*convolutional_layers_encoder[10:]),
        )

        self.mu = nn.Linear(encoder_decoder_z_space_flatten, latent_space)
        self.logvar = nn.Linear(encoder_decoder_z_space_flatten, latent_space)

        self.upsample = nn.Linear(latent_space, encoder_decoder_z_space_flatten)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(*convolutional_layers_decoder[:5]),
            nn.ReLU(True),
            nn.ConvTranspose2d(*convolutional_layers_decoder[5:11]),
            nn.ReLU(True),
            nn.ConvTranspose2d(*convolutional_layers_decoder[11:]),
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

    def reparametize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def z_sample(self, mu, logvar):
        latent_result = self.reparametize(mu, logvar)
        upsampled = self.upsample(latent_result)
        upsampled = upsampled.view(-1, *self.encoder_decoder_z_space)
        return upsampled

    def loss(self, x, recon, mu, logvar):
        reproduction_loss = nn.functional.binary_cross_entropy(recon, x, reduction='sum')
        KLD = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return reproduction_loss + KLD

    def forward(self, x):
        encoded_image = self.encoder(x)

        mu = self.mu(encoded_image.view(-1, self.encoder_decoder_z_space_flatten))
        logvar = self.logvar(encoded_image.view(-1, self.encoder_decoder_z_space_flatten))
        upsampled_z = self.z_sample(mu, logvar)
        recon = self.decoder(upsampled_z)
        return recon, mu, logvar
