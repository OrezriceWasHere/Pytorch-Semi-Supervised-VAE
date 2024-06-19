# A Standard VAE

"""NICE model
"""


from torch.nn import Module
import torch
import torch.nn as nn
import torch.distributions as D
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



class M2_VAE(Module):
    """
    Data model (SSL paper eq 2):
        p(y) = Cat(y|pi)
        p(z) = Normal(z|0,1)
        p(x|y,z) = f(x; z,y,theta)

    Recognition model / approximate posterior q_phi (SSL paper eq 4):
        q(y|x) = Cat(y|pi_phi(x))
        q(z|x,y) = Normal(z|mu_phi(x,y), diag(sigma2_phi(x)))


    """

    def __init__(self,
                 number_of_classes,
                 m1_space,
                 latent_space,
                 hidden_space,
                 device
                 ):
        super(M2_VAE, self).__init__()
        # C, H, W = sample_space
        # x_space = C * H * W
        # --------------------
        # p model -- SSL paper generative semi supervised model M2
        # --------------------

        self.p_y = D.OneHotCategorical(probs=1 / number_of_classes * torch.ones(1, number_of_classes, device=device))
        self.p_z = D.Normal(torch.tensor(0., device=device), torch.tensor(1., device=device))

        # parametrized data likelihood p(x|y,z)
        self.decoder = nn.Sequential(nn.Linear(latent_space + number_of_classes, hidden_space),
                                     nn.Softplus(),
                                     nn.Linear(hidden_space, hidden_space),
                                     nn.Softplus(),
                                     nn.Linear(hidden_space, m1_space))

        # --------------------
        # q model -- SSL paper eq 4
        # --------------------

        # parametrized q(y|x) = Cat(y|pi_phi(x)) -- outputs parametrization of categorical distribution
        self.encoder_y = nn.Sequential(nn.Linear(m1_space, hidden_space),
                                       nn.Softplus(),
                                       nn.Linear(hidden_space, hidden_space),
                                       nn.Softplus(),
                                       nn.Linear(hidden_space, number_of_classes))

        # parametrized q(z|x,y) = Normal(z|mu_phi(x,y), diag(sigma2_phi(x))) -- output parametrizations for mean and diagonal variance of a Normal distribution
        self.encoder_z = nn.Sequential(nn.Linear(m1_space + number_of_classes, hidden_space),
                                       nn.Softplus(),
                                       nn.Linear(hidden_space, hidden_space),
                                       nn.Softplus(),
                                       nn.Linear(hidden_space, 2 * latent_space))

        # initialize weights to N(0, 0.001) and biases to 0 (cf SSL section 4.4)
        for p in self.parameters():
            p.data.normal_(0, 0.001)
            if p.ndimension() == 1: p.data.fill_(0.)

    # q(z|x,y) = Normal(z|mu_phi(x,y), diag(sigma2_phi(x))) -- SSL paper eq 4
    def encode_z(self, x, y):
        xy = torch.cat([x, y], dim=1)
        mu, logsigma = self.encoder_z(xy).chunk(2, dim=-1)
        return D.Normal(mu, logsigma.exp())

    # q(y|x) = Categorical(y|pi_phi(x)) -- SSL paper eq 4
    def encode_y(self, x):
        return D.OneHotCategorical(logits=self.encoder_y(x))

    # p(x|y,z) = Bernoulli
    def decode(self, y, z):
        yz = torch.cat([y, z], dim=1)

        return D.continuous_bernoulli.ContinuousBernoulli(logits=self.decoder(yz))

    # classification model q(y|x) using the trained q distribution
    def forward(self, x):
        y_probs = self.encode_y(x).probs
        return y_probs.max(dim=1)[1]  # return pred labels = argmax

    @staticmethod
    def loss_components_fn(x, y, z, p_y, p_z, p_x_yz, q_z_xy):
        # SSL paper eq 6 for an given y (observed or enumerated from q_y)
        return - p_x_yz.log_prob(abs(x)/10).sum(1) \
            - p_y.log_prob(y) \
            - p_z.log_prob(z).sum(1) \
            + q_z_xy.log_prob(z).sum(1)
