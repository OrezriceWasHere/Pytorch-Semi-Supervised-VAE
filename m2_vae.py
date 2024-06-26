from torch.nn import Module
import torch
import torch.nn as nn
import torch.distributions as D


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
                 sample_space,
                 latent_space,
                 hidden_space,
                 device
                 ):
        super().__init__()
        C, H, W = sample_space
        x_space = C * H * W

        # --------------------
        # p model -- SSL paper generative semi supervised model M2
        # --------------------

        self.p_y = D.OneHotCategorical(probs=1 / number_of_classes * torch.ones(1, number_of_classes, device=device))
        self.p_z = D.Normal(torch.tensor(0., device=device), torch.tensor(1., device=device))

        # parametrized data likelihood p(x|y,z)
        self.decoder = nn.Sequential(
            nn.Linear(latent_space + number_of_classes, hidden_space),
            nn.Softplus(),
            nn.Linear(hidden_space, hidden_space),
            nn.Softplus(),
            nn.Linear(hidden_space, x_space))

        # --------------------
        # q model -- SSL paper eq 4
        # --------------------

        # parametrized q(y|x) = Cat(y|pi_phi(x)) -- outputs parametrization of categorical distribution
        self.encoder_y = nn.Sequential(
            nn.Linear(x_space, hidden_space),
            nn.Softplus(),
            nn.Linear(hidden_space, hidden_space),
            nn.Softplus(),
            nn.Linear(hidden_space, number_of_classes))

        # parametrized q(z|x,y) = Normal(z|mu_phi(x,y), diag(sigma2_phi(x))) -- output parametrizations for mean and diagonal variance of a Normal distribution
        self.encoder_z = nn.Sequential(
            nn.Linear(x_space + number_of_classes, hidden_space),
            nn.Softplus(),
            nn.Linear(hidden_space, hidden_space),
            nn.Softplus(),
            nn.Linear(hidden_space, 2 * latent_space))


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
        return - p_x_yz.log_prob(x).sum(1) \
            - p_y.log_prob(y) \
            - p_z.log_prob(z).sum(1) \
            + q_z_xy.log_prob(z).sum(1)
