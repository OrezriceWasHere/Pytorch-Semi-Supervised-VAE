from torch.nn import Module, Sequential, Conv2d, ReLU, ConvTranspose2d, Sigmoid, Linear, Embedding, BatchNorm2d
import torch

class M2_VAE_Classifier(Module):
    def __init__(self):
        super(M2_VAE_Classifier, self).__init__()
        self.classifier = Sequential(
            torch.nn.Dropout2d(),
            Conv2d(1, 32, 3, 1),
            BatchNorm2d(32),
            
        )

    def forward(self, x):
        return self.classifier(x)

    def loss(self, preds, labels):
        return nn.functional.cross_entropy(input=preds, target=labels)


class M2_VAE(Module):
    def __init__(self,
                 latent_dim,
                 classification_embedding_dim,
                 num_of_classes,
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
        super(M2_VAE, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.classification_embedding_dim = classification_embedding_dim
        self.num_of_classes = num_of_classes
        self.sample_space_flatten = sample_space_flatten
        self.sample_space = sample_space
        self.encoder = Sequential(
            Conv2d(*convolutional_layers_encoder[:5]),
            ReLU(True),
            Conv2d(*convolutional_layers_encoder[5:10]),
            ReLU(True),
            Conv2d(*convolutional_layers_encoder[10:]),
        )

        self.label_embedding = Embedding(num_of_classes, classification_embedding_dim)

        self.mu = Linear(sample_space_flatten + classification_embedding_dim, latent_dim)
        self.logvar = Linear(sample_space_flatten + classification_embedding_dim, latent_dim)

        self.upsample = Linear(latent_dim, sample_space_flatten)
        self.decoder = Sequential(
            ConvTranspose2d(*convolutional_layers_decoder[:5]),
            ReLU(True),
            ConvTranspose2d(*convolutional_layers_decoder[5:11]),
            ReLU(True),
            ConvTranspose2d(*convolutional_layers_decoder[11:]),
            Sigmoid()
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


    def forward_with_classification(self, x, y):
        embedded_y = self.label_embedding(y)

        encoded_image = self.encoder(x)
        encoded_image_flatten = encoded_image.view(-1, self.sample_space_flatten)
        encoded_image_flatten_with_y = torch.cat((encoded_image_flatten, embedded_y), dim=1)
        mu = self.mu(encoded_image_flatten_with_y)
        logvar = self.logvar(encoded_image_flatten_with_y)
        z = self.reparametize(mu, logvar)



