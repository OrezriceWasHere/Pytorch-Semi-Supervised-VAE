import copy
from torch.utils.data import DataLoader
from PIL import Image
import torch, torchvision
from torchvision import transforms
import clearml_interface
import params
import os
import numpy as np


def with_noise(x):
    return x + torch.zeros_like(x).uniform_(0., 1. / 256.)


def get_dataset(dataset_name='mnist'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(with_noise),  # dequantization
        transforms.Normalize((0.,), (257. / 256.,)),  # rescales to [0,1]
    ])

    if dataset_name == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data/MNIST',
                                              train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data/MNIST',
                                             train=False, download=True, transform=transform)

    return trainset, testset


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def produce_images(vae, epoch, filename):
    sample_shape = params.Params.SAMPLE_SPACE

    samples = vae.sample(sample_size=100).cpu()
    a, b = samples.min(), samples.max()
    samples = (samples - a) / (b - a + 1e-10)
    samples = samples.view(-1, *sample_shape)
    write_file = f'./samples/{filename}/epoch{epoch}.png'
    os.makedirs(os.path.dirname(write_file), exist_ok=True)
    torchvision.utils.save_image(torchvision.utils.make_grid(samples), write_file)
    image = Image.open(write_file)
    clearml_interface.clearml_display_image(image, epoch, description='epoch{epoch}', series='generation_images')


def create_semisupervised_datasets(dataset, n_labeled, batch_size):
    # note this is only relevant for training the model
    assert dataset.train == True, 'Dataset must be the training set; assure dataset.train = True.'

    # compile new x and y and replace the dataset.train_data and train_labels with the
    x = dataset.train_data
    y = dataset.train_labels
    n_x = x.shape[0]
    n_classes = len(torch.unique(y))

    assert n_labeled % n_classes == 0, 'n_labeld not divisible by n_classes; cannot assure class balance.'
    n_labeled_per_class = n_labeled // n_classes

    x_labeled = [0] * n_classes
    x_unlabeled = [0] * n_classes
    y_labeled = [0] * n_classes
    y_unlabeled = [0] * n_classes

    for i in range(n_classes):
        idxs = (y == i).nonzero().data.numpy()
        np.random.shuffle(idxs)

        x_labeled[i] = x[idxs][:n_labeled_per_class]
        y_labeled[i] = y[idxs][:n_labeled_per_class]
        x_unlabeled[i] = x[idxs][n_labeled_per_class:]
        y_unlabeled[i] = y[idxs][n_labeled_per_class:]

    # construct new labeled and unlabeled datasets
    labeled_dataset = copy.deepcopy(dataset)
    labeled_dataset.data = torch.cat(x_labeled, dim=0).squeeze()
    labeled_dataset.targets = torch.cat(y_labeled, dim=0)

    unlabeled_dataset = copy.deepcopy(dataset)
    unlabeled_dataset.data = torch.cat(x_unlabeled, dim=0).squeeze()
    unlabeled_dataset.targets = torch.cat(y_unlabeled, dim=0)

    del dataset

    labeled_loader = torch.utils.data.DataLoader(labeled_dataset,
                                              batch_size=batch_size, shuffle=True, num_workers=2)
    unlabeled_loader = torch.utils.data.DataLoader(unlabeled_dataset,
                                                batch_size=batch_size, shuffle=True, num_workers=2)
    return labeled_loader, unlabeled_loader
