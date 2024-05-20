import torch.nn.functional as F
import uuid
import torch, torchvision
from torchvision import transforms
import params
import os
from torchvision.utils import save_image
from torch.nn.functional import one_hot
from m2_vae_take_two import M2_VAE
from tqdm import tqdm


def train_epoch(model,
                trainloader,
                optimizer,
                supervised_training_factor,
                number_of_classes,
                n_labeled,
                device,
                alpha
                ):
    model.train()

    for index, (inputs, labels) in enumerate(trainloader):
        is_supervised = index % supervised_training_factor == 0

        # get batch from respective dataloader
        if is_supervised:
            x, y = inputs, labels
            y = F.one_hot(y, num_classes=number_of_classes).float().to(device)
        else:
            x, y = inputs, labels
            y = None
        x = x.to(device).view(x.shape[0], -1)

        # compute loss -- SSL paper eq 6, 7, 9
        q_y = model.encode_y(x)
        # labeled data loss -- SSL paper eq 6 and eq 9

        if y is not None:
            q_z_xy = model.encode_z(x, y)
            z = q_z_xy.rsample()
            p_x_yz = model.decode(y, z)
            loss = M2_VAE.loss_components_fn(x, y, z, model.p_y, model.p_z, p_x_yz, q_z_xy)
            loss -= alpha * n_labeled * q_y.log_prob(y)  # SSL eq 9
        # unlabeled data loss -- SSL paper eq 7
        else:
            # marginalize y according to q_y
            loss = - q_y.entropy()
            for y in q_y.enumerate_support():
                q_z_xy = model.encode_z(x, y)
                z = q_z_xy.rsample()
                p_x_yz = model.decode(y, z)
                L_xy = M2_VAE.loss_components_fn(x, y, z, model.p_y, model.p_z, p_x_yz, q_z_xy)
                loss += q_y.log_prob(y).exp() * L_xy
        loss = loss.mean(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def with_noise(x):
    return x + torch.zeros_like(x).uniform_(0., 1. / 256.)


def get_dataset(batch_size, dataset_name='mnist'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(with_noise),  # dequantization
        transforms.Normalize((0.,), (257. / 256.,)),  # rescales to [0,1]
    ])

    if dataset_name == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data/MNIST',
                                              train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.MNIST(root='./data/MNIST',
                                             train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    device = get_device()
    batch_size = params.Params.BATCH_SIZE
    trainloader, testloader = get_dataset(batch_size, 'mnist')

    epochs = params.Params.EPOCHS

    m2_model = \
        M2_VAE(
            number_of_classes=params.Params.NUMBER_OF_CLASSES,
            sample_space=params.Params.SAMPLE_SPACE,
            latent_space=params.Params.LATENT_DIM,
            hidden_dim=params.Params.HIDDEN_DIM,
            device=device
        ).to(device)

    filename = str(uuid.uuid4())
    print('images uuid: ', filename)

    optimizer = torch.optim.Adam(m2_model.parameters(), lr=params.Params.LR)
    n_labeled = 60000 / params.Params.SUPERVISED_DATASET_FACTOR

    for e in tqdm(range(epochs)):
        train_epoch(model=m2_model,
                    trainloader=trainloader,
                    optimizer=optimizer,
                    supervised_training_factor=params.Params.SUPERVISED_DATASET_FACTOR,
                    number_of_classes=params.Params.NUMBER_OF_CLASSES,
                    device=device,
                    alpha=params.Params.ALPHA,
                    n_labeled=n_labeled
                    )

        produce_images(model=m2_model,
                       number_of_classes=params.Params.NUMBER_OF_CLASSES,
                       image_space=[1, 28, 28],
                       output_dir='./samples',
                       device=device)


def produce_images(model,
                   number_of_classes, 
                   image_space,
                   output_dir,
                   device,
                   ):

    for y in range(2, 5):
        y = one_hot(torch.tensor(y).unsqueeze(-1), number_of_classes).expand(100, number_of_classes).to(device)

        # order the first dim of the z latent
        # c = torch.linspace(-5, 5, 100).view(-1, 1).repeat(1, 10)

        c = torch.zeros((100, 10))
        z = torch.cat([c, torch.zeros(c.shape[0], c.shape[1] *4 ) ], dim=1).reshape(100, 50).to(device)

        # combine into z and pass through decode(y, z)
        x = model.decode(y, z).sample().view(y.shape[0], *image_space)
        save_image(x.cpu(),
                   os.path.join(output_dir, 'latent_var_grid_sample_c1_y{}.png'.format(y[0].nonzero().item())),
                   nrow=10)

        # order second dim of latent and pass through decoder
        z = z.flip(1)
        x = model.decode(y, z).sample().view(y.shape[0], *image_space)
        save_image(x.cpu(),
                   os.path.join(output_dir, 'latent_var_grid_sample_c2_y{}.png'.format(y[0].nonzero().item())),
                   nrow=10)


if __name__ == "__main__":
    main()
