import torch, torchvision
from torchvision import transforms
from tqdm import tqdm
import params
from m1_vae import M1_VAE


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


def train(vae, trainloader, optimizer):
    vae.train()  # set to training mode
    loss_list = []
    for inputs, _ in trainloader:
        batch_size = inputs.size(0)
        inputs = inputs.to(vae.device)
        outputs, mu, logvar = vae(inputs)
        loss = vae.loss(inputs, outputs, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item() / batch_size)

    return sum(loss_list) / len(loss_list)


def test(vae, testloader):
    vae.eval()
    loss_list = []
    for inputs, _ in testloader:
        batch_size = inputs.size(0)
        inputs = inputs.to(vae.device)
        outputs, mu, logvar = vae(inputs)
        loss = vae.loss(inputs, outputs, mu, logvar)
        loss_list.append(loss.item() / batch_size)
    return sum(loss_list) / len(loss_list)


def main():
    device = get_device()
    batch_size = params.Params.BATCH_SIZE
    trainloader, testloader = get_dataset(batch_size, 'mnist')

    epochs = params.Params.EPOCHS

    m1_model = \
        M1_VAE(latent_dim=params.Params.LATENT_DIM,
               device=device,
               convolutional_layers_encoder=params.Params.ENCODER_CONVOLUTIONS,
               convolutional_layers_decoder=params.Params.DECODER_CONVOLUTIONS,
               sample_space_flatten=params.Params.SAMPLE_SPACE_FLATTEN,
               sample_space=params.Params.SAMPLE_SPACE).to(device)

    optimizer = torch.optim.Adam(
        m1_model.parameters(), lr=params.Params.LR)

    for e in tqdm.tqdm(range(epochs)):
        train_loss = train(m1_model, trainloader, optimizer)
        test_loss = test(m1_model, testloader)


if __name__ == "__main__":
    main()
