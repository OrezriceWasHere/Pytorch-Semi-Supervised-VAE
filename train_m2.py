from uuid import uuid4
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import torch, torchvision
from torchvision import transforms
import clearml_interface
from tqdm import tqdm
import params
import os
from clearml_interface import clearml_display_image

from m2_vae import M2_VAE


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

    assert trainset, "No Trainset was chosen"

    keep_labels_factor = params.Params.SUPERVISED_DATASET_FACTOR


    testset = torchvision.datasets.MNIST(root='./data/MNIST',
                                             train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=batch_size, shuffle=False, num_workers=2)



    return trainloader, testloader


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def one_hot(x, label_size):
    out = torch.one_hotzeros(len(x), label_size).to(x.device)
    out[torch.arange(len(x)), x.squeeze()] = 1
    return out

def produce_images(vae, epoch, filename, device):
    sample_shape = params.Params.SAMPLE_SPACE
    number_of_classes = params.Params.NUMBER_OF_CLASSES
    
    
    for y in range(0, 9):
        y = one_hot(torch.tensor(y).unsqueeze(-1), number_of_classes).expand(100, number_of_classes).to(device)

        # order the first dim of the z latent
        c = torch.linspace(-5, 5, 10).view(-1,1).repeat(1,10).reshape(-1,1)
        z = torch.cat([c, torch.zeros_like(c)], dim=1).reshape(100, 2).to(device)

        # combine into z and pass through decoder
        x = vae.decode(y, z).sample().view(y.shape[0], *sample_shape)
        clearml_save_image(x.cpu(), f'latent_var_grid_sample_c1_y{y[0].nonzero().item()}')

        # order second dim of latent and pass through decoder
        z = z.flip(1)
        x = vae.decode(y, z).sample().view(y.shape[0], *sample_shape)
        clearml_save_image(x.cpu(), f'latent_var_grid_sample_c2_y{y[0].nonzero().item()}')

def clearml_save_image(tensor, description, series="image_generation"):
    uuid = str(uuid4())
    write_file = f'./samples/uuid{uuid}.png'
    print("creating image at: ", write_file)
    os.makedirs(os.path.dirname(write_file), exist_ok=True)
    torchvision.utils.save_image(torchvision.utils.make_grid(tensor), write_file)
    image = Image.open(write_file)
    clearml_display_image(image, 1 ,description=description, series=series)



def train_generation(vae,
          trainloader,
          optimizer
          ):
    vae.train()  # set to training mode
    generation_loss_list, classification_loss_list = [], []
    for index, (inputs, labels) in enumerate(trainloader):
        batch_size = params.Params.BATCH_SIZE
        inputs = inputs.to(vae.device)

        outputs, mu, logvar = vae(inputs)
        generation_loss = vae.loss(inputs, outputs, mu, logvar)
        generation_loss_list.append(generation_loss.item() / batch_size)
        optimizer.zero_grad()
        generation_loss.backward()
        optimizer.step()

    return {
        "avg_gen_loss": sum(generation_loss_list) / len(generation_loss_list)
    }


def test_generation(vae, testloader, epoch, filename):
    vae.eval()
    gen_loss_list, cls_loss_list = [], []
    # produce_images(vae, epoch, filename)
    for index, (inputs, labels) in enumerate(testloader):
        batch_size = inputs.size(0)
        inputs = inputs.to(vae.device)
        outputs, mu, logvar = vae(inputs)
        gen_loss = vae.loss(inputs, outputs, mu, logvar)
        gen_loss_list.append(gen_loss.item() / batch_size)

    return {
        "avg_gen_loss": sum(gen_loss_list) / len(gen_loss_list)
        # "avg_cls_loss": sum(cls_loss_list) / len(cls_loss_list)
    }

def train_classification(vae, classifier_model, trainloader, optimizer, supervised_training_factor):
    vae.eval()  # set to training mode
    classifier_model.train()
    classification_loss_list = []
    predictions, ground_truth = [], []
    for index, (inputs, labels) in enumerate(trainloader):
        batch_size = inputs.size(0)
        if index % supervised_training_factor == 0:
            inputs = inputs.to(vae.device)
            outputs, mu, logvar = vae(inputs)
            labels = labels.to(vae.device)
            latent = vae.reparametize(mu, logvar)
            latent = latent.detach().clone()
            preds = classifier_model(latent)
            predictions.extend(torch.argmax(preds, dim=-1).cpu().tolist())
            ground_truth.extend(labels.cpu().tolist())
            classification_loss = classifier_model.loss(preds, labels)
            optimizer.zero_grad()
            classification_loss.backward()
            optimizer.step()
            classification_loss_list.append(classification_loss.item() / batch_size)

    return {
        "avg_cls_loss": sum(classification_loss_list) / len(classification_loss_list),
        "predictions": predictions,
        "ground_truth": ground_truth
    }

def test_classification(vae, classifier_model, testloader):
    vae.eval()
    classifier_model.eval()
    cls_loss_list = []
    predictions, ground_truth = [], []
    for index, (inputs, labels) in enumerate(testloader):
        batch_size = inputs.size(0)
        inputs = inputs.to(vae.device)
        outputs, mu, logvar = vae(inputs)
        labels = labels.to(vae.device)
        latent = vae.reparametize(mu, logvar)
        preds = classifier_model(latent)
        predictions.extend(torch.argmax(preds, dim=-1).cpu().tolist())
        ground_truth.extend(labels.cpu().tolist())
        classification_loss = classifier_model.loss(preds, labels)
        cls_loss_list.append(classification_loss.item() / batch_size)

    return {
        "avg_cls_loss": sum(cls_loss_list) / len(cls_loss_list),
        "predictions": predictions,
        "ground_truth": ground_truth
    }

def log_results_generation(train_results, test_results, epoch):
    train_gen_loss = train_results['avg_gen_loss']
    test_gen_loss = test_results['avg_gen_loss']
    clearml_interface.add_point_to_graph('train_gen_loss', 'train_gen_loss', epoch, train_gen_loss)
    clearml_interface.add_point_to_graph('test_gen_loss', 'test_gen_loss', epoch, test_gen_loss)

def log_results_classification(train_results, test_results, epoch):
    train_cls_loss = train_results['avg_cls_loss']
    test_cls_loss = test_results['avg_cls_loss']
    clearml_interface.add_point_to_graph('train_cls_loss', 'train_cls_loss', epoch, train_cls_loss)

    clearml_interface.add_point_to_graph('test_cls_loss', 'test_cls_loss', epoch, test_cls_loss)
    test_matrix = confusion_matrix(test_results['ground_truth'], test_results['predictions'])
    clearml_interface.add_confusion_matrix(test_matrix, 'test_confusion_matrix', 'test_confusion_matrix', epoch)
    train_matrix = confusion_matrix(train_results['ground_truth'], train_results['predictions'])
    clearml_interface.add_confusion_matrix(train_matrix, 'train_confusion_matrix', 'train_confusion_matrix', epoch)
    test_classification_report = classification_report(test_results['ground_truth'], test_results['predictions'])
    clearml_interface.add_text(test_classification_report, 'test_classification_report', epoch)



def main():
    clearml_interface.clearml_init()
    device = get_device()
    batch_size = params.Params.BATCH_SIZE
    trainloader, testloader = get_dataset(batch_size, 'mnist')

    latent_space = params.Params.LATENT_DIM
    number_of_classes = params.Params.NUMBER_OF_CLASSES
    sample_space = params.Params.SAMPLE_SPACE
    
    epochs = params.Params.GENERATION_EPOCHS

    m2_model = \
        M2_VAE(number_of_classes=number_of_classes,
                 sample_space=sample_space,
                 latent_space=latent_space,
                 hidden_space=params.Params.HIDDEN_DIM,
                 device=device,
               ).to(device)

    filename = str(uuid4())
    print('images uuid: ', filename)

    optimizer = torch.optim.Adam(m2_model.parameters(), lr=params.Params.LR)

    for e in tqdm(range(epochs)):
        train_report = train_generation(m2_model,  trainloader, optimizer)
        test_report = test_generation(m2_model, testloader, e, filename)
        log_results_generation(train_report, test_report, e)
        produce_images(m2_model, e, filename, device)



if __name__ == "__main__":
    main()
