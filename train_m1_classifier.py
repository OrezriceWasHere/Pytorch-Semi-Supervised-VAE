import uuid
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import torch, torchvision
from torchvision import transforms
import clearml_interface
from tqdm import tqdm
import params
from m1_vae import M1_VAE, M1_VAE_Classifier
import os


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


def produce_images(vae, epoch, filename):
    sample_shape = [1, 28, 28]

    samples = vae.sample(sample_size=100).cpu()
    a, b = samples.min(), samples.max()
    samples = (samples - a) / (b - a + 1e-10)
    samples = samples.view(-1, sample_shape[0], sample_shape[1], sample_shape[2])
    write_file = f'./samples/{filename}/epoch{epoch}.png'
    os.makedirs(os.path.dirname(write_file), exist_ok=True)
    torchvision.utils.save_image(torchvision.utils.make_grid(samples), write_file)
    image = Image.open(write_file)
    clearml_interface.clearml_display_image(image, epoch, f'epoch{epoch}', series='generation_images')


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

        # if index % supervised_training_factor == 0:
        #     labels = labels.to(vae.device)
        #     latent = vae.reparametize(mu, logvar)
        #     latent = latent.detach().clone()
        #     preds = classifier_model(latent)
        #     predictions.extend(torch.argmax(preds, dim=-1).cpu().tolist())
        #     ground_truth.extend(labels.cpu().tolist())
        #     classification_loss = classifier_model.loss(preds, labels)
        #     classifier_optmizer.zero_grad()
        #     classification_loss.backward()
        #     classifier_optmizer.step()
        #     classification_loss_list.append(classification_loss.item() / batch_size)

    return {
        "avg_gen_loss": sum(generation_loss_list) / len(generation_loss_list)
        # "avg_cls_loss": sum(classification_loss_list) / len(classification_loss_list),
        # "predictions": predictions,
        # "ground_truth": ground_truth
    }


def test_generation(vae, testloader, epoch, filename):
    vae.eval()
    gen_loss_list, cls_loss_list = [], []
    produce_images(vae, epoch, filename)
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

    epochs = params.Params.EPOCHS

    m1_model = \
        M1_VAE(latent_dim=params.Params.LATENT_DIM,
               device=device,
               convolutional_layers_encoder=params.Params.ENCODER_CONVOLUTIONS,
               convolutional_layers_decoder=params.Params.DECODER_CONVOLUTIONS,
               sample_space_flatten=params.Params.SAMPLE_SPACE_FLATTEN,
               sample_space=params.Params.SAMPLE_SPACE).to(device)

    filename = str(uuid.uuid4())
    print('images uuid: ', filename)

    optimizer = torch.optim.Adam(m1_model.parameters(), lr=params.Params.LR)

    classifier_model = M1_VAE_Classifier(latent_dim=params.Params.LATENT_DIM, num_of_classes=10).to(device)
    classifier_optimizer = torch.optim.Adam(
        classifier_model.parameters(), lr=params.Params.LR)

    supervised_training_factor = params.Params.SUPERVISED_DATASET_FACTOR

    for e in tqdm(range(epochs)):
        train_report = train_generation(m1_model,   trainloader, optimizer)
        test_report = test_generation(m1_model,  testloader, e, filename)
        log_results_generation(train_report, test_report, e)

    for e in tqdm(range(epochs)):
        train_report = train_classification(m1_model, classifier_model, trainloader, classifier_optimizer, supervised_training_factor)
        test_report = test_classification(m1_model, classifier_model, testloader)
        log_results_classification(train_report, test_report, e)


if __name__ == "__main__":
    main()
