import uuid
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import torch, torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import clearml_interface
from tqdm import tqdm
import params
from m1_vae import M1_VAE, M1_VAE_Classifier
from params import Params
from utilities import get_device, get_dataset, produce_images, create_semisupervised_datasets


def train_generation(vae,
                     trainloader,
                     optimizer
                     ):
    vae.train()  # set to training mode
    generation_loss_list, classification_loss_list = [], []
    for index, (inputs, labels) in enumerate(trainloader):
        batch_size = Params.BATCH_SIZE
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


def train_classification(vae, classifier_model, trainloader, optimizer):
    vae.eval()  # set to training mode
    classifier_model.train()
    classification_loss_list = []
    predictions, ground_truth = [], []
    for index, (inputs, labels) in enumerate(trainloader):
        batch_size = inputs.size(0)
        # if index % supervised_training_factor == 0:
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
    print("avg_cls_loss", str(sum(classification_loss_list) / len(classification_loss_list)))

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
    clearml_interface.add_text(test_classification_report,  epoch)


def main():
    clearml_interface.clearml_init()
    device = get_device()
    batch_size = Params.BATCH_SIZE
    trainset, testset = get_dataset('mnist')
    n_labeld = Params.LABELED_COUNT
    labeledloader, unlabeledloader = create_semisupervised_datasets(trainset,
                                                                    n_labeld,
                                                                    batch_size)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    m1_model = \
        M1_VAE(latent_space=Params.LATENT_DIM,
               device=device,
               convolutional_layers_encoder=Params.ENCODER_CONVOLUTIONS,
               convolutional_layers_decoder=Params.DECODER_CONVOLUTIONS,
               encoder_decoder_z_space=Params.ENCODER_DECODER_Z_SPACE).to(device)

    filename = str(uuid.uuid4())
    print('images uuid: ', filename)

    optimizer = torch.optim.Adam(m1_model.parameters(), lr=Params.LR)

    classifier_model = M1_VAE_Classifier(latent_space=Params.LATENT_DIM,
                                         num_of_classes=Params.NUMBER_OF_CLASSES).to(device)
    classifier_optimizer = torch.optim.Adam(
        classifier_model.parameters(), lr=Params.LR)

    for e in tqdm(range(Params.GENERATION_EPOCHS)):
        train_report = train_generation(m1_model, unlabeledloader, optimizer)
        test_report = test_generation(m1_model, unlabeledloader, e, filename)
        log_results_generation(train_report, test_report, e)
        produce_images(m1_model, e, filename)

    for e in tqdm(range(Params.CLASSIFICATION_EPOCHS)):
        train_report = train_classification(m1_model, classifier_model, labeledloader, classifier_optimizer)
        test_report = test_classification(m1_model, classifier_model, testloader)
        log_results_classification(train_report, test_report, e)


if __name__ == "__main__":
    main()
