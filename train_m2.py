import uuid
from sklearn.metrics import confusion_matrix, classification_report
import torch
from torch.utils.data import DataLoader
import clearml_interface
from tqdm import tqdm
from m2_vae import M2_VAE
from params import Params
from utilities import get_device, get_dataset, produce_images, create_semisupervised_datasets, one_hot


def train_generation(model,
                     unlabeled_loader,
                     labeled_loader,
                     optimizer, 
                     device,
                     ):
    model.train()

    generation_loss_list = []

    n_batches = len(labeled_loader) + len(unlabeled_loader)
    n_unlabeled_per_labeled = n_batches // len(labeled_loader) + 1

    labeled_loader = iter(labeled_loader)
    unlabeled_loader = iter(unlabeled_loader)

    for i in range(n_batches):
        is_supervised = i % n_unlabeled_per_labeled == 0

        # get batch from respective dataloader
        if is_supervised:
            x, y = next(labeled_loader)
            y = torch.nn.functional.one_hot(y, Params.NUMBER_OF_CLASSES).to(device)
        else:
            x, y = next(unlabeled_loader)
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
            loss -= Params.ALPHA * Params.LABELED_COUNT * q_y.log_prob(y)  # SSL eq 9
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
        generation_loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    return {
        "avg_gen_loss": -sum(generation_loss_list) / len(generation_loss_list)
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


def test_classification(vae, testloader):
    vae.eval()
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
    batch_size = Params.BATCH_SIZE
    trainset, testset = get_dataset('mnist')
    n_labeld = Params.LABELED_COUNT
    labeledloader, unlabeledloader = create_semisupervised_datasets(trainset,
                                                                    n_labeld,
                                                                    batch_size)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    m2_model = M2_VAE(
        number_of_classes=Params.NUMBER_OF_CLASSES,
        sample_space=Params.SAMPLE_SPACE,
        latent_space=Params.LATENT_DIM,
        hidden_space=Params.HIDDEN_DIM,
        device=device
    ).to(device)

    filename = str(uuid.uuid4())
    print('images uuid: ', filename)

    optimizer = torch.optim.Adam(m2_model.parameters(), lr=Params.LR)

    for e in tqdm(range(Params.GENERATION_EPOCHS)):
        train_report = train_generation(m2_model, unlabeledloader, labeledloader, optimizer, device)
        test_report = test_generation(m2_model, unlabeledloader, e, filename)
        log_results_generation(train_report, test_report, e)
        produce_images(m2_model, e, filename)


if __name__ == "__main__":
    main()
