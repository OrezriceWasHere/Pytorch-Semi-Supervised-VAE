import uuid
from sklearn.metrics import confusion_matrix, classification_report
import torch
from torch.utils.data import DataLoader
import clearml_interface
from tqdm import tqdm
from m2_vae import M2_VAE
from params import Params
from utilities import get_device, get_dataset, save_image, create_semisupervised_datasets
import numpy as np


def train_generation(model,
                     unlabeled_loader,
                     labeled_loader,
                     optimizer,
                     device,
                     ):
    model.train()

    generation_loss_list = []
    predictions, ground_truth = [], []

    n_batches = len(labeled_loader) + len(unlabeled_loader)
    n_unlabeled_per_labeled = n_batches // len(labeled_loader) + 1

    labeled_loader = iter(labeled_loader)
    unlabeled_loader = iter(unlabeled_loader)

    for i in range(n_batches):
        is_supervised = i % n_unlabeled_per_labeled == 0

        # get batch from respective dataloader
        if is_supervised:
            x, y = next(labeled_loader)
            x = x.to(device).view(x.shape[0], -1)

            with torch.no_grad():
                preds = model(x)
                predictions.extend(preds.cpu().numpy())
                ground_truth.extend(y.cpu().numpy())

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

    all_preds = np.asarray(predictions)
    all_ground_truth = np.asarray(ground_truth)

    return {
        "avg_gen_loss": -sum(generation_loss_list) / len(generation_loss_list),
        "avg_cls_accuracy": np.sum(all_preds == all_ground_truth) / len(all_preds),
        "ground_truth": all_ground_truth,
        "predictions": all_preds

    }


def test_generation(model,
                    test_loader,
                    device):
    model.eval()

    generation_loss_list = []
    predictions, ground_truth = [], []


    for x, y in test_loader:
        # get batch from respective dataloader

        preds = model(x.view(x.shape[0], -1))
        predictions.extend(preds.cpu().numpy())
        ground_truth.extend(y.cpu().numpy())

        y = torch.nn.functional.one_hot(y, Params.NUMBER_OF_CLASSES).to(device)
        x = x.to(device).view(x.shape[0], -1)

        # compute loss -- SSL paper eq 6, 7, 9
        q_y = model.encode_y(x)
        # labeled data loss -- SSL paper eq 6 and eq 9
        q_z_xy = model.encode_z(x, y)
        z = q_z_xy.rsample()
        p_x_yz = model.decode(y, z)
        loss = M2_VAE.loss_components_fn(x, y, z, model.p_y, model.p_z, p_x_yz, q_z_xy)
        loss -= Params.ALPHA * Params.LABELED_COUNT * q_y.log_prob(y)  # SSL eq 9
        # unlabeled data loss -- SSL paper eq 7
        loss = loss.mean(0)
        generation_loss_list.append(loss.item())

    all_preds = np.asarray(predictions)
    all_ground_truth = np.asarray(ground_truth)


    return {
        "avg_gen_loss": -sum(generation_loss_list) / len(generation_loss_list),
        "avg_cls_accuracy": np.sum(all_preds == all_ground_truth) / len(all_preds),
        "ground_truth": all_ground_truth,
        "predictions": all_preds
    }


def log_results_generation(train_results, test_results, epoch):
    train_gen_loss = train_results['avg_gen_loss']
    test_gen_loss = test_results['avg_gen_loss']
    clearml_interface.add_point_to_graph('train_gen_loss', 'train_gen_loss', epoch, train_gen_loss)
    clearml_interface.add_point_to_graph('test_gen_loss', 'test_gen_loss', epoch, test_gen_loss)


def log_results_classification(train_results, test_results, epoch):
    train_cls_accuracy = train_results['avg_cls_accuracy']
    test_cls_accuracy = test_results['avg_cls_accuracy']
    clearml_interface.add_point_to_graph('train_cls_accuracy', 'train_cls_accuracy', epoch, train_cls_accuracy)
    clearml_interface.add_point_to_graph('test_cls_accuracy', 'test_cls_accuracy', epoch, test_cls_accuracy)

    test_matrix = confusion_matrix(test_results['ground_truth'], test_results['predictions'])
    clearml_interface.add_confusion_matrix(test_matrix, 'test_confusion_matrix', 'test_confusion_matrix', epoch)
    train_matrix = confusion_matrix(train_results['ground_truth'], train_results['predictions'])
    clearml_interface.add_confusion_matrix(train_matrix, 'train_confusion_matrix', 'train_confusion_matrix', epoch)
    test_classification_report = classification_report(test_results['ground_truth'], test_results['predictions'])
    message = f"Test Classification report: \n{test_classification_report}\nepoch={epoch}\n*******\n"
    clearml_interface.add_text(message, epoch)

def vis_styles(model, device, epoch):
    model.eval()
    assert Params.LATENT_DIM == 2, 'Style viualization requires z_dim=2'
    y_dim = Params.NUMBER_OF_CLASSES
    for y in range(2,5):
        # y = (torch.tensor(y).unsqueeze(-1), args.y_dim).expand(100, args.y_dim).to(args.device)
        y = torch.nn.functional.one_hot(torch.tensor(y).unsqueeze(-1), y_dim).expand(100, y_dim).to(device)

        # order the first dim of the z latent
        c = torch.linspace(-5, 5, 10).view(-1,1).repeat(1,10).reshape(-1,1)
        z = torch.cat([c, torch.zeros_like(c)], dim=1).reshape(100, 2).to(device)

        # combine into z and pass through decoder
        x = model.decode(y, z).sample().view(y.shape[0], *Params.SAMPLE_SPACE)
        save_image(x.cpu(), f'latent_var_grid_sample_c1_y{y[0].nonzero().item()}', epoch)

        # order second dim of latent and pass through decoder
        z = z.flip(1)
        x = model.decode(y, z).sample().view(y.shape[0], *Params.SAMPLE_SPACE)
        save_image(x.cpu(), f'latent_var_grid_sample_c2_y{y[0].nonzero().item()}', epoch)

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
        test_report = test_generation(m2_model, testloader, device)
        log_results_generation(train_report, test_report, e)
        log_results_classification(train_report, test_report, e)

    vis_styles(m2_model, device, Params.GENERATION_EPOCHS)


if __name__ == "__main__":
    main()
