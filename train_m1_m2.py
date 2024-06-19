from m1_m2_vae import M1_VAE, M2_VAE
from utilities import produce_images
from sklearn.metrics import confusion_matrix, classification_report
import torch
from torch.utils.data import DataLoader
import clearml_interface
from tqdm import tqdm
from params import Params
from utilities import get_device, get_dataset, save_image, create_semisupervised_datasets
import numpy as np
from torchvision.utils import  make_grid


def train_generation_m1(vae,
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


def test_generation_m1(vae, testloader, epoch, filename):
    vae.eval()
    gen_loss_list, cls_loss_list = [], []
    for index, (inputs, labels) in enumerate(testloader):
        batch_size = inputs.size(0)
        inputs = inputs.to(vae.device)
        outputs, mu, logvar = vae(inputs)
        gen_loss = vae.loss(inputs, outputs, mu, logvar)
        gen_loss_list.append(gen_loss.item() / batch_size)

    return {
        "avg_gen_loss": sum(gen_loss_list) / len(gen_loss_list)
    }

def log_results_generation(train_results, test_results, epoch):
    train_gen_loss = train_results['avg_gen_loss']
    test_gen_loss = test_results['avg_gen_loss']
    clearml_interface.add_point_to_graph('train_gen_loss', 'train_gen_loss', epoch, train_gen_loss)
    clearml_interface.add_point_to_graph('test_gen_loss', 'test_gen_loss', epoch, test_gen_loss)

def generate(model, test_dataset, device, epoch):
    n_samples_per_label = 10

    # some interesting samples per paper implementation
    idxs = [7910, 8150, 3623, 2645, 4066, 9660, 5083, 948, 2595, 2]

    x = torch.stack([test_dataset[i][0] for i in idxs], dim=0).to(device)
    y = torch.tensor([test_dataset[i][1] for i in idxs]).to(device)
    y = torch.nn.functional.one_hot(y, Params.NUMBER_OF_CLASSES).to(device)

    q_z_xy = model.encode_z(x.view(n_samples_per_label, -1), y)
    z = q_z_xy.loc
    z = z.repeat(Params.NUMBER_OF_CLASSES, 1, 1).transpose(0, 1).contiguous().view(-1, Params.LATENT_DIM)

    # hold z constant and vary y:
    y = (torch.eye(Params.NUMBER_OF_CLASSES)
         .repeat(n_samples_per_label, 1)
         .to(device))
    generated_x = model.decode(y, z).sample().view(n_samples_per_label, Params.NUMBER_OF_CLASSES, *Params.SAMPLE_SPACE)
    generated_x = generated_x.contiguous().view(-1, *Params.SAMPLE_SPACE)  # out (n_samples * n_label, C, H, W)

    x = make_grid(x.cpu(), nrow=1)
    spacer = torch.ones(x.shape[0], x.shape[1], 5)
    generated_x = make_grid(generated_x.cpu(), nrow=Params.NUMBER_OF_CLASSES)
    image = torch.cat([x, spacer, generated_x], dim=-1)
    save_image(image.cpu(), f'analogies_sample_at_epoch_{epoch}', epoch)


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
    m2_model = M2_VAE(
        number_of_classes=Params.NUMBER_OF_CLASSES,
        m1_space=Params.LATENT_DIM,
        latent_space=Params.M2_LATENT_DIM,
        hidden_space=Params.HIDDEN_DIM,
        device=device
    ).to(device)

    filename = 'latent2'
    print('images uuid: ', filename)

    optimizer = torch.optim.Adam(m1_model.parameters(), lr=Params.LR)
    optimizer2 = torch.optim.Adam(m2_model.parameters(), lr=Params.LR)

    for e in tqdm(range(Params.M1_ALONE_GENERATION_EPOCHS)):
        train_report = train_generation_m1(m1_model, unlabeledloader, optimizer)
        test_report = test_generation_m1(m1_model, unlabeledloader, e, filename)
        log_results_generation(train_report, test_report, e)
        produce_images(m1_model, e, filename)

    for e in tqdm(range(Params.GENERATION_EPOCHS)):
        train_report = train_generation(m1_model, m2_model, unlabeledloader, labeledloader, optimizer2, device)
        test_report = test_generation(m1_model, m2_model, testloader, device)
        log_results_generation(train_report, test_report, e)
        log_results_classification(train_report, test_report, e)
        # vis_styles(m1_model, m2_model, device, Params.GENERATION_EPOCHS)




def train_generation(m1_model, m2_model,
                     unlabeled_loader,
                     labeled_loader,
                     optimizer,
                     device,
                     ):
    m1_model.train()
    m2_model.train()
    generation_loss_list = []
    predictions, ground_truth = [], []

    n_batches = len(labeled_loader) + len(unlabeled_loader)
    n_unlabeled_per_labeled = n_batches // len(labeled_loader) + 1

    labeled_loader = iter(labeled_loader)
    unlabeled_loader = iter(unlabeled_loader)
    for i in range(n_batches):
        is_supervised = i % n_unlabeled_per_labeled == 0
        if is_supervised:

            index_l, (inputs_labeled, labels_l) = next(enumerate(labeled_loader))
            batch_size = inputs_labeled.size(0)
            # if index % supervised_training_factor == 0:
            inputs_labeled = inputs_labeled.to(m1_model.device)
            outputs_l, mu_l, logvar_l = m1_model(inputs_labeled)
            labels_l = labels_l.to(m1_model.device)
            latent_l = m1_model.reparametize(mu_l, logvar_l)
            latent_l = latent_l.detach().clone()
            x = latent_l
            y = labels_l
            x = x.to(device).view(x.shape[0], -1)

            with torch.no_grad():
                preds = m2_model(x)
                predictions.extend(preds.cpu().numpy())
                ground_truth.extend(y.cpu().numpy())

            y = torch.nn.functional.one_hot(y, Params.NUMBER_OF_CLASSES).to(device)
        else:
            (inputs, labels) = next(unlabeled_loader)
            batch_size = inputs.size(0)
            # if index % supervised_training_factor == 0:
            inputs = inputs.to(m1_model.device)
            outputs, mu, logvar = m1_model(inputs)
            labels = labels.to(m1_model.device)
            latent = m1_model.reparametize(mu, logvar)
            latent = latent.detach().clone()
            x = latent
            y = None
            x = x.to(device).view(x.shape[0], -1)






        # compute loss -- SSL paper eq 6, 7, 9
        q_y = m2_model.encode_y(x)
        # labeled data loss -- SSL paper eq 6 and eq 9
        if y is not None:
            q_z_xy = m2_model.encode_z(x, y)
            z = q_z_xy.rsample()
            p_x_yz = m2_model.decode(y, z)
            loss = M2_VAE.loss_components_fn(x, y, z, m2_model.p_y, m2_model.p_z, p_x_yz, q_z_xy)
            loss -= Params.ALPHA * Params.LABELED_COUNT * q_y.log_prob(y)  # SSL eq 9
        # unlabeled data loss -- SSL paper eq 7
        else:
            # marginalize y according to q_y
            loss = - q_y.entropy()
            for y in q_y.enumerate_support():
                q_z_xy = m2_model.encode_z(x, y)
                z = q_z_xy.rsample()
                p_x_yz = m2_model.decode(y, z)
                L_xy = M2_VAE.loss_components_fn(x, y, z, m2_model.p_y, m2_model.p_z, p_x_yz, q_z_xy)
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


def test_generation(m1_model, m2_model,
                    test_loader,
                    device):
    m1_model.eval()
    m2_model.eval()

    generation_loss_list = []
    predictions, ground_truth = [], []


    for x, y in test_loader:
        # get batch from respective dataloader
        # x = x.view(x.shape[0], -1)
        batch_size = x.size(0)
        inputs = x.to(m1_model.device)

        outputs, mu, logvar = m1_model(inputs)
        # y = y.to(m1_model.device)
        latent = m1_model.reparametize(mu, logvar)
        latent= latent.to(device).view(x.shape[0], -1)

        preds = m2_model(latent)
        predictions.extend(preds.cpu().numpy())
        ground_truth.extend(y.cpu().numpy())

        y = torch.nn.functional.one_hot(y, Params.NUMBER_OF_CLASSES).to(device)

        # compute loss -- SSL paper eq 6, 7, 9
        q_y = m2_model.encode_y(latent)
        # labeled data loss -- SSL paper eq 6 and eq 9
        q_z_xy = m2_model.encode_z(latent, y)
        z = q_z_xy.rsample()
        p_x_yz = m2_model.decode(y, z)
        loss = M2_VAE.loss_components_fn(latent, y, z, m2_model.p_y, m2_model.p_z, p_x_yz, q_z_xy)
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

def vis_styles(model1, model2, device, epoch):
    model1.eval()
    model2.eval()
    assert Params.M2_LATENT_DIM == 2, 'Style viualization requires z_dim=2'
    y_dim = Params.NUMBER_OF_CLASSES
    for y in range(2,5):
        # y = (torch.tensor(y).unsqueeze(-1), args.y_dim).expand(100, args.y_dim).to(args.device)
        y = torch.nn.functional.one_hot(torch.tensor(y).unsqueeze(-1), y_dim).expand(100, y_dim).to(device)

        # order the first dim of the z latent
        c = torch.linspace(-5, 5, 10).view(-1,1).repeat(1,10).reshape(-1,1)
        z = torch.cat([c, torch.zeros_like(c)], dim=1).reshape(100, 2).to(device)

        # combine into z and pass through decoder
        x = model2.decode(y, z).sample()
        x = model1.upsample(x)
        x = x.view(-1, *model1.encoder_decoder_z_space)
        x = model1.decoder(x)
        save_image(x.cpu(), f'results/latent_var_grid_sample_c1_y{y[0].nonzero().item()}', epoch)

        # order second dim of latent and pass through decoder
        z = z.flip(1)
        x = model2.decode(y, z).sample().view(y.shape[0], Params.LATENT_DIM)
        x = model1.upsample(x)
        x = x.view(-1, *model1.encoder_decoder_z_space)
        x = model1.decoder(x)

        save_image(x.cpu(), f'results/latent_var_grid_sample_c2_y{y[0].nonzero().item()}', epoch)



if __name__ == "__main__":
    main()
