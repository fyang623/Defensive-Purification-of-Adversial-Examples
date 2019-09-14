import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import target_models
from generators import Generator
from discriminators import Discriminator
from prepare_dataset import load_dataset
import datetime

import os
import argparse

torch.backends.cudnn.benchmark = True


def CWLoss(logits, target, is_targeted, num_classes=10, kappa=0):
    # inputs to the softmax function are called logits.
    # https://arxiv.org/pdf/1608.04644.pdf
    target_one_hot = torch.eye(num_classes).type(logits.type())[target.long()]

    # workaround here.
    # subtract large value from target class to find other max value
    # https://github.com/carlini/nn_robust_attacks/blob/master/l2_attack.py
    real = torch.sum(target_one_hot*logits, 1)
    other = torch.max((-target_one_hot + 1)*logits - target_one_hot*10000, 1)[0]
    kappa = torch.zeros_like(other).fill_(kappa)

    if is_targeted:
        return torch.sum(torch.max(other-real, kappa))
    return torch.sum(torch.max(real-other, kappa))


def train(G, D, f, target, is_targeted, thres, criterion_adv, criterion_gan, alpha, beta, train_loader, optimizer_G, optimizer_D, epoch, epochs, device, num_steps=3, verbose=True):
    n = 0
    acc = 0

    G.train()
    D.train()
    for i, (img, label) in enumerate(train_loader):
        valid = torch.ones(img.size(0), 1, requires_grad=False).to(device)
        fake = torch.zeros(img.size(0), 1, requires_grad=False).to(device)
        img_real = img.to(device)

        optimizer_G.zero_grad()

        pert = torch.clamp(G(img_real), -thres, thres)
        img_fake = pert + img_real
        img_fake = img_fake.clamp(min=0, max=1)

        y_pred = f(img_fake)

        if is_targeted:
            y_target = torch.empty_like(label).fill_(target).to(device)
            loss_adv = criterion_adv(y_pred, y_target, is_targeted)
            acc += torch.sum(torch.max(y_pred, 1)[1] == y_target).item()
        else:
            y_true = label.to(device)
            loss_adv = criterion_adv(y_pred, y_true, is_targeted)
            acc += torch.sum(torch.max(y_pred, 1)[1] != y_true).item()

        loss_gan = criterion_gan(D(img_fake), valid)
        loss_hinge = torch.mean(torch.max(torch.zeros(1, ).type(y_pred.type()), torch.norm(pert.view(pert.size(0), -1), p=2, dim=1) - thres))

        loss_g = loss_adv + alpha*loss_gan + beta*loss_hinge

        loss_g.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()
        if i % num_steps == 0:
            loss_real = criterion_gan(D(img_real), valid)
            loss_fake = criterion_gan(D(img_fake.detach()), fake)

            loss_d = 0.5*loss_real + 0.5*loss_fake

            loss_d.backward()
            optimizer_D.step()

        n += img.size(0)
        if verbose:
            print("\rEpoch [%d/%d]: [%d/%d], D Loss: %1.4f, G Loss: %3.4f [H %3.4f, A %3.4f], Acc: %.4f"
                  %(epoch+1, epochs, i, len(train_loader), loss_d.mean().item(), loss_g.mean().item(),
                  loss_hinge.mean().item(), loss_adv.mean().item(), acc/n), end="")

    if verbose: print()
    return acc/n


def test(G, f, target, is_targeted, thres, test_loader, epoch, epochs, device, verbose=True):
    n = 0
    acc = 0

    G.eval()
    for i, (img, label) in enumerate(test_loader):
        img_real = img.to(device)

        pert = torch.clamp(G(img_real), -thres, thres)
        img_fake = pert + img_real
        img_fake = img_fake.clamp(min=0, max=1)

        y_pred = f(img_fake)

        if is_targeted:
            y_target = torch.empty_like(label).fill_(target).to(device)
            acc += torch.sum(torch.max(y_pred, 1)[1] == y_target).item()
        else:
            y_true = label.to(device)
            acc += torch.sum(torch.max(y_pred, 1)[1] != y_true).item()

        n += img.size(0)

        if verbose:
            print('\rTest [%d/%d]: [%d/%d]' %(epoch+1, epochs, i, len(test_loader)), end="")

    if verbose: print()
    return acc/n


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train AdvGAN')
    parser.add_argument('--model', type=str, default="Model_MNIST", required=False, choices=["Model_MNIST", "Model_CIFAR"], help='model name (default: Model_MNIST)')
    parser.add_argument('--dataset_name', type=str, default="mnist", required=False, choices=["mnist", "fmnist", "cifar10"], help='dataset name (default: mnist)')
    parser.add_argument('--epochs', type=int, default=20, required=False, help='no. of epochs (default: 20)')
    parser.add_argument('--batch_size', type=int, default=128, required=False, help='batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.001, required=False, help='learning rate (default: 0.001)')
    parser.add_argument('--num_workers', type=int, default=4, required=False, help='no. of workers (default: 4)')
    parser.add_argument('--target', type=int, required=False, help='Target label')
    parser.add_argument('--thres', type=float, required=False, default=0.2, help='Perturbation bound')
    parser.add_argument('--gpu', action='store_true', default=True, help='Use GPU?')

    args = parser.parse_args()
    lr = args.lr
    batch_size = args.batch_size
    num_workers = args.num_workers
    epochs = args.epochs
    model_name = args.model
    target = args.target
    thres = args.thres # perturbation bound, used in loss_hinge
    gpu = args.gpu
    dataset_name = args.dataset_name

    device = 'cuda' if gpu else 'cpu'
    # print(torch.cuda.get_device_name(0))
    torch.cuda.set_device(0)

    is_targeted = False
    if target in range(0, 10):
        is_targeted = True # bool variable to indicate targeted or untargeted attack

    print('Training AdvGAN ', '(Target %d)'%(target) if is_targeted else '(Untargeted)')

    train_data, test_data, in_channels, num_classes = load_dataset(dataset_name)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    D = Discriminator(dataset_name).to(device)
    G = Generator(dataset_name).to(device)
    f = getattr(target_models, model_name)(in_channels, num_classes).to(device)

    # load a pre-trained target model
    checkpoint_path = os.path.join('saved', 'target_models', 'best_%s.pth.tar'%(model_name))
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    f.load_state_dict(checkpoint["state_dict"])
    f.eval()

    optimizer_G = optim.Adam(G.parameters(), lr=lr)
    optimizer_D = optim.Adam(D.parameters(), lr=lr)

    scheduler_G = StepLR(optimizer_G, step_size=5, gamma=0.1)
    scheduler_D = StepLR(optimizer_D, step_size=5, gamma=0.1)

    criterion_adv = CWLoss # loss for fooling target model
    criterion_gan = nn.MSELoss() # for gan loss
    alpha = 1 # gan loss multiplication factor
    beta = 1 # for hinge loss
    num_steps = 3 # number of generator updates for 1 discriminator update

    for epoch in range(epochs):
        start_time = datetime.datetime.now()

        acc_train = train(G, D, f, target, is_targeted, thres, criterion_adv, criterion_gan, alpha, beta, train_loader, optimizer_G, optimizer_D, epoch, epochs, device, num_steps, verbose=True)
        acc_test = test(G, f, target, is_targeted, thres, test_loader, epoch, epochs, device, verbose=True)

        scheduler_G.step(epoch)
        scheduler_D.step(epoch)

        end_time = datetime.datetime.now()

        print('Epoch [%d/%d]: %.2f seconds\t'%(epoch+1, epochs, (end_time - start_time).total_seconds()))
        print('Train Acc: %.5f\t'%(acc_train))
        print('Test Acc: %.5f\n'%(acc_test))

        torch.save({"epoch": epoch+1,
                    "epochs": epochs,
                    "is_targeted": is_targeted,
                    "target": target,
                    "thres": thres,
                    "state_dict": G.state_dict(),
                    "acc_test": acc_test,
                    "optimizer": optimizer_G.state_dict()
                    }, "saved/%s_%s.pth.tar"%(model_name, 'target_%d'%(target) if is_targeted else 'untargeted'))
