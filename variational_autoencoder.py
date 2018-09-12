import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms, utils


class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc2_mu = nn.Linear(400, 20)
        self.fc2_var = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2_mu(h), self.fc2_var(h)

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if self.training else mu
        return self.decode(z), mu, logvar


def loss_function(output, x, mu, logvar):
    reconstruction_loss = F.binary_cross_entropy(output, x.view(-1, 784), size_average=False)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu * mu - torch.exp(logvar))
    loss = reconstruction_loss + kl_divergence
    return loss


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Bayes by Backprop')
    parser.add_argument('--data-path', required=True, metavar='DIR', help='path to dataset')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default 128)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N', help='number of epochs to train (default 10)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (default 0.01)')
    parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default 42)')
    parser.add_argument('--sample-dir', default='sample', metavar='DIR', help='directory for output images')
    args = parser.parse_args()
    print(args)
    return args


def train(model, optimizer, train_loader, test_loader, device, num_epochs, sample_dir):
    model.train()
    for epoch in range(1, num_epochs + 1):
        avg_train_loss = 0
        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output, mu, logvar = model(data)
            loss = loss_function(output, data, mu, logvar)
            avg_train_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_train_loss /= len(train_loader.dataset)
        avg_test_loss = evaluate(epoch, model, test_loader, device, sample_dir)
        print('Epoch {}: train_loss = {:.6f}, test_loss = {:.6f}'.format(epoch, avg_train_loss, avg_test_loss))


def evaluate(epoch, model, data_loader, device, sample_dir):
    model.eval()
    avg_loss = 0
    for idx, (data, _) in enumerate(data_loader):
        data = data.to(device)
        output, mu, logvar = model(data)
        loss = loss_function(output, data, mu, logvar)
        avg_loss += loss.item()
        if idx == 0:
            n = min(data.size(0), 64)
            concat_output = torch.cat([data[:n], output.view(data.size(0), 1, 28, 28)[:n]])
            utils.save_image(concat_output.cpu(), '{}/sample_epoch_{:03d}.png'.format(sample_dir, epoch), nrow=16)

    avg_loss /= len(data_loader.dataset)
    return avg_loss


def main():
    args = get_args()
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    os.makedirs(args.sample_dir, exist_ok=True)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data_path, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data_path, train=False, transform=transforms.Compose([
            transforms.ToTensor()
        ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    model = VariationalAutoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train(model, optimizer, train_loader, test_loader, device, args.epochs, args.sample_dir)


if __name__ == '__main__':
    main()

