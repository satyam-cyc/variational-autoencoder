import argparse
import logging
import numpy as np
import os
import random
import torch
import torch.nn.functional as F

from datetime import datetime
from scipy.special import logsumexp
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

from vae import models, utils
from vae.models import ARCH_MODEL_REGISTRY, ARCH_CONFIG_REGISTRY


def get_args():
    parser = argparse.ArgumentParser('Training Variational Autoencoders (VAE)')
    parser.add_argument('--seed', default=0, type=int, help='random number generator seed')

    # Add data arguments
    parser.add_argument('--data', default='data', help='path to MNIST_static/binarized_mnist_train.amat')
    parser.add_argument('--batch-size', default=100, type=int, help='batch size')
    parser.add_argument('--num-workers', default=4, type=int, help='number of workers')

    # Add model arguments
    parser.add_argument('--arch', default='vanilla', choices=ARCH_MODEL_REGISTRY.keys(), help='model architecture')

    # Add optimization arguments
    parser.add_argument('--max-epoch', default=2000, type=int, help='force stop training at specified iteration')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--lr-shrink', default=0.1, type=float, help='learning rate shrink factor for annealing')
    parser.add_argument('--min-lr', default=1e-6, type=float, help='minimum learning rate')
    parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay')
    parser.add_argument('--warmup', type=int, default=100, help='number of warm-up epochs for warm-up')
    parser.add_argument('--max_beta', type=float, default=1.0, help='max beta for warm-up')
    parser.add_argument('--min_beta', type=float, default=0.0, help='min beta for warm-up')

    # Add checkpoint arguments
    parser.add_argument('--checkpoint-dir', default='checkpoints', help='path to save checkpoints')
    parser.add_argument('--restore-file', default='checkpoint_last.pt', help='filename to load checkpoint')
    parser.add_argument('--valid-interval', type=int, default=1, help='validate every N epochs')
    parser.add_argument('--save-interval', type=int, default=1, help='save a checkpoint every N steps')
    parser.add_argument('--no-save', action='store_true', help='don\'t save models or checkpoints')
    parser.add_argument('--epoch-checkpoints', action='store_true', help='store all step checkpoints')

    # Add logging arguments
    parser.add_argument('--experiment', default='vae', help='experiment name to be used with Tensorboard')
    parser.add_argument('--no-log', action='store_true', help='don\'t save logs to file or Tensorboard directory')
    parser.add_argument('--log-dir', default='logs', help='directory to save logs')
    parser.add_argument('--log-interval', type=int, default=100, help='log every N steps')
    args = parser.parse_args()

    # Parse twice as model arguments are not known the first time
    args, _ = parser.parse_known_args()
    model_parser = parser.add_argument_group(argument_default=argparse.SUPPRESS)
    ARCH_MODEL_REGISTRY[args.arch].add_args(model_parser)
    args = parser.parse_args()
    ARCH_CONFIG_REGISTRY[args.arch](args)

    # Modify arguments
    args.experiment = '-'.join([datetime.now().strftime('%b-%d-%H:%M:%S'), args.experiment])
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.experiment)
    args.log_file = os.path.join(args.log_dir, args.experiment)
    return args


def main(args):
    if not torch.cuda.is_available():
        raise NotImplementedError('Training on CPU is not supported.')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    utils.init_logging(args)

    train_loader, valid_loader, test_loader = load_static_mnist(args)

    # Build a model and an optimizer
    model = models.build_model(args).cuda()
    logging.info('Built a model with {} parameters'.format(sum(p.numel() for p in model.parameters())))

    optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=args.lr_shrink)

    # Load last checkpoint if one exists
    state_dict = utils.load_checkpoint(args, model, optimizer, lr_scheduler)
    last_epoch = state_dict['last_epoch'] if state_dict is not None else -1

    loss_meter = utils.RunningAverageMeter(0.98)
    nll_meter = utils.RunningAverageMeter(0.98)
    kl_meter = utils.RunningAverageMeter(0.98)
    writer = SummaryWriter(log_dir='runs/{}'.format(args.experiment))

    for epoch in range(last_epoch + 1, args.max_epoch):
        model.train()
        beta = min([(epoch + 1) / max([args.warmup, 1.]), args.max_beta])
        progress_bar = tqdm(train_loader, desc='| Epoch {:03d}'.format(epoch), leave=False)

        for batch_id, sample in enumerate(progress_bar):
            input = utils.move_to_cuda(sample[0])
            output, mean, variance, log_det, z0, zk = model(input)
            nll = F.binary_cross_entropy(output, input, reduction='sum') / args.batch_size
            kl = ((utils.normal_log_prob(z0, mean, variance) - utils.normal_log_prob(zk)).sum() - log_det.sum()) / args.batch_size

            loss = (nll + beta * kl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update statistics for progress bar
            loss_meter.update(loss.item())
            nll_meter.update(nll.item())
            kl_meter.update(kl.item())
            stats = {'loss': loss_meter.avg, 'nll': nll_meter.avg, 'kl': kl_meter.avg, 'beta': beta, 'lr': optimizer.param_groups[0]['lr']}
            progress_bar.set_postfix({key: '{:.4g}'.format(value) for key, value in stats.items()}, refresh=True)

            # Log statistics and save sample images
            global_step = epoch * len(train_loader) + batch_id + 1
            if global_step % args.log_interval == 0:
                writer.add_scalar('loss/train_loss', loss.item(), global_step)
                writer.add_scalar('loss/train_nll', nll.item(), global_step)
                writer.add_scalar('loss/train_kl', kl.item(), global_step)
                writer.add_scalar('loss/beta', beta, global_step)

        logging.info('Epoch {:03d}: loss {:.3f} | nll {:.3f} | kl {:.3f} | lr {:.4g} | beta {:.3f}'.format(
            epoch, loss_meter.avg, nll_meter.avg, kl_meter.avg, optimizer.param_groups[0]['lr'], beta))

        # Perform validation and checkpoint saving
        if valid_loader is not None and (epoch + 1) % args.valid_interval == 0:
            valid_loss = validate(args, model, valid_loader, writer, epoch)
            lr_scheduler.step(valid_loss)
            if epoch % args.save_interval == 0:
                utils.save_checkpoint(args, model, optimizer, lr_scheduler, epoch, valid_loss)

        if optimizer.param_groups[0]['lr'] <= args.min_lr:
            logging.info('Done training!')
            break

    # Calculate negative log likelihood
    test(args, model, test_loader)


def validate(args, model, valid_loader, writer, epoch):
    loss_meter = utils.AverageMeter()
    nll_meter = utils.AverageMeter()
    kl_meter = utils.AverageMeter()
    progress_bar = tqdm(valid_loader, desc='| Epoch {:03d}'.format(epoch), leave=False)

    model.eval()
    with torch.no_grad():
        for batch_id, sample in enumerate(progress_bar):
            # Forward pass
            input = utils.move_to_cuda(sample[0])
            output, mean, variance, log_det, z0, zk = model(input)
            nll = F.binary_cross_entropy(output, input, reduction='sum') / args.batch_size
            kl = ((utils.normal_log_prob(z0, mean, variance) - utils.normal_log_prob(zk)).sum() - log_det.sum()) / args.batch_size
            loss = nll + kl

            # Update statistics for progress bar
            loss_meter.update(loss.item())
            nll_meter.update(nll.item())
            kl_meter.update(kl.item())
            stats = {'valid_loss': loss_meter.avg, 'valid_nll': nll_meter.avg, 'valid_kl': kl_meter.avg}
            progress_bar.set_postfix({key: '{:.4g}'.format(value) for key, value in stats.items()}, refresh=True)

        if writer is not None:
            writer.add_scalar('loss/valid_loss', loss_meter.avg, epoch)
            writer.add_scalar('loss/valid_nll', nll_meter.avg, epoch)
            writer.add_scalar('loss/valid_kl', kl_meter.avg, epoch)

        logging.info('Epoch {:03d}: valid_loss {:.3f} | valid_nll {:.3f} | valid_kl {:.3f}'.format(
            epoch, loss_meter.avg, nll_meter.avg, kl_meter.avg))
    return loss_meter.avg


def test(args, model, test_loader, minibatch_size=1000, replicates=2):
    nll_meter = utils.AverageMeter()
    progress_bar = tqdm(test_loader, desc='| Testing', leave=False)

    model.eval()
    with torch.no_grad():
        for batch_id, sample in enumerate(progress_bar):
            likelihoods = []
            for _ in range(replicates):
                input = utils.move_to_cuda(sample[0])
                input = input.expand(minibatch_size, *input.size()[1:])

                output, mean, variance, log_det, z0, zk = model(input)
                nll = F.binary_cross_entropy(output, input, reduction='none').view(input.size(0), -1).sum(dim=-1)
                kl = (utils.normal_log_prob(z0, mean, variance) - utils.normal_log_prob(zk)).sum(dim=-1) - log_det
                loss = nll + kl
                likelihoods.append(-loss.cpu().numpy())

            # log p(x) = log E(p(x|z) * p(z) / q(z))
            likelihoods = np.array(likelihoods).reshape(-1, 1)
            likelihoods = logsumexp(likelihoods) - np.log(len(likelihoods))

            nll_meter.update(-likelihoods)
            progress_bar.set_postfix({'nll': '{:.4f}'.format(nll_meter.avg)}, refresh=True)

    logging.info('Final evaluation: nll {:.4f}'.format(nll_meter.avg))
    return nll_meter.avg


def load_static_mnist(args):
    def load_data(filepath):
        with open(filepath) as file:
            data = np.array([[int(i) for i in line.split()] for line in file.readlines()])
            return TensorDataset(torch.from_numpy(data.astype('float32')).view(-1, 1, 28, 28))

    train_data = load_data(os.path.join(args.data, 'MNIST_static', 'binarized_mnist_train.amat'))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)

    valid_data = load_data(os.path.join(args.data, 'MNIST_static', 'binarized_mnist_valid.amat'))
    valid_loader = DataLoader(valid_data, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)

    test_data = load_data(os.path.join(args.data, 'MNIST_static', 'binarized_mnist_test.amat'))
    test_loader = DataLoader(test_data, shuffle=False, batch_size=1, num_workers=args.num_workers)
    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    args = get_args()
    main(args)
