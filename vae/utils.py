import os
import logging
import math
import sys
import torch

from torch.serialization import default_restore_location


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RunningAverageMeter(object):
    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def init_logging(args):
    os.makedirs(args.log_dir, exist_ok=True)
    handlers = [logging.StreamHandler()]
    if not args.no_log and args.log_file is not None:
        checkpoint_path = os.path.join(args.checkpoint_dir, args.restore_file)
        mode = 'a' if os.path.isfile(checkpoint_path) else 'w'
        handlers.append(logging.FileHandler(args.log_file, mode=mode))
    logging.basicConfig(handlers=handlers, format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
    logging.info('COMMAND: %s' % ' '.join(sys.argv))
    logging.info('Arguments: {}'.format(vars(args)))


def save_checkpoint(args, model, optimizer, lr_scheduler, epoch, loss):
    if args.no_save:
        return
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    last_epoch = getattr(save_checkpoint, 'last_epoch', -1)
    save_checkpoint.last_epoch = max(last_epoch, epoch)
    prev_best = getattr(save_checkpoint, 'loss', float('inf'))
    save_checkpoint.best_score = min(prev_best, loss)

    state_dict = {
        'epoch': epoch,
        'loss': loss,
        'best_score': save_checkpoint.best_score,
        'last_epoch': save_checkpoint.last_epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'args': args,
    }

    if args.epoch_checkpoints and epoch % args.save_interval == 0:
        torch.save(state_dict, os.path.join(args.checkpoint_dir, 'checkpoint{}_{:.3f}.pt'.format(epoch, loss)))
    if loss < prev_best:
        torch.save(state_dict, os.path.join(args.checkpoint_dir, 'checkpoint_best.pt'))
    if epoch > last_epoch:
        torch.save(state_dict, os.path.join(args.checkpoint_dir, 'checkpoint_last.pt'))


def load_checkpoint(args, model, optimizer, lr_scheduler):
    checkpoint_path = os.path.join(args.restore_file)
    if os.path.isfile(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=lambda s, l: default_restore_location(s, 'cpu'))
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
        save_checkpoint.best_score = state_dict['best_score']
        save_checkpoint.last_epoch = state_dict['last_epoch']
        logging.info('Loaded checkpoint {}'.format(checkpoint_path))
        return state_dict


def move_to_cuda(sample):
    if torch.is_tensor(sample):
        return sample.cuda()
    elif isinstance(sample, list):
        return [move_to_cuda(x) for x in sample]
    elif isinstance(sample, dict):
        return {key: move_to_cuda(value) for key, value in sample.items()}
    else:
        return sample


def normal_log_prob(input, mean=None, variance=None):
    if mean is None or variance is None:
        return -0.5 * math.log(2 * math.pi) - 0.5 * input.pow(2)
    return -0.5 * math.log(2 * math.pi) - 0.5 * (variance + 1e-8).log() - 0.5 * (input - mean).pow(2) / (variance + 1e-8)
