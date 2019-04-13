### Overview
This repository contains a PyTorch implementation of several variational autencoders based on normalizing flows:
* [Variational Inference with Normalizing Flows](https://arxiv.org/pdf/1505.05770.pdf)
* [Improved Variational Inference with Inverse Autoregressive Flow](https://arxiv.org/pdf/1606.04934.pdf)
* [Improving Variational Auto-Encoders using Householder Flow](https://arxiv.org/pdf/1611.09630.pdf)

### Installation
The code was written for Python 3.6 or higher, and it has been tested with [PyTorch](http://pytorch.org/) 1.0.1. Training is only available with GPU. To get started, try to clone the repository

```bash
git clone https://github.com/tangbinh/variational-autoencoder
cd variational-autoencoder
```

### Download
The current code supports training a binarized version of MNIST, wich can be downloaded with the following commands:
```bash
DATA_DIR=data/MNIST_static
mkdir -p $DATA_DIR && cd $DATA_DIR
wget http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_train.amat
wget http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_valid.amat
wget http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_test.amat
cd ../..
```

### Training
 Once the data is downloaded, you can run the following command to train a model:
```bash
python train.py --arch iaf --latent-dim 64 --num-flows 2 --num-layers 1
```
The code also makes use of Tensorboard, so you can call `tensorboard --logdir runs` and open `http://localhost:6006` with your web browser to see plots of training and validation statistics.