import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEModel(nn.Module):
    def __init__(self, encoder, decoder, flow=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.flow = flow

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        pass

    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        raise NotImplementedError

    def forward(self, input, eps=None):
        # Calculate mean, variance, and context vector
        mean, variance, encoder_output = self.encoder(input)

        # Sample noise
        if eps is None:
            eps = input.new(mean.size()).normal_()
        z0 = eps.mul(variance.sqrt()).add_(mean)

        # Pass latent vector through normalizing flow
        if self.flow is not None:
            zk, log_det = self.flow(z0, encoder_output)
        else:
            zk, log_det = z0, input.new(input.size(0)).zero_()

        # Decode output
        output = self.decoder(zk)
        return output, mean, variance, log_det, z0, zk


class VAEEncoder(nn.Module):
    def __init__(self, input_channels=1, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            GatedConv2d(input_channels, 32, 5, stride=1, padding=2),
            GatedConv2d(32, 32, 5, stride=2, padding=2),
            GatedConv2d(32, 64, 5, stride=1, padding=2),
            GatedConv2d(64, 64, 5, stride=2, padding=2),
            GatedConv2d(64, 64, 5, stride=1, padding=2),
            GatedConv2d(64, 256, 7, stride=1, padding=0),
        )
        self.mean_proj = nn.Linear(256, latent_dim)
        self.variance_proj = nn.Linear(256, latent_dim)

    def forward(self, x):
        hidden = self.encoder(x).view(x.size(0), -1)
        mean = self.mean_proj(hidden)
        variance = F.softplus(self.variance_proj(hidden))
        return mean, variance, hidden


class VAEDecoder(nn.Module):
    def __init__(self, input_channels=1, latent_dim=64):
        super().__init__()
        self.decoder = nn.Sequential(
            GatedConvTranspose2d(latent_dim, 64, 7, stride=1, padding=0),
            GatedConvTranspose2d(64, 64, 5, stride=1, padding=2),
            GatedConvTranspose2d(64, 32, 5, stride=2, padding=2, output_padding=1),
            GatedConvTranspose2d(32, 32, 5, stride=1, padding=2),
            GatedConvTranspose2d(32, 32, 5, stride=2, padding=2, output_padding=1),
            GatedConvTranspose2d(32, 32, 5, stride=1, padding=2),
        )
        self.output_conv = nn.Conv2d(32, input_channels, 1)

    def forward(self, z):
        hidden = self.decoder(z.view(z.size(0), -1, 1, 1))
        output = torch.sigmoid(self.output_conv(hidden))
        return output


class GatedConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding)

    def forward(self, x):
        return self.conv1(x) * torch.sigmoid(self.conv2(x))


class GatedConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding=0):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.conv2 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)

    def forward(self, input):
        return self.conv1(input) * torch.sigmoid(self.conv2(input))
