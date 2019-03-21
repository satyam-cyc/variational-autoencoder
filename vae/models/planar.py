import torch
import torch.nn as nn
import torch.nn.functional as F
from vae.models import VAEModel, VAEEncoder, VAEDecoder
from vae.models import register_model, register_model_architecture


@register_model('planar')
class PlanarVAE(VAEModel):
    def __init__(self, encoder, decoder, flow):
        """See Rezende et al. (https://arxiv.org/pdf/1505.05770.pdf)"""
        super().__init__(encoder, decoder, flow)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--input-channels', default=1, type=int, help='number of input channels')
        parser.add_argument('--latent-dim', default=64, type=int, help='dimension of latent space')
        parser.add_argument('--num-flows', default=4, type=int, help='number of flow layers')

    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        base_architecture(args)
        encoder = VAEEncoder(args.input_channels, args.latent_dim)
        decoder = VAEDecoder(args.input_channels, args.latent_dim)
        flow = VAEFlow(nn.ModuleList([PlanarFlow(args.latent_dim) for _ in range(args.num_flows)]))
        return cls(encoder, decoder, flow)


class VAEFlow(nn.Module):
    def __init__(self, flows):
        super().__init__()
        self.flows = flows

    def forward(self, input, encoder_output):
        output = input
        log_det = input.new_zeros(input.size(0), 1)
        for flow in self.flows:
            output, log_det_ = flow(output, encoder_output)
            log_det += log_det_
        return output, log_det


class PlanarFlow(nn.Module):
    def __init__(self, latent_dim=64, encoder_output_dim=256):
        super().__init__()
        self.scale_proj = nn.Linear(encoder_output_dim, latent_dim)
        self.weight_proj = nn.Linear(encoder_output_dim, latent_dim)
        self.bias_proj = nn.Linear(encoder_output_dim, 1)

    def forward(self, input, encoder_output):
        # Apply projections on encoder output
        scale = self.scale_proj(encoder_output)
        weight = self.weight_proj(encoder_output)
        bias = self.bias_proj(encoder_output)

        # Reparametrize scale to ensure invertibility
        # See Appendix A.1 (https://arxiv.org/pdf/1505.05770.pdf)
        dot = torch.sum(scale * weight, dim=-1, keepdim=True)
        scale = scale + (-1 + F.softplus(dot) - dot) / torch.norm(weight, dim=-1, keepdim=True) * weight

        hidden = torch.sum(input * weight, dim=-1, keepdim=True) + bias
        output = input + scale * torch.tanh(hidden)
        psi = weight * (1 - torch.tanh(hidden) ** 2)
        log_det = torch.log((1.0 + torch.sum(psi * scale, dim=-1, keepdim=True)).abs() + 1e-8)
        return output, log_det


@register_model_architecture('planar', 'planar')
def base_architecture(args):
    args.input_dim = getattr(args, 'input_channels', 1)
    args.latent_dim = getattr(args, 'latent_dim', 64)
    args.num_layers = getattr(args, 'num_flows', 4)
