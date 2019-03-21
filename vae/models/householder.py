import torch
import torch.nn as nn
from vae.models import VAEModel, VAEEncoder, VAEDecoder
from vae.models import register_model, register_model_architecture


@register_model('householder')
class HouseholderVAE(VAEModel):
    def __init__(self, encoder, decoder, flow):
        """See Tomczak and Welling (https://arxiv.org/pdf/1611.09630.pdf)"""
        super().__init__(encoder, decoder, flow)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--input-channels', default=1, type=int, help='number of input channels')
        parser.add_argument('--latent-dim', default=64, type=int, help='dimension of latent space')
        parser.add_argument('--num-flows', default=10, type=int, help='number of flow layers')

    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        base_architecture(args)
        encoder = VAEEncoder(args.input_channels, args.latent_dim)
        decoder = VAEDecoder(args.input_channels, args.latent_dim)
        flow = VAEFlow(nn.ModuleList([HouseholderFlow(args.latent_dim) for _ in range(args.num_flows)]))
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


class HouseholderFlow(nn.Module):
    def __init__(self, latent_dim=64, encoder_output_dim=256):
        super().__init__()
        self.vector_proj = nn.Linear(encoder_output_dim, latent_dim)

    def forward(self, input, encoder_output):
        # input: B x L, encoder_output: B x D
        vector = self.vector_proj(encoder_output)
        outer = torch.bmm(vector.unsqueeze(dim=2), vector.unsqueeze(dim=1))
        norm = torch.norm(vector, dim=1, keepdim=True)
        output = input - 2 * torch.bmm(outer, input.unsqueeze(dim=2)).squeeze(dim=2) / norm
        # Householder flow preserves volume
        log_det = input.new_zeros(input.size(0), 1)
        return output, log_det


@register_model_architecture('householder', 'householder')
def base_architecture(args):
    args.input_dim = getattr(args, 'input_channels', 1)
    args.latent_dim = getattr(args, 'latent_dim', 64)
    args.num_layers = getattr(args, 'num_flows', 10)
