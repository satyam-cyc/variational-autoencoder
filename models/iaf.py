import torch
import torch.nn as nn
import torch.nn.functional as F

from models import VAEModel, VAEEncoder, VAEDecoder
from models import register_model, register_model_architecture


@register_model('iaf')
class IAFVAE(VAEModel):
    def __init__(self, encoder, decoder, flow):
        """See Kingma et al. (https://arxiv.org/pdf/1606.04934.pdf)"""
        super().__init__(encoder, decoder, flow)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--input-channels', default=1, type=int, help='number of input channels')
        parser.add_argument('--latent-dim', default=64, type=int, help='dimension of latent space')
        parser.add_argument('--hidden-size', default=320, type=int, help='hidden size for masked linear layers')
        parser.add_argument('--num-flows', default=2, type=int, help='number of flow layers')
        parser.add_argument('--num-layers', default=1, type=int, help='number of hidden layers for each flow layer')

    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        base_architecture(args)
        encoder = VAEEncoder(args.input_channels, args.latent_dim)
        decoder = VAEDecoder(args.input_channels, args.latent_dim)
        flow = InverseAutoregressiveFlow(args.latent_dim, args.hidden_size, args.num_flows, args.num_layers)
        return cls(encoder, decoder, flow)


class InverseAutoregressiveFlow(nn.Module):
    def __init__(self, latent_dim=64, hidden_size=320, num_flows=2, num_layers=1, encoder_output_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.flow_steps = nn.ModuleList()

        # Build linear layers for MADE
        input_mask, hidden_mask, output_mask = self.build_masks()
        for step in range(num_flows):
            input_layers = nn.Sequential(MaskedLinear(latent_dim, hidden_size, input_mask), nn.ELU())
            hidden_layers = []
            for _ in range(num_layers):
                hidden_layers.append(MaskedLinear(hidden_size, hidden_size, hidden_mask))
                hidden_layers.append(nn.ELU())
            hidden_layers = nn.Sequential(*hidden_layers)

            mean_layer = MaskedLinear(hidden_size, latent_dim, output_mask)
            std_layer = MaskedLinear(hidden_size, latent_dim, output_mask)
            self.flow_steps.append(nn.ModuleList([input_layers, hidden_layers, mean_layer, std_layer]))

        self.context_proj = nn.Linear(encoder_output_dim, hidden_size)
        # For reordering latent vector after each flow
        reverse_order = torch.arange(latent_dim - 1, -1, -1).long()
        self.register_buffer('reverse_order', reverse_order)

    def build_masks(self):
        latent_degrees = torch.arange(self.latent_dim)
        hidden_degrees = torch.arange(self.hidden_size) % (self.latent_dim - 1)
        input_mask = (latent_degrees.unsqueeze(0) <= hidden_degrees.unsqueeze(-1)).float()
        hidden_mask = (hidden_degrees.unsqueeze(0) <= hidden_degrees.unsqueeze(-1)).float()
        output_mask = (hidden_degrees.unsqueeze(0) < latent_degrees.unsqueeze(-1)).float()
        return input_mask, hidden_mask, output_mask

    def forward(self, input, encoder_out):
        context = self.context_proj(encoder_out)
        log_det = input.new_zeros(input.size(0), 1)

        for i, (input_layers, hidden_layers, mean_layer, std_layer) in enumerate(self.flow_steps):
            # Reverse ordering helps mixing
            if (i + 1) % 2 == 0:
                input = input[:, self.reverse_order]

            context = self.context_proj(encoder_out)
            hidden = hidden_layers(input_layers(input) + context)
            mean = mean_layer(hidden)
            gate = torch.sigmoid(std_layer(hidden) + 1.0)
            input = gate * input + (1 - gate) * mean
            log_det += gate.log().view(gate.size(0), -1).sum(dim=1, keepdim=True)
        return input, log_det


class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, mask, bias=True):
        super(MaskedLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.register_buffer('mask', mask)

    def forward(self, inputs, cond_inputs=None):
        output = F.linear(inputs, self.linear.weight * self.mask, self.linear.bias)
        return output


@register_model_architecture('iaf', 'iaf')
def base_architecture(args):
    args.input_dim = getattr(args, 'input_channels', 1)
    args.latent_dim = getattr(args, 'latent_dim', 64)
    args.hidden_size = getattr(args, 'hidden_size', 320)
    args.num_flows = getattr(args, 'num_flows', 2)
    args.num_layers = getattr(args, 'num_layers', 1)
