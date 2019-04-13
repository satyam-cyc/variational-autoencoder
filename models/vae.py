from models import VAEModel, VAEEncoder, VAEDecoder
from models import register_model, register_model_architecture


@register_model('vanilla')
class VanillaVAE(VAEModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--input-channels', default=1, type=int, help='number of input channels')
        parser.add_argument('--latent-dim', default=64, type=int, help='dimension of latent space')

    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        base_architecture(args)
        encoder = VAEEncoder(args.input_channels, args.latent_dim)
        decoder = VAEDecoder(args.input_channels, args.latent_dim)
        return cls(encoder, decoder)


@register_model_architecture('vanilla', 'vanilla')
def base_architecture(args):
    args.input_channels = getattr(args, 'input_channels', 1)
    args.latent_dim = getattr(args, 'latent_dim', 64)
