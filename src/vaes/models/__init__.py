from .linear import LinearAutoEncoder, LinearVAE
from .conv import ConvAutoEncoder, ConvVAE

MODEL_REGISTRY: dict[str, type] = {
    "linear_ae": LinearAutoEncoder,
    "linear_vae": LinearVAE,
    "conv_ae": ConvAutoEncoder,
    "conv_vae": ConvVAE,
}
