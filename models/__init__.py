# __init__.py
# Import model classes
from .conv_ae_3L import SpectralConvAutoencoder_3L
from .conv_ae_3L12 import SpectralConvAutoencoder_3L12
from .conv_ae_3L12DW import SpectralConvAutoencoder_3L12DW
from .conv_ae_4L12DW import SpectralConvAutoencoder_4L12DW

# Registry maps short string → model class
MODEL_REGISTRY = {
    "3L": SpectralConvAutoencoder_3L,
    "3L12": SpectralConvAutoencoder_3L12,
    "3L12DW": SpectralConvAutoencoder_3L12DW,
    "4L12DW": SpectralConvAutoencoder_4L12DW,
}

# Convenience function to fetch class by name
def get_model_class(name):
    try:
        return MODEL_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown model name: {name}")
