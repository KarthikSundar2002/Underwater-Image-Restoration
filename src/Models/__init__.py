# Copyright (c) EEEM071, University of Surrey
from src.model.model import MyModel as NewModel
from .SpectralTransformer import (
    SpectralTransformer,
)

__model_factory = {
    "SpectralTransformer": SpectralTransformer,
    "NewModel": NewModel,
}

def get_names():
    return list(__model_factory.keys())

def init_model(name, *args, **kwargs):
    if name not in list(__model_factory.keys()):
        raise KeyError(f"Unknown model: {name}")
    return __model_factory[name](*args, **kwargs)
