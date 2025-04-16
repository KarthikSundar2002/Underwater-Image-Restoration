# Copyright (c) EEEM071, University of Surrey
from src.model.model import MyModel as NewModel
from .SpectralTransformer import (
    SpectralTransformer,
)
from src.Models.AST import AST

__model_factory = {
    "SpectralTransformer": SpectralTransformer,
    "NewModel": NewModel,
    "AST": AST
}

def get_names():
    return list(__model_factory.keys())

def init_model(name, *args, **kwargs):
    print("modelCreated")
    if name not in list(__model_factory.keys()):
        raise KeyError(f"Unknown model: {name}")
    if name is "NewModel" and "use_dwt" in kwargs:
        return __model_factory[name].init_model(*args, **kwargs)
    else:
        kwargs.pop("use_dwt")
    return __model_factory[name](*args, **kwargs)
