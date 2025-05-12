# Copyright (c) EEEM071, University of Surrey
from src.model.model import MyModel as NewModel
from src.model.model import MyBigModel as NewBigModel
from src.model.model import MyBigFRFNModel as NewBigFRFNModel
from .SpectralTransformer import (
    SpectralTransformer,
)
from src.Models.AST import AST

__model_factory = {
    "SpectralTransformer": SpectralTransformer,
    "NewModel": NewModel,
    "NewBigModel": NewBigModel,
    "NewBigFRFNModel": NewBigFRFNModel,
    "AST": AST
}

def get_names():
    return list(__model_factory.keys())

def init_model(name, *args, **kwargs):
    print("modelCreated")
    if name not in list(__model_factory.keys()):
        raise KeyError(f"Unknown model: {name}")
    if "use_dwt" in kwargs: # dont combine these ifs - otherwise trying to remove use_dwt when it doesnt exist.
        if name is "NewModel":
            return __model_factory[name].init_model(*args, **kwargs)
        else:
            kwargs.pop("use_dwt")
    return __model_factory[name](*args, **kwargs)

def resume_model(model, *args, **kwargs):
    if name not in list(__model_factory.keys()):
        raise KeyError(f"Unknown model: {model}")

    __model_factory[name](*args, **kwargs)
