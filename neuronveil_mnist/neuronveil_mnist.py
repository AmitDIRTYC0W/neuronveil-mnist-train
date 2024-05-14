import os

import numpy as np
import orjson

from .dense import Dense
from .relu import ReLU
from .softmax import Softmax
from .model import Model
from .train import train_mnist_model

def _save_model_to_file(model: Model, filepath: str) -> None:
    serialized_model = orjson.dumps(model.serialize())

    with open(filepath, 'wb') as f:
        f.write(serialized_model)

def main():
    model = train_mnist_model()
    _save_model_to_file(model, '/artifacts/mnist.json')

