import typing

import functools

from jax.typing import ArrayLike
import jax.numpy as jnp

from .layer import Layer, SerializableLayer
from . import com
from .dense import Dense
from .relu import ReLU


class Model:
    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def infer(self, input_: ArrayLike, fraction_bits: int) -> ArrayLike:
        activations_com = jnp.int32(jnp.array(input_) * (1 << fraction_bits))
        #activations_com = input_ * (1 << com.FRACTION_BITS)

        for layer in self.layers:
            activations_com = layer.infer(activations_com, fraction_bits)

        return activations_com

    def serialize(self) -> dict:
        is_serializable = lambda layer: issubclass(type(layer), SerializableLayer)
        serializable_layers = filter(is_serializable, self.layers)

        serialized_layers = list(map(lambda layer: layer.serialize(), serializable_layers))

        return {
            'layers': serialized_layers
        }
    
    def __deserialize_layer(dict: dict) -> Layer:
        match dict['type']:
            case 'DenseLayer':
                return Dense.deserialize(dict)
            case 'ReLULayer':
                return ReLU.deserialize(dict)

    def deserialize(dict: dict) -> typing.Self:
        layers = map(Model.__deserialize_layer, dict['layers'])
        return Model(list(layers))
