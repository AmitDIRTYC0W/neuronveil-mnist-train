from jax.typing import ArrayLike
import jax.numpy as jnp

from . import com
from .layer import SerializableLayer
from .utils import serialize_array

class Dense(SerializableLayer):
	def __init__(self, weights: ArrayLike, biases: ArrayLike):
		self.weights = weights
		self.biases = biases

	def infer(self, input_: ArrayLike) -> ArrayLike:
		return jnp.matmul(self.weights, input_) // (1 << com.FRACTION_BITS) + self.biases

	def serialize(self) -> dict:
		return {
			'type': 'DenseLayer',
			'biases': serialize_array(self.biases),
			'weights': serialize_array(self.weights)
		}
