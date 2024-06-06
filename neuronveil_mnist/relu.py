import jax.numpy as jnp
from jax.typing import ArrayLike

from .layer import SerializableLayer

class ReLU(SerializableLayer):
	def infer(self, input_: ArrayLike, fraction_bits: int) -> ArrayLike:
		_ = fraction_bits
		return jnp.maximum(0, input_)

	def serialize(self) -> dict:
		return {
			'type': 'ReLULayer'
		}
