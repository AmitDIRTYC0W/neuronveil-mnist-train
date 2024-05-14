import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .layer import Layer
from . import com

class Softmax(Layer):
	def __init__(self, log: bool = False):
		self.log = log
		
	def infer(self, input_: ArrayLike) -> ArrayLike:
		input_ = jnp.float32(input_) / (1 << com.FRACTION_BITS)

		if self.log:
			return jax.nn.log_softmax(input_)
		else:
			return jax.nn.softmax(input_)
