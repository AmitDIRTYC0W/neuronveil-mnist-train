from abc import ABC, abstractmethod

from jax.typing import ArrayLike

class Layer(ABC):
	@abstractmethod
	def infer(self, input_: ArrayLike, fraction_bits: int) -> ArrayLike:
		pass
	
class SerializableLayer(Layer):
	@abstractmethod
	def serialize(self) -> dict:
		pass
