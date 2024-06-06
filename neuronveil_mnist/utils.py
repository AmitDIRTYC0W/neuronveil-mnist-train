from jax.typing import ArrayLike
import jax.numpy as jnp

def serialize_array(arr: ArrayLike):
	return {
		'data': [{ 'bits': int(x) } for x in arr.ravel()],
		'dim': arr.shape,
		'v': 1
	}

def deserialize_array(d: dict) -> jnp.ndarray:
	return jnp.array([int(x['bits']) for x in d['data']], dtype=jnp.int32).reshape(d['dim'])
