from jax.typing import ArrayLike

def serialize_array(arr: ArrayLike):
	return {
		'data': [{ 'bits': int(x) } for x in arr.ravel()],
		'dim': arr.shape,
		'v': 1
	}

