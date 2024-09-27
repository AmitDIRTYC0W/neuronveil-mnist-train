# neuronveil-mnist-train

[![PyPI - Version](https://img.shields.io/pypi/v/an-ml-riddle-mnist.svg)](https://pypi.org/project/an-ml-riddle-mnist)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/an-ml-riddle-mnist.svg)](https://pypi.org/project/an-ml-riddle-mnist)

This is part of the NeuronVeil project (formally an-ml-riddle). This projects aims to develop an open source implementation
of privacy-preserving neural network inference, resembling FssNN[^fssnn]. This specific repository helps training neural
network models for NeuronVeil.

I chose to train a digit detection model on the well-known MNIST dataset, yet the code can be easily extended and modified.
Because the cryptography is done using discrete numbers, regular training methods (i.e. gradient descent) are unviable.
Thus, I developped this program to train it via alternative methods using [PyPop7](https://pypop.readthedocs.io/en/latest/index.html) and [Jax](https://jax.readthedocs.io/en/latest/).

-----

**Table of Contents**

- [Installation](#installation)
- [License](#license)

## Installation

```console
pip install an-ml-riddle-mnist
```

## License

`an-ml-riddle-mnist` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

## Citations

[^fssnn]: Yang, P., Jiang, Z. L., Gao, S., Wang, H., Zhou, J., Jin, Y., Yiu, S.-M., & Fang, J. (2023). *FssNN: Communication-efficient secure neural network training via function secret sharing*. *Cryptology ePrint Archive*, Paper 2023/073. https://eprint.iacr.org/2023/073
