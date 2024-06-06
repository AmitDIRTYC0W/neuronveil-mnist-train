import random

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from pypop7.optimizers.pso.clpso import CLPSO
from pypop7.optimizers.ga.gl25 import GL25
from pypop7.optimizers.de.shade import SHADE
from pypop7.optimizers.de.jade import JADE
from sklearn.datasets import load_digits
import numpy as np

from . import com
from .model import Model
from .dense import Dense
from .relu import ReLU
from .softmax import Softmax

dataset = load_digits()

BATCH_SIZE = 64


def obtain_batch(size=BATCH_SIZE) -> tuple[ArrayLike, ArrayLike]:
    index = random.randint(0, len(dataset.data) - size)
    return dataset.data[index:index + size], dataset.target[index:index + size]


def define_model(parameters: ArrayLike, log: bool = False) -> Model:
    # Cast the parameters from np.float64 to Com
    #parameters = jnp.int32(parameters * (1 << com.FRACTION_BITS))
    parameters = parameters * (1 << com.FRACTION_BITS)

    no_inputs = 8 * 8
    no_hidden = 16
    no_classes = 10

    W1 = parameters[0:1024].reshape((no_hidden, no_inputs))
    b1 = parameters[1024:1040].reshape((no_hidden, ))
    W2 = parameters[1040:1200].reshape((no_classes, no_hidden))
    b2 = parameters[1200:1210].reshape((no_classes, ))

    return Model([Dense(W1, b1), ReLU(), Dense(W2, b2), Softmax(log)])


def evaluate(parameters: ArrayLike, input_: ArrayLike, class_: int) -> float:
    # Perform one-hot encoding. For example, 3 shall become [0, 0, 1, 0, 0, ...]
    expected_output = jax.nn.one_hot(class_, 10)

    # Evaluate the neural network
    model = define_model(parameters)
    actual_output = model.infer(input_)

    # Comppute the loss
    difference = actual_output - expected_output
    return jnp.dot(difference, difference)


def objective(parameters: ArrayLike) -> float:
    batch_evaluate = jax.vmap(evaluate, in_axes=(None, 0, 0))
    images, image_classes = obtain_batch()
    return jnp.mean(batch_evaluate(parameters, images, image_classes))


batch_objective = jax.jit(jax.vmap(objective, in_axes=(0)))


def train_mnist_model() -> Model:
    dimensions = 1210
    problem = {
        'fitness_function':
        jax.jit(objective),  # Cost function, maybe batch_objective instead?
        'ndim_problem': dimensions,
        #'lower_boundary': np.full(dimensions, com.MINIMUM_VALUE_COM),
        #'upper_boundary': np.full(dimensions, com.MAXIMUM_VALUE_COM)
        'lower_boundary': np.full(dimensions, -5000),
        'upper_boundary': np.full(dimensions, +5000)
    }

    # optimiser = CLPSO(problem, {'n_individuals': 20_000, 'max_function_evaluations': 500_000})

    # optimiser = GL25(
    #     problem, {
    #         'max_function_evaluations': 300_000,
    #         'n_male_global': 800,
    #         'n_female_global': 400,
    #         'n_female_local': 10,
    #         'n_male_local': 200
    #     })

    print('Optimizer: SHADE (64)')
    optimizer = SHADE(
        problem,
        {
            'max_function_evaluations': 100_000,
            'n_individuals': 2_000,
            'seed_rng': 0
        }
    )

    # optimizer = JADE(
    #     problem,
    #     {
    #         'max_function_evaluations': 100_000,
    #         'n_individuals': 1_000,
    #         'seed_rng': 0
    #     }
    # )

    results = optimizer.optimize()
    print(results)

    model = define_model(results['best_so_far_x'])

    totest = 10_000
    correct = 0
    for i in range(totest):
        example_x, example_y = obtain_batch(1)
        example_x = example_x[0]
        example_y = example_y[0]

        # Test
        y = model.infer(example_x)
        y_i = np.argmax(y)

        # print('Output of predicted:', y[y_i], 'Output of correct answer:',
        #       y[example_y])
        # print('Output', y_i, 'Correct answer:', example_y)

        if example_y == y_i:
            correct += 1

    print(f'{correct / totest * 100}%')

    return model


if __name__ == '__main__':
    train()
