import random
import argparse

import optuna
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from pypop7.optimizers.pso.clpso import CLPSO
from pypop7.optimizers.ga.gl25 import GL25
from pypop7.optimizers.de.shade import SHADE
from pypop7.optimizers.de.jade import JADE
from pypop7.optimizers.ep.lep import LEP
from sklearn.datasets import load_digits
import numpy as np

from . import com
from .model import Model
from .dense import Dense
from .relu import ReLU
from .softmax import Softmax

dataset = load_digits()

BATCH_SIZE = 768


def obtain_batch(size=BATCH_SIZE) -> tuple[ArrayLike, ArrayLike]:
    index = random.randint(0, len(dataset.data) - size)
    return dataset.data[index:index + size], dataset.target[index:index + size]


def define_model(parameters: ArrayLike, fraction_bits: int, log: bool = False) -> Model:
    # Cast the parameters from np.float64 to Com
    parameters = jnp.int32(parameters * (1 << fraction_bits))
    # parameters = parameters# * (1 << com.FRACTION_BITS)

    no_inputs = 8 * 8
    no_hidden = 32 # 16
    no_classes = 10

    total_0 = 0
    total_1 = no_inputs * no_hidden
    W1 = parameters[total_0:total_1].reshape((no_inputs, no_hidden))
    total_0 = total_1
    total_1 += no_hidden
    b1 = parameters[total_0:total_1].reshape((no_hidden, ))
    total_0 = total_1
    total_1 += no_hidden * no_classes
    W2 = parameters[total_0:total_1].reshape((no_hidden, no_classes))
    total_0 = total_1
    total_1 += no_classes
    b2 = parameters[total_0:total_1].reshape((no_classes, ))
    # print(total_1)
    # exit()

    return Model([Dense(W1, b1), ReLU(), Dense(W2, b2), Softmax(log)])


def evaluate(parameters: ArrayLike, input_: ArrayLike, class_: int, fraction_bits: int) -> float:
    # Perform one-hot encoding. For example, 3 shall become [0, 0, 1, 0, 0, ...]
    expected_output = jax.nn.one_hot(class_, 10)

    # Evaluate the neural network
    model = define_model(parameters, fraction_bits)
    actual_output = model.infer(input_, fraction_bits)

    # Compute the loss
    difference = actual_output - expected_output
    return jnp.dot(difference, difference)


def objective(parameters: ArrayLike, fraction_bits: int) -> float:
    batch_evaluate = jax.vmap(evaluate, in_axes=(None, 0, 0, None))
    images, image_classes = obtain_batch()
    print(images[0], image_classes[0])
    print(images.shape)
    exit()
    return jnp.mean(batch_evaluate(parameters, images, image_classes, fraction_bits))


batch_objective = jax.jit(jax.vmap(objective, in_axes=(0, None)))


def test(model: Model, fraction_bits: int) -> float:
    totest = 10_000
    correct = 0

    for i in range(totest):
        example_x, example_y = obtain_batch(1)
        example_x = example_x[0]
        example_y = example_y[0]

        # Test
        y = model.infer(example_x, fraction_bits)
        y_i = np.argmax(y)

        # print('Output of predicted:', y[y_i], 'Output of correct answer:',
        #       y[example_y])
        # print('Output', y_i, 'Correct answer:', example_y)

        if example_y == y_i:
            correct += 1

    return correct / totest


def train_by_trial(trial: optuna.Trial, final: bool = False) -> float:
    boundary = trial.suggest_int('boundary', 1, 100000, log=True) - 0.9
    fraction_bits = trial.suggest_int('fraction_bits', 1, 20)
    print('fraction_bits:', fraction_bits)

    dimensions = 2410
    problem = {
        'fitness_function':
        jax.jit(lambda x: objective(x, fraction_bits)),  # Cost function, maybe batch_objective instead?
        'ndim_problem': dimensions,
        #'lower_boundary': np.full(dimensions, com.MINIMUM_VALUE_COM),
        #'upper_boundary': np.full(dimensions, com.MAXIMUM_VALUE_COM)
        'lower_boundary': np.full(dimensions, -boundary),
        'upper_boundary': np.full(dimensions, +boundary)
    }

    # optimizer = SHADE(
    #     problem, {
    #         'max_function_evaluations':
    #         50_000,
    #         'n_individuals':
    #         trial.suggest_int('n_individuals', 10, 100_000, log=True),
    #         'seed_rng':
    #         0,
    #         'mu': trial.suggest_float('mu', 0.05, 1.0),
    #         'median': trial.suggest_float('median', 0.05, 1.0),
    #         'h': trial.suggest_int('h', 5, 300)
    #     })
    optimizer = LEP(
        problem, {
            'max_function_evaluations':
            500_000 if final else 50_000,
            'n_individuals':
            trial.suggest_int('n_individuals', 10, 1_500, log=True),
            'seed_rng':
            0,
            'sigma': trial.suggest_float('mu', 0.05, 10, log=True),
            # 'tau': trial.suggest_float('median', 0.05, 1.0),
            'q': trial.suggest_int('h', 3, 100)
        })
    print(optimizer.options)
    # optimizer = GL25(
    #     problem, {
    #         'max_function_evaluations':
    #         50_000,
    #         'n_individuals':
    #         trial.suggest_int('n_individuals', 10, 100_000, log=True),
    #         'seed_rng':
    #         0,
    #         'alpha':
    #         trial.suggest_float('alpha', 0.05, 10),
    #         'n_female_global':
    #         trial.suggest_int('n_female_global', 20, 2000, log=True),
    #         'n_male_global':
    #         trial.suggest_int('n_male_global', 20, 2000, log=True),
    #         'n_female_local':
    #         trial.suggest_int('n_female_local', 1, 100, log=True),
    #         'n_male_local':
    #         trial.suggest_int('n_male_local', 10, 1000, log=True),
    #         'p':
    #         trial.suggest_float('h', 0.01, 1)
    #     })

    try:
        results = optimizer.optimize()
        print(results)
    except:
        return 0

    model = define_model(results['best_so_far_x'], fraction_bits)
    if final:
        return model, test(model, fraction_bits)
    return test(model, fraction_bits)


def train_mnist_model() -> Model:
    argparser = argparse.ArgumentParser(
                    description='NeuronVeil MNIST training')
    argparser.add_argument('--final', type=bool, help='Load the best trial and train it for release')
    args = argparser.parse_args()

    # dimensions = 1210
    # problem = {
    #     'fitness_function':
    #     jax.jit(objective),  # Cost function, maybe batch_objective instead?
    #     'ndim_problem': dimensions,
    #     #'lower_boundary': np.full(dimensions, com.MINIMUM_VALUE_COM),
    #     #'upper_boundary': np.full(dimensions, com.MAXIMUM_VALUE_COM)
    #     'lower_boundary': np.full(dimensions, -5000),
    #     'upper_boundary': np.full(dimensions, +5000)
    # }

    # optimiser = CLPSO(problem, {'n_individuals': 20_000, 'max_function_evaluations': 500_000})

    # optimiser = GL25(
    #     problem, {
    #         'max_function_evaluations': 300_000,
    #         'n_male_global': 800,
    #         'n_female_global': 400,
    #         'n_female_local': 10,
    #         'n_male_local': 200
    #     })

    # print('Optimizer: SHADE (64)')
    # optimizer = SHADE(problem, {
    #     'max_function_evaluations': 100_000,
    #     'n_individuals': 2_000,
    #     'seed_rng': 0
    # })

    # optimizer = JADE(
    #     problem,
    #     {
    #         'max_function_evaluations': 100_000,
    #         'n_individuals': 1_000,
    #         'seed_rng': 0
    #     }
    # )

    # results = optimizer.optimize()
    # print(results)

    if args.final:
        study = optuna.load_study(study_name='fixed-point LEP', storage='sqlite:///study.sqlite')
        model, accuracy = train_by_trial(study.best_trial, final=True)
        print(f'Accuracy: {accuracy}')
        return model
    else:
        study = optuna.create_study(study_name='fixed-point LEP', storage='sqlite:///study.sqlite', direction='maximize')
        study.optimize(train_by_trial, n_trials=50)

    # model = define_model(results['best_so_far_x'])
    # print(f'{test(model) * 100}%')
    # return model
