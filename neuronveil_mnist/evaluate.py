import os

import orjson
from pprint import pp as pprint

from .softmax import Softmax
from .model import Model

FRACTION_BITS = 2

# myinput = [ 0, 0, 0, 11, 16, 8, 0, 0, 0, 0, 6, 16, 13, 3, 0, 0, 0, 0,
#   8, 16, 8, 0, 0, 0, 0, 0, 13, 16, 2, 0, 0, 0, 0, 0, 15, 16,
#   5, 0, 0, 0, 0, 2, 16, 16, 16, 5, 0, 0, 0, 1, 10, 16, 16, 14,
#   0, 0, 0, 0, 0, 12, 16, 15, 0, 0] # 6

myinput = [ 0, 0, 0, 12, 16, 9, 0, 0, 0, 0, 2, 16, 16, 6, 0, 0, 0, 0, 
  3, 16, 16, 2, 0, 0, 0, 0, 8, 16, 12, 0, 0, 0, 0, 0, 6, 16, 
 16, 0, 0, 0, 0, 0, 10, 16, 15, 1, 0, 0, 0, 0, 9, 16, 11, 0, 
  0, 0, 0, 0, 8, 16, 10, 0, 0, 0, ] # 1

print(len(myinput))

def _read_model_from_file(filepath: str) -> Model:
    with open(filepath, 'rb') as f:
        model_dict = orjson.loads(f.read())

    model = Model.deserialize(model_dict)
    model.layers.append(Softmax())
    return model

def main():
    model = _read_model_from_file('/home/user/Desktop/Cyber/Secure Inference/neuronveil-mnist-train/artifacts/mnist.json')

    output = model.infer(myinput, FRACTION_BITS)

    # print(_re)
    pprint(output / 4)

if __name__ == '__main__':
    main()

