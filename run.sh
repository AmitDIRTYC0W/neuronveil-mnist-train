#!/bin/sh

docker build -t neuronveil_mnist .

mkdir -p ./artifacts
docker run -v "$(pwd)"/artifacts:/artifacts neuronveil_mnist

