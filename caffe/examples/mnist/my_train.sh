#!/usr/bin/env sh
set -e

./build/tools/caffe time --model=examples/mnist/lenet_train_test.prototxt
#./build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt $@
