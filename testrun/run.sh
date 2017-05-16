#!/bin/bash

set -e
TOOLS=../build/tools

$TOOLS/caffe train \
    --solver=prototxt/solver.prototxt \
    --weights=model/init.caffemodel \
    --gpu=2 \
    2>&1 | tee log/train.log
