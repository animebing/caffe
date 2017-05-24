#!/bin/bash

set -e
TOOLS=/home/bingbing/git/yjxiong/caffe/build/tools

mpirun -np 3 $TOOLS/caffe train \
    --solver=prototxt/solver.prototxt \
    --weights=model/init.caffemodel \
    2>&1 | tee log/train_val_sync.log
