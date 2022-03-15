#!/bin/bash

set -ex

# export deepmd_root=$HOME/deepmd/deepmd-kit
source $deepmd_root/script/x86_64_cuda/env.sh
# bash $deepmd_root/script/x86_64/build_deepmd.sh

model_path=$deepmd_root/examples/copper/model
mkdir -p $model_path
mkdir -p $model_path/double

dp train ./input_v2_compat_train.json --restart model.ckpt
dp freeze -o ../model/double/graph.pb
dp test -m ../model/double/graph.pb -s ../data/iter.000047/02.fp/data.051 -n 1
dp compress -i ../model/double/graph.pb -o ../model/double/graph-compress.pb 