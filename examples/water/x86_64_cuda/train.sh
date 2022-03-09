#!/bin/bash


source $deepmd_root/script/x86_64_cuda/env.sh
bash $deepmd_root/script/x86_64_cuda/build_deepmd.sh

model_path=$deepmd_root/examples/water/model

set -ex

mkdir -p $model_path
mkdir -p $model_path/double

dp train ../se_e2_a/input.json
dp freeze -o $model_path/double/graph.pb
dp test -m $model_path/double/graph.pb -s ../data/data_3 -n 1
