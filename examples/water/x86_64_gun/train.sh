#!/bin/bash


source $deepmd_root/script/x86_64_gnu/env.sh
bash $deepmd_root/script/x86_64_gnu/build_deepmd.sh

model_path=$deepmd_root/examples/water/model

set -ex

mkdir -p $model_path
mkdir -p $model_path/double

export OMP_NUM_THREADS=48

dp train ../se_e2_a/input_double_1000.json
dp freeze -o $model_path/double/graph.pb
dp test -m $model_path/double/graph.pb -s ../data/data_3 -n 1
