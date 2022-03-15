#!/bin/bash

source $deepmd_root/script/x86_64_cuda/env.sh

python_file_path=$deepmd_root/_skbuild/linux-x86_64-3.7/cmake-install/deepmd/tools/pb2txt.py

pb_file_path=$deepmd_root/../model/water/double/compress

python $python_file_path $pb_file_path/graph-compress-baseline.pb $pb_file_path/graph-compress-baseline.pbtxt
