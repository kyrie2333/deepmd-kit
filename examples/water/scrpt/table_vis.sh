#!/bin/bash

source $deepmd_root/script/x86_64_cuda/env.sh

python_file_path=$deepmd_root/_skbuild/linux-x86_64-3.7/cmake-install/deepmd/tools/table_visualization.py

cp $deepmd_root/deepmd/tools/table_visualization.py $python_file_path

python $python_file_path $deepmd_root/../model/water/double/compress/graph-compress-baseline.pb