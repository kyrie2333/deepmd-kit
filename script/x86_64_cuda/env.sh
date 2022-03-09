#!/bin/bash 

export deepmd_root=/home/zhongyun/deepmd/deepmd-kit

if [ -z $deepmd_root ]
then
    echo "not found envoriment variable : deepmd_root"
fi

# export tensorflow_root=/vol0004/hp200266/u01036/gzq/fj_software/tensorflow/TensorFlow-2.2.0
export tensorflow_root=$deepmd_root/../dependents/tensorflow-gpu-2.4

lammps_version=stable_29Sep2021
export lammps_root=$deepmd_root/../dependents/lammps-$lammps_version

export LD_LIBRARY_PATH=$deepmd_root/lib:$LD_LIBRARY_PATH
export CPATH=$deepmd_root/include:$CPATH
export PATH=$deepmd_root/bin:$PATH

source $tensorflow_root/env_cuda.sh

export DP_VARIANT=cuda

. /usr/local/spack/share/spack/setup-env.sh
spack load gcc@7.5.0
spack load cmake
spack load openmpi
spack load openblas@0.3.18

export CC="gcc -Ofast -fopenmp -lopenblas"
export CXX="g++ -Ofast -fopenmp -lopenblas"