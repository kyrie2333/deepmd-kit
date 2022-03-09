#pragma once

#include <string>
// using namespace std;

#ifdef HIGH_PREC
const std::string global_float_prec="double";
#else 
const std::string global_float_prec="float";
#endif

const std::string global_install_prefix="/home/zhongyun/deepmd/deepmd-kit-offical";
const std::string global_git_summ="v2.1.0";
const std::string global_git_hash="3e54fea";
const std::string global_git_date="2022-03-07 11:41:41 +0800";
const std::string global_git_branch="master";
const std::string global_tf_include_dir="/home/zhongyun/deepmd/dependents/tensorflow-gpu-2.4/include;/home/zhongyun/deepmd/dependents/tensorflow-gpu-2.4/include";
const std::string global_tf_lib="/home/zhongyun/deepmd/dependents/tensorflow-gpu-2.4/lib/libtensorflow_cc.so;/home/zhongyun/deepmd/dependents/tensorflow-gpu-2.4/lib/libtensorflow_framework.so";
const std::string global_model_version="1.1 ";
