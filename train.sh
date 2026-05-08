#!/bin/sh
PARTITION=Segmentation


GPU_ID=0
dataset=iSAID # iSAID/LoveDA

exp_name=split0

arch=FVL_HFA 

net=resnet50 # vgg resnet50

model=model/${arch}.py
Cocontrast=model/ProtoContrastModule.py
exp_dir=exp/FVL_HFA_iSAID${arch}/${exp_name}/${net}
snapshot_dir=${exp_dir}/snapshot
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}_resnet50_DMNet.yaml
#config=config/${dataset}/${dataset}_${exp_name}_vgg16_DMNet.yaml
mkdir -p ${snapshot_dir} ${result_dir}
now=$(date +"%Y%m%d_%H%M%S")
cp train.sh train.py  ${model}  ${Cocontrast} ${config} ${exp_dir}

echo ${arch}
echo ${config}

# =========== 添加以下调试信息 ===========
echo "======================================"
echo "DEBUG INFO before launching Python:"
echo "--------------------------------------"
echo "which python: $(which python)"
echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "CUDA_HOME: $CUDA_HOME" # 检查 CUDA_HOME 变量
echo "CONDA_PREFIX: $CONDA_PREFIX" # Conda 环境的路径
echo "======================================"
# ==========================================

CUDA_VISIBLE_DEVICES=${GPU_ID} $(which python) train.py \
    --config=${config} \
    --arch=${arch} \
    2>&1 | tee ${result_dir}/train-$now.log
