#!/bin/sh
PARTITION=Segmentation



GPU_ID=0

dataset=LoveDA # iSAID/ LoveDA

exp_name=split0

arch=FVL_HFA # DMNet DMNet_CLIP DMNet_CLIP_MoE
visualize=True

net=resnet50 # vgg resnet50

exp_dir=exp/FVL_HFA_iSAID/${dataset}/${arch}/${exp_name}/${net}
snapshot_dir=${exp_dir}/snapshot
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}_resnet50_DMNet.yaml
mkdir -p ${snapshot_dir} ${result_dir}
now=$(date +"%Y%m%d_%H%M%S")
cp test.sh test.py ${config} ${exp_dir}

echo ${arch}
echo ${config}
echo ${visualize}

USE_BLIP2_TEXT="--use_blip2_text"
SAVE_BLIP2_TEXT="--save_blip2_text"
BLIP2_TEXT_SAVE_PATH="--blip2_text_save_path=${exp_dir}/blip2_texts"

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -u test.py \
        --config=${config} \
        --arch=${arch} \
        --visualize \
        ${USE_BLIP2_TEXT} \
        ${SAVE_BLIP2_TEXT} \
        ${BLIP2_TEXT_SAVE_PATH} \
        2>&1 | tee ${result_dir}/test-$now.log