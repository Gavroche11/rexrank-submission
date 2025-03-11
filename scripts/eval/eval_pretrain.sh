#!/bin/bash

export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO

LLM_VERSION="meta-llama/Llama-3.2-1B"
VISION_MODEL_VERSION="eva-x-base-448"
VISION_MODEL_PRETRAINED="/home/data1/workspace/bih1122/model_weights/vis-encoder-cls/vis-v2.4/2024-12-17-01-25-46/best.pth"

BASE_RUN_NAME="v2.3"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

DATASET="openi" # mimic, chexpert, openi
echo "DATASET: ${DATASET}"

python llava/eval/eval_pretrain.py \
    --model_name_or_path ${LLM_VERSION} \
    --version ${BASE_RUN_NAME} \
    --vision_tower ${VISION_MODEL_VERSION} \
    --vision_tower_pretrained ${VISION_MODEL_PRETRAINED} \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --device cuda:0 \
    --cache_dir /home/data1/huggingface \
    --model_max_length 2048 \
    --max_new_tokens 128 \
    --image_aspect_ratio pad \
    --image_folder /home/BARO_Cluster/data/data \
    --question_file /home/BARO_Cluster/data/data/llava_input/${BASE_RUN_NAME}/${DATASET}-test.json \
    --answers_file /home/data1/workspace/bih1122/llava_test_outputs/${BASE_RUN_NAME}/${DATASET}_test_outputs.json \
    --pretrain_mm_mlp_adapter /home/data1/workspace/bih1122/llava_next_check/${BASE_RUN_NAME}/checkpoint-3334/mm_projector.bin \
    --convert_to_csv True