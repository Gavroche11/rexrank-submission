export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_P2P_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export NCCL_BLOCKING_WAIT=1 
export NCCL_ASYNC_ERROR_HANDLING=1

export HF_HOME=/home/data1/huggingface
export HUGGINGFACE_HUB_CACHE=/home/data1/huggingface
export TRANSFORMERS_CACHE=/home/data1/huggingface
export HF_DATASETS_CACHE=/home/data1/huggingface

LLM_VERSION="meta-llama/Llama-3.2-3B"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="eva-x-base-448"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
VISION_MODEL_PRETRAINED_WEIGHT="/home/data1/eva-x/pretrained_weights/eva_x_base_patch16_merged520k_mim.pt"

############### Pretrain ################

PROMPT_VERSION=plain

PRETRAIN_NAME="debug"

BASE_RUN_NAME="${PRETRAIN_NAME}-lora"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

DEVICES="0,1"

deepspeed --master_port 12345 --include localhost:${DEVICES} llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path /home/xray-mllm-develop/debug/debug.json \
    --eval_data_path /home/xray-mllm-develop/debug/debug.json \
    --image_aspect_ratio pad \
    --image_folder /home/BARO_Cluster/data/data \
    --pretrain_mm_mlp_adapter /home/data1/workspace/bih1122/llava_next_check/${PRETRAIN_NAME}/mm_projector.bin \
    --mm_tunable_parts "mm_mlp_adapter" \
    --vision_tower ${VISION_MODEL_VERSION} \
    --vision_tower_pretrained ${VISION_MODEL_PRETRAINED_WEIGHT} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /home/data1/workspace/bih1122/llava_next_check/${BASE_RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --eval_strategy "steps" \
    --eval_steps 5 \
    --save_strategy "steps" \
    --save_steps 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $BASE_RUN_NAME \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --attn_implementation sdpa \
    --lora_enable True \
    --bits 16 \
    --per_device_eval_batch_size 4

# You can delete the sdpa attn_implementation if you want to use flash attn