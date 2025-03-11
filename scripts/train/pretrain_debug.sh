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
# VISION_MODEL_VERSION="eva-x-base-448"
VISION_MODEL_VERSION="EVA02-CLIP-L-14-448"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
# VISION_MODEL_PRETRAINED_WEIGHT="/home/data1/eva-x/pretrained_weights/eva_x_base_patch16_merged520k_mim.pt"
VISION_MODEL_PRETRAINED_WEIGHT="/home/BARO_Cluster/data/data/research/model/llm2clip/pytorch_model.bin"

############### Pretrain ################

PROMPT_VERSION=plain

BASE_RUN_NAME="debug6"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

DEVICES="0,1,2,3"

deepspeed --include localhost:${DEVICES} --master_port 12345 llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path /home/BARO_Cluster/data/data/llava_input/v3.0/mimic+chexpert-train.json \
    --eval_data_path /home/BARO_Cluster/data/data/llava_input/v3.0/mimic-val.json \
    --image_aspect_ratio pad \
    --image_folder /home/BARO_Cluster/data/data \
    --vision_tower ${VISION_MODEL_VERSION} \
    --vision_tower_pretrained ${VISION_MODEL_PRETRAINED_WEIGHT} \
    --mm_tunable_parts "mm_mlp_adapter" \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /home/data1/workspace/bih1122/llava_next_check/${BASE_RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
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
    --dataloader_drop_last True \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $BASE_RUN_NAME \
    --attn_implementation sdpa \
    --per_device_eval_batch_size 4 \
    --lora_enable False \

# You can delete the sdpa attn_implementation if you want to use flash attn

    # --per_device_eval_batch_size 2 \