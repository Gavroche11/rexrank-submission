import argparse
import os
import json
from tqdm import tqdm
import torch
from dataclasses import dataclass
from typing import Dict

from llava.eval.eval_finetune import ModelManager, ModelEvaluator, disable_torch_init

# Classifier
CLASSIFIER = "eva-x"
CLASSIFIER_PRETRAINED = "/home/data1/workspace/bih1122/model_weights/vis-encoder-cls/vis-v2.4/2024-12-17-01-25-46/best.pth"

LLAVA_INPUT_DIR = "rexrank/llava_input"

# Llava eval configs
MODEL_NAME = "meta-llama/Llama-3.2-1B"
VERSION = "plain"

VISION_TOWER = "eva-x-base-448"
VISION_TOWER_PRETRAINED = "/home/data1/workspace/bih1122/model_weights/vis-encoder-cls/vis-v2.4/2024-12-17-01-25-46/best.pth"

MAX_NEW_TOKENS = 128
BATCH_SIZE = 8

LORA_OUTPUT_DIR = ""  # Set to None if not using LoRA

@dataclass
class Arguments:
    def __init__(self, **kwargs):
        # Model Configuration
        self.model_name_or_path = kwargs.get('model_name_or_path', MODEL_NAME)
        self.model_class_name = kwargs.get('model_class_name', "LlavaLlama")
        self.version = kwargs.get('version', 'plain')
        
        # Vision Configuration
        self.vision_tower = kwargs.get('vision_tower', VISION_TOWER)
        self.vision_tower_pretrained = kwargs.get('vision_tower_pretrained', VISION_TOWER_PRETRAINED)
        self.mm_vision_select_layer = kwargs.get('mm_vision_select_layer', -2)
        self.mm_patch_merge_type = "flat"
        self.mm_vision_select_feature = "patch"
        
        # Projection Configuration
        self.mm_projector_type = kwargs.get('mm_projector_type', 'mlp2x_gelu')
        self.mm_use_im_start_end = kwargs.get('mm_use_im_start_end', False)
        self.mm_use_im_patch_token = kwargs.get('mm_use_im_patch_token', False)
        
        # Training/Hardware Configuration
        self.bf16 = kwargs.get('bf16', True)
        self.fp16 = kwargs.get('fp16', False)
        self.device = kwargs.get('device', 'cuda')
        self.local_rank = kwargs.get('local_rank', -1)
        self.cache_dir = kwargs.get('cache_dir', None)
        self.model_max_length = kwargs.get('model_max_length', 8192)
        self.max_new_tokens = kwargs.get('max_new_tokens', MAX_NEW_TOKENS)
        self.batch_size = kwargs.get('batch_size', BATCH_SIZE)
        self.attn_implementation = "flash_attention_2"
        
        # Data Configuration
        self.image_folder = kwargs.get('image_folder', None)
        self.image_aspect_ratio = kwargs.get('image_aspect_ratio', 'pad')
        self.input_json_file = kwargs.get('input_json_file', None)
        self.answers_file = kwargs.get('answers_file', None)
        
        # Classifier Configuration
        self.classifier_model_name = kwargs.get('classifier_model_name', CLASSIFIER)
        self.classifier_pretrained = kwargs.get('classifier_pretrained', CLASSIFIER_PRETRAINED)
        
        # Generation Configuration
        self.temperature = kwargs.get('temperature', 0.2)
        self.top_p = kwargs.get('top_p', None)
        self.num_beams = kwargs.get('num_beams', 1)
        
        # Weight Loading
        self.pretrain_mm_mlp_adapter = kwargs.get('pretrain_mm_mlp_adapter', None)
        self.lora_output_dir = kwargs.get('lora_output_dir', LORA_OUTPUT_DIR if LORA_OUTPUT_DIR else None)
        
        # Miscellaneous
        self.convert_to_csv = kwargs.get('convert_to_csv', True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json_file', type=str, default="rexrank/datasets/ReXRank_MIMICCXR_test.json")
    parser.add_argument('--save_json_file', type=str, default="rexrank/outputs/mimic_outputs.json")
    parser.add_argument('--img_root_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--model_name_or_path', type=str, default=MODEL_NAME)
    parser.add_argument('--vision_tower', type=str, default=VISION_TOWER)
    parser.add_argument('--vision_tower_pretrained', type=str, default=VISION_TOWER_PRETRAINED)
    parser.add_argument('--classifier_pretrained', type=str, default=CLASSIFIER_PRETRAINED)
    parser.add_argument('--lora_output_dir', type=str, default=LORA_OUTPUT_DIR)
    args = parser.parse_args()

    # Check if paths exist
    required_paths = [
        ('vision_tower_pretrained', args.vision_tower_pretrained),
        ('classifier_pretrained', args.classifier_pretrained)
    ]
    
    if args.lora_output_dir:
        required_paths.append(('lora_output_dir', args.lora_output_dir))
    
    for name, path in required_paths:
        if path and not os.path.exists(path):
            print(f"Warning: {name} path does not exist: {path}")
            if name in ['vision_tower_pretrained', 'classifier_pretrained']:
                raise ValueError(f"Required path {name} does not exist: {path}")
    
    # Disable torch init
    disable_torch_init()

    # Set up args for model evaluation
    model_args = Arguments(
        model_name_or_path=args.model_name_or_path,
        vision_tower=args.vision_tower,
        vision_tower_pretrained=args.vision_tower_pretrained,
        image_folder=args.img_root_dir,
        input_json_file=args.input_json_file,
        answers_file=args.save_json_file,
        batch_size=args.batch_size,
        classifier_pretrained=args.classifier_pretrained,
        lora_output_dir=args.lora_output_dir if args.lora_output_dir else None
    )

    # Set up model manager and load models
    model_manager = ModelManager(model_args)
    model, tokenizer, image_processor = model_manager.setup_model()
    
    # Create evaluator and run evaluation
    evaluator = ModelEvaluator(model_args, model, tokenizer, image_processor)
    evaluator.evaluate()
    
    print(f"Results saved to {args.save_json_file}")

    