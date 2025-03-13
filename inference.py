import argparse
import os
import json
from tqdm import tqdm
import torch
from dataclasses import dataclass
from typing import Dict

from llava.eval.eval_finetune import setup_model, disable_torch_init, create_data_loader
from rexrank.preprocess_rexrank import get_right_llava_input

# Classifier

CLASSIFIER = "eva-x"
CLASSIFIER_PRETRAINED = "/model/workspace/bih1122/model_weights/vis-encoder-cls/vis-v2.4-best/best.pth"

LLAVA_INPUT_DIR = "rexrank/llava_input"

# Llava eval configs

MODEL_NAME = "meta-llama/Llama-3.2-1B"
VERSION = "plain"

VISION_TOWER = "eva-x-base-448"
VISION_TOWER_PRETRAINED = "/model/workspace/bih1122/model_weights/vis-encoder-cls/vis-v2.4-best/best.pth" # must be changed

MAX_NEW_TOKENS = 128

LORA_OUTPUT_DIR = "/model/workspace/bih1122/llava_next_check/v2.9-full/checkpoint-6692" # must be changed

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
        self.max_new_tokens = kwargs.get('max_new_tokens', 128)
        self.attn_implementation = "flash_attention_2"
        
        # Data Configuration
        self.image_folder = kwargs.get('image_folder', None)
        self.image_aspect_ratio = kwargs.get('image_aspect_ratio', 'pad')
        self.question_file = kwargs.get('question_file', None)
        self.answers_file = kwargs.get('answers_file', None)
        
        # Generation Configuration
        self.temperature = kwargs.get('temperature', 0.2)
        self.top_p = kwargs.get('top_p', None)
        self.num_beams = kwargs.get('num_beams', 1)
        
        # Weight Loading
        self.pretrain_mm_mlp_adapter = kwargs.get('pretrain_mm_mlp_adapter', None)
        self.lora_output_dir = kwargs.get('lora_output_dir', LORA_OUTPUT_DIR)

def eval_model(args: Arguments) -> Dict[str, Dict[str, str]]:
    # Model initialization
    model, tokenizer, image_processor = setup_model(args)
    disable_torch_init()
    
    # Load questions
    with open(os.path.expanduser(args.question_file)) as f:
        questions = json.load(f)
    
    # Prepare output file
    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)
    
    # Create data loader
    data_loader = create_data_loader(
        questions, 
        args.image_folder, 
        tokenizer, 
        image_processor, 
        model.config
    )
    
    # Run inference
    results_dict = {}

    for (input_ids, image_tensors, image_sizes, answers), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["id"]
        cur_prompt = line["conversations"][0]["value"]

        input_ids = input_ids.to(device=args.device, non_blocking=True)

        with torch.inference_mode():
            output_ids = model.generate(
                inputs=input_ids,
                images=image_tensors.to(dtype=torch.bfloat16, device=args.device, non_blocking=True),
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        results_dict[idx] = {
            "model_prediction": outputs
        }

    return results_dict
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json_file', type=str, default="rexrank/datasets/ReXRank_MIMICCXR_test.json")
    parser.add_argument('--save_json_file', type=str, default="rexrank/outputs/mimic_outputs.json")
    parser.add_argument('--img_root_dir', type=str, required=True)
    args = parser.parse_args()

    if "mimic" in args.input_json_file.lower():
        llava_input_json = os.path.join(LLAVA_INPUT_DIR, "mimic_llava_input.json")
    elif "iuxray" in args.input_json_file.lower():
        llava_input_json = os.path.join(LLAVA_INPUT_DIR, "iuxray_llava_input.json")
    elif "chexpert" in args.input_json_file.lower():
        llava_input_json = os.path.join(LLAVA_INPUT_DIR, "chexpert_llava_input.json")
    else:
        llava_input_json = os.path.join(LLAVA_INPUT_DIR, "llava_input.json")
    
    raw_input = json.load(open(args.input_json_file, "r"))
    get_right_llava_input(input_json_file=args.input_json_file,
                         llava_input_json=llava_input_json,
                         img_root_dir=args.img_root_dir,
                         classifier_model_name=CLASSIFIER,
                         classifier_pretrained=CLASSIFIER_PRETRAINED)

    args = Arguments(
        image_folder=args.img_root_dir,
        question_file=llava_input_json,
        answers_file=args.save_json_file
    )

    model_predictions = eval_model(args)
    for study_id, data in raw_input.items():
        data["model_prediction"] = model_predictions[study_id]["model_prediction"]

    with open(args.save_json_file, "w") as f:
        json.dump(raw_input, f, indent=4)

    