import argparse
from dataclasses import dataclass
from llava.eval.eval_finetune import Arguments, eval_model

from rexrank.preprocess_rexrank import get_right_input_json

# arguments:
# parser.add_argument('--input_json_file', type=str,default='../../data/mimic-cxr/test_data.json')
# parser.add_argument('--save_json_file', type=str,default='../../results/mimic-cxr/MedVersa.json')
# parser.add_argument('--img_root_dir', type=str,default='../../data/mimic-cxr/MIMIC-CXR-JPG/files')

MODEL_NAME = "meta-llama/Llama-3.2-1B"
VERSION = "plain"

CLASSIFIER = "eva-x"
CLASSIFIER_PRETRAINED = "" # must be changed

VISION_TOWER = "eva-x-base-448"
VISION_TOWER_PRETRAINED = "/home/data1/workspace/bih1122/model_weights/vis-encoder-cls/vis-v2.4/2024-12-17-01-25-46/best.pth" # must be changed

MAX_NEW_TOKENS = 128

LORA_OUTPUT_DIR = "" # must be changed

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

        # etc
        self.convert_to_csv = kwargs.get('convert_to_csv', True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json_file', type=str, default="rexrank/datasets/ReXRank_MIMICCXR_test.json")
    parser.add_argument('--save_json_file', type=str, default="rexrank/submission/mimic_outputs.json")
    parser.add_argument('--img_root_dir', type=str, required=True)
    args = parser.parse_args()
    
    get_right_input_json(input_json_file=args.input_json_file,
                         preprocessed_json_file="rexrank/preprocessed/mimic_inference.json",
                         img_root_dir=args.img_root_dir,
                         classifier_model_name=CLASSIFIER,
                         classifier_pretrained=CLASSIFIER_PRETRAINED)

    args = Arguments(
        image_folder=args.img_root_dir,
        question_file="rexrank/preprocessed/mimic_inference.json",
        answers_file=args.save_json_file
    )
    # eval_model(args)