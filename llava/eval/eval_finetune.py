import argparse
import torch
import os
import json
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from peft import PeftModel
from llava.utils import MedicalImagePreprocessor

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from llava.model import *

import warnings
warnings.filterwarnings("ignore")

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

@dataclass
class Arguments:
    def __init__(self, **kwargs):
        # Model Configuration
        self.model_name_or_path = kwargs.get('model_name_or_path', None)
        self.model_class_name = kwargs.get('model_class_name', "LlavaLlama")
        self.version = kwargs.get('version', 'plain')
        
        # Vision Configuration
        self.vision_tower = kwargs.get('vision_tower', None)
        self.vision_tower_pretrained = kwargs.get('vision_tower_pretrained', None)
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
        self.max_new_tokens = kwargs.get('max_new_tokens', 512)
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
        self.lora_output_dir = kwargs.get('lora_output_dir', None)

        # etc
        self.convert_to_csv = kwargs.get('convert_to_csv', True)

def setup_model(args: Arguments):
    compute_dtype = (torch.float16 if args.fp16 
                    else (torch.bfloat16 if args.bf16 
                    else torch.float32))

    print("Load weights via LlavaLlamaForCausalLM...")
    # model = LlavaLlamaForCausalLM.from_pretrained(
    #     args.lora_output_dir,
    #     cache_dir=args.cache_dir,
    #     attn_implementation=args.attn_implementation,
    #     torch_dtype=(torch.bfloat16 if args.bf16 else None),
    #     low_cpu_mem_usage=False
    # )

    # if args.lora_output_dir is not None:
    #     args.pretrain_mm_mlp_adapter = os.path.join(args.lora_output_dir, "non_lora_trainables.bin")

    model = LlavaLlamaForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        attn_implementation=args.attn_implementation,
        torch_dtype=(torch.bfloat16 if args.bf16 else None),
        low_cpu_mem_usage=False
    )

    if args.lora_output_dir is not None:
        if os.path.exists(os.path.join(args.lora_output_dir, "mm_projector.bin")):
            args.pretrain_mm_mlp_adapter = os.path.join(args.lora_output_dir, "mm_projector.bin")
        else:
            args.pretrain_mm_mlp_adapter = os.path.join(args.lora_output_dir, "non_lora_trainables.bin")

        print(f"Load LoRA adapter from {args.lora_output_dir}")
        model = PeftModel.from_pretrained(
            model,
            args.lora_output_dir,
            torch_dtype=compute_dtype,
            device_map={"": args.device}
        )

###################################
        # Load non_lora_trainables weights
        non_lora_state_dict = torch.load(args.pretrain_mm_mlp_adapter, map_location='cpu')
        
        # Filter out language model weights from non_lora_trainables
        lm_state_dict = {k: v for k, v in non_lora_state_dict.items() 
                        if not 'mm_projector' in k and not 'vision_tower' in k}
        new_lm_state_dict = {}
        for k, v in lm_state_dict.items():
            if k.startswith('module.'):
                new_lm_state_dict[k[7:]] = v
            else:
                new_lm_state_dict[k] = v
        # Update the model's state dict with language model weights
        model_state_dict = model.state_dict()
        model_state_dict.update(new_lm_state_dict)
        model.load_state_dict(model_state_dict, strict=False)
###################################

    model.get_model().initialize_vision_modules(
        args,
        fsdp=None
    )
    
    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=compute_dtype, device=args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    model.config.mm_use_im_start_end = args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = args.mm_use_im_patch_token
    
    image_processor = vision_tower.image_processor
    model = model.to(dtype=compute_dtype, device=args.device)
    
    return model, tokenizer, image_processor

class CustomDataset(Dataset):
    def __init__(self, conversations, image_folder, tokenizer, image_processor, model_config):
        self.conversations = conversations
        self.image_folder = image_folder
        self.image_aspect_ratio = "pad"
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.medical_image_preprocessor = MedicalImagePreprocessor()
    def __getitem__(self, index):
        line = self.conversations[index]
        image_files = line["image"]
        qs = line["conversations"][0]["value"]
        answer = line["conversations"][1]["value"]

        # if self.model_config.mm_use_im_start_end:
        #     # qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        #     pass
        # else:
        #     # qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        #     pass

        conv = conv_templates["plain"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image_tensors = []

        if isinstance(image_files, list):
            for image_file in image_files:
                image_pth = os.path.join(self.image_folder, image_file)
                image = self.medical_image_preprocessor.preprocess(image_pth, do_windowing=False)
                image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values']
                image_tensors.append(image_tensor.half().cuda())
        elif isinstance(image_files, str):
            image_pth = os.path.join(self.image_folder, image_file)
            image = self.medical_image_preprocessor.preprocess(image_pth, do_windowing=False)
            if self.image_aspect_ratio == "pad":
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result

                image = expand2square(image, tuple(int(x * 255) for x in self.image_processor.image_mean))
            image_tensors = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        else:
            raise ValueError(f"Unexpected image_files type: {type(image_files)}")

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensors, image.size, answer

    def __len__(self):
        return len(self.conversations)

def collate_fn(batch):
    input_ids, image_tensors, image_sizes, answers = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    answers = list(answers)
    return input_ids, image_tensors, image_sizes, answers

def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader

def eval_model(args: Arguments):
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

    results_list = []

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
        
        # ans_file.write(json.dumps({
        #     "question_id": idx,
        #     "prompt": cur_prompt,
        #     "text": outputs,
        #     "answer": answers,
        #     "model_id": "llava_llama",
        #     "metadata": {}
        # }) + "\n")

        results_list.append({
            "question_id": idx,
            "prompt": cur_prompt,
            "text": outputs,
            "answer": answers,
            "model_id": "llava_llama",
            "metadata": {}
        })

        print("\n")
        print("#"*50)
        print("pred")
        print(outputs)
        print("\n")
        print("gt")
        print(answers[0])
        print("#"*50)
        print("\n")
        
    with open(args.answers_file, 'w') as f:
        json.dump(results_list , f)

    print("Saved test outputs to ", args.answers_file)

    if args.convert_to_csv:
        import pandas as pd
        df = pd.DataFrame(results_list)
        df = df[['question_id', 'prompt', 'text', 'answer']]
        df = df.rename(columns={'question_id': 'object_id', 'answer': 'gt', 'text': 'pred'})
        df['gt'] = df['gt'].apply(lambda x: x[0])
        df.to_csv(args.answers_file.replace('.json', '.csv'), index=False)

        print("Saved test outputs csv to ", args.answers_file.replace('.json', '.csv'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model Configuration
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--model_class_name", type=str, default="LlavaLlama")
    parser.add_argument("--version", type=str, default="plain")
    
    # Vision Configuration
    parser.add_argument("--vision_tower", type=str, required=True)
    parser.add_argument("--vision_tower_pretrained", type=str, required=True)
    parser.add_argument("--mm_vision_select_layer", type=int, default=-2)
    
    # Projection Configuration
    parser.add_argument("--mm_projector_type", type=str, default="mlp2x_gelu")
    parser.add_argument("--mm_use_im_start_end", type=bool, default=False)
    parser.add_argument("--mm_use_im_patch_token", type=bool, default=False)
    
    # Training/Hardware Configuration
    parser.add_argument("--bf16", type=bool, default=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--model_max_length", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    
    # Data Configuration
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--image_aspect_ratio", type=str, default="pad")
    parser.add_argument("--question_file", type=str, required=True)
    parser.add_argument("--answers_file", type=str, required=True)
    
    # Generation Configuration
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    
    # Weight Loading
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str,
                       help="Path to the trained mm_projector.bin weights")
    parser.add_argument("--lora_output_dir", type=str, help="Directory path to the trained lora weights")

    # etc
    parser.add_argument("--convert_to_csv", type=bool, default=True)

    args = Arguments(**vars(parser.parse_args()))
    eval_model(args)