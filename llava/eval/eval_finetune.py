import argparse
import torch
import os
import json
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any, Union
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from peft import PeftModel
from llava.utils import MedicalImagePreprocessor, disable_torch_init
from rexrank.preprocess_rexrank import get_batch_llava_input

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import tokenizer_image_token, process_images
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from llava.model import *

import warnings
warnings.filterwarnings("ignore")

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

@dataclass
class Arguments:
    # Model Configuration
    model_name_or_path: str
    model_class_name: str = "LlavaLlama"
    version: str = 'plain'
    
    # Vision Configuration
    vision_tower: str = None
    vision_tower_pretrained: str = None
    mm_vision_select_layer: int = -2
    mm_patch_merge_type: str = "flat"
    mm_vision_select_feature: str = "patch"
    
    # Projection Configuration
    mm_projector_type: str = 'mlp2x_gelu'
    mm_use_im_start_end: bool = False
    mm_use_im_patch_token: bool = False
    
    # Training/Hardware Configuration
    bf16: bool = True
    fp16: bool = False
    device: str = 'cuda'
    local_rank: int = -1
    cache_dir: str = None
    model_max_length: int = 8192
    max_new_tokens: int = 512
    attn_implementation: str = "flash_attention_2"
    batch_size: int = 8
    
    # Data Configuration
    image_folder: str = None
    image_aspect_ratio: str = 'pad'
    input_json_file: str = None
    answers_file: str = None
    
    # Classifier Configuration
    classifier_model_name: str = "eva-x"
    classifier_pretrained: str = None
    
    # Generation Configuration
    temperature: float = 0.2
    top_p: float = None
    num_beams: int = 1
    
    # Weight Loading
    pretrain_mm_mlp_adapter: str = None
    lora_output_dir: str = None
    
    # Miscellaneous
    convert_to_csv: bool = True


class ImageProcessor:
    """Helper class for image preprocessing operations"""
    
    def __init__(self, processor, aspect_ratio="pad"):
        self.processor = processor
        self.aspect_ratio = aspect_ratio
        self.medical_preprocessor = MedicalImagePreprocessor()
    
    def preprocess_single_image(self, image_file: str, image_folder: str) -> Tuple[torch.Tensor, Any]:
        """Process a single image file and return tensor and size"""
        image_path = os.path.join(image_folder, image_file)
        image = self.medical_preprocessor.preprocess(image_path, do_windowing=False)
        
        if self.aspect_ratio == "pad":
            image = self._expand_to_square(image)
            
        image_tensor = self.processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        return image_tensor, image.size if hasattr(image, 'size') else None
    
    def preprocess_multiple_images(self, image_files: List[str], image_folder: str) -> List[Tuple[torch.Tensor, Any]]:
        """Process multiple image files and return tensors and sizes"""
        results = []
        for image_file in image_files:
            tensor, size = self.preprocess_single_image(image_file, image_folder)
            results.append((tensor, size))
        return results
    
    def _expand_to_square(self, pil_img: Image.Image) -> Image.Image:
        """Expand an image to a square with padding"""
        width, height = pil_img.size
        if width == height:
            return pil_img
        
        background_color = tuple(int(x * 255) for x in self.processor.image_mean)
        
        if width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            
        return result


class ModelManager:
    """Manages model loading and initialization"""
    
    def __init__(self, args: Arguments):
        self.args = args
    
    def setup_model(self):
        """Initialize the model, tokenizer, and image processor"""
        compute_dtype = (torch.float16 if self.args.fp16 
                        else (torch.bfloat16 if self.args.bf16 
                        else torch.float32))

        print("Loading model...")
        model = LlavaLlamaForCausalLM.from_pretrained(
            self.args.model_name_or_path,
            cache_dir=self.args.cache_dir,
            attn_implementation=self.args.attn_implementation,
            torch_dtype=(torch.bfloat16 if self.args.bf16 else None),
            low_cpu_mem_usage=False
        )

        if self.args.lora_output_dir is not None:
            self._setup_lora_weights(model, compute_dtype)

        model.get_model().initialize_vision_modules(
            self.args,
            fsdp=None
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=compute_dtype, device=self.args.device)

        tokenizer = self._setup_tokenizer()
        
        model.config.mm_use_im_start_end = self.args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = self.args.mm_use_im_patch_token
        
        image_processor = vision_tower.image_processor
        model = model.to(dtype=compute_dtype, device=self.args.device)
        
        return model, tokenizer, image_processor
    
    def _setup_lora_weights(self, model, compute_dtype):
        """Set up LoRA adapter weights"""
        if os.path.exists(os.path.join(self.args.lora_output_dir, "mm_projector.bin")):
            self.args.pretrain_mm_mlp_adapter = os.path.join(self.args.lora_output_dir, "mm_projector.bin")
        else:
            self.args.pretrain_mm_mlp_adapter = os.path.join(self.args.lora_output_dir, "non_lora_trainables.bin")

        print(f"Loading LoRA adapter from {self.args.lora_output_dir}")
        model = PeftModel.from_pretrained(
            model,
            self.args.lora_output_dir,
            torch_dtype=compute_dtype,
            device_map={"": self.args.device}
        )
        
        # Load non_lora_trainables weights
        non_lora_state_dict = torch.load(self.args.pretrain_mm_mlp_adapter, map_location='cpu')
        
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
    
    def _setup_tokenizer(self):
        """Set up the tokenizer"""
        tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name_or_path,
            cache_dir=self.args.cache_dir,
            model_max_length=self.args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
            
        return tokenizer


class CustomDataset(Dataset):
    """Dataset for batch processing with prompt generation"""
    
    def __init__(self, raw_data, image_folder, tokenizer, image_processor, model_config, 
                 classifier_model_name=None, classifier_pretrained=None):
        self.raw_data = raw_data
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        
        # Store classifier information for batch processing
        self.classifier_model_name = classifier_model_name
        self.classifier_pretrained = classifier_pretrained
        self.study_ids = list(raw_data.keys())
        
    def __getitem__(self, index):
        study_id = self.study_ids[index]
        study_data = self.raw_data[study_id]
        
        # Return the raw study data for batch processing in collate_fn
        return {
            "id": study_id,
            "data": study_data,
            "image_folder": self.image_folder
        }

    def __len__(self):
        return len(self.study_ids)


def collate_fn(batch_items, tokenizer, image_processor, model_config, 
               classifier_model_name, classifier_pretrained):
    """Process a batch of items to prepare model inputs"""
    # Extract batch data
    batch_raw_data = {item["id"]: item["data"] for item in batch_items}
    image_folder = batch_items[0]["image_folder"]  # Same for all items
    
    # Process the batch to generate prompts and prepare images
    processed_batch = get_batch_llava_input(
        batch_raw_data,
        image_folder,
        classifier_model_name,
        classifier_pretrained
    )
    
    # Now prepare model inputs from processed data
    input_ids_list = []
    image_tensors_list = []
    image_sizes_list = []
    answers_list = []
    
    img_processor = ImageProcessor(image_processor)
    
    for item in processed_batch:
        # Extract data from the processed item
        image_files = item["image"]
        qs = item["conversations"][0]["value"]
        answer = item["conversations"][1]["value"] if len(item["conversations"]) > 1 else ""
        
        # Create conversation prompt
        conv = conv_templates["plain"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # Process images
        if isinstance(image_files, list):
            processed_images = img_processor.preprocess_multiple_images(image_files, image_folder)
            image_tensors = [tensor.half().cuda() for tensor, _ in processed_images]
            image_sizes = [size for _, size in processed_images]
            if len(image_sizes) == 1:
                image_sizes = image_sizes[0]
        elif isinstance(image_files, str):
            tensor, size = img_processor.preprocess_single_image(image_files, image_folder)
            image_tensors = tensor
            image_sizes = size
        else:
            raise ValueError(f"Unexpected image_files type: {type(image_files)}")
        
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        
        # Append to batch lists
        input_ids_list.append(input_ids)
        image_tensors_list.append(image_tensors)
        image_sizes_list.append(image_sizes)
        answers_list.append(answer)
    
    # Stack tensors where appropriate
    input_ids = torch.stack(input_ids_list, dim=0)
    
    # Handle different tensor structures
    if all(isinstance(tensor, torch.Tensor) and tensor.dim() == 3 for tensor in image_tensors_list):
        image_tensors = torch.stack(image_tensors_list, dim=0)
    else:
        # Keep as list if tensors have different shapes
        image_tensors = image_tensors_list
    
    return input_ids, image_tensors, image_sizes_list, answers_list


def create_data_loader(raw_data, image_folder, tokenizer, image_processor, model_config, 
                      classifier_model_name=None, classifier_pretrained=None,
                      batch_size=8, num_workers=4):
    """Create a data loader for batch processing"""
    dataset = CustomDataset(
        raw_data, 
        image_folder, 
        tokenizer, 
        image_processor, 
        model_config,
        classifier_model_name,
        classifier_pretrained
    )
    
    # Create a partially applied collate function with the required parameters
    partial_collate_fn = lambda batch: collate_fn(
        batch, 
        tokenizer, 
        image_processor, 
        model_config,
        classifier_model_name,
        classifier_pretrained
    )
    
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=False, 
        collate_fn=partial_collate_fn
    )
    return data_loader


class ModelEvaluator:
    """Handles model evaluation and results processing"""
    
    def __init__(self, args: Arguments, model, tokenizer, image_processor):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        
    def evaluate(self):
        """Run model evaluation on the dataset"""
        # Load raw input data directly
        with open(os.path.expanduser(self.args.input_json_file)) as f:
            raw_data = json.load(f)
        
        # Prepare output directory
        os.makedirs(os.path.dirname(self.args.answers_file), exist_ok=True)
        
        # Create data loader with raw data and classifier info
        data_loader = create_data_loader(
            raw_data, 
            self.args.image_folder, 
            self.tokenizer, 
            self.image_processor, 
            self.model.config,
            classifier_model_name=self.args.classifier_model_name,
            classifier_pretrained=self.args.classifier_pretrained,
            batch_size=self.args.batch_size
        )
        
        # Run inference
        results_dict = {}

        for input_ids, image_tensors, image_sizes, answers in tqdm(data_loader, total=len(data_loader)):
            # Move input_ids to device
            input_ids = input_ids.to(device=self.args.device, non_blocking=True)
            
            # Process images
            if isinstance(image_tensors, torch.Tensor):
                processed_images = image_tensors.to(
                    dtype=torch.bfloat16, 
                    device=self.args.device, 
                    non_blocking=True
                )
            else:
                # Handle list of tensors
                processed_images = image_tensors

            # Generate output
            with torch.inference_mode():
                output_ids = self.model.generate(
                    inputs=input_ids,
                    images=processed_images,
                    image_sizes=image_sizes,
                    do_sample=True if self.args.temperature > 0 else False,
                    temperature=self.args.temperature,
                    top_p=self.args.top_p,
                    num_beams=self.args.num_beams,
                    max_new_tokens=self.args.max_new_tokens,
                    use_cache=True
                )

            # Decode outputs
            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            
            # Process each item in the batch
            self._process_batch_results(data_loader.dataset, outputs, answers, results_dict)
        
        # Update the original raw data with model predictions
        for study_id, pred_info in results_dict.items():
            raw_data[study_id]["model_prediction"] = pred_info["model_prediction"]
        
        # Save results
        with open(self.args.answers_file, 'w') as f:
            json.dump(raw_data, f, indent=4)

        print(f"Saved test outputs to {self.args.answers_file}")
    
    def _process_batch_results(self, dataset, outputs, answers, results_dict):
        """Process and store batch results"""
        for i, output in enumerate(outputs):
            study_id = dataset.study_ids[i]
            
            results_dict[study_id] = {
                "model_prediction": output.strip()
            }
            
            print("\n")
            print("#"*50)
            print("pred")
            print(output.strip())
            print("\n")
            print("gt")
            print(answers[i])
            print("#"*50)
            print("\n")


def main():
    """Main function to parse arguments and run evaluation"""
    parser = argparse.ArgumentParser(description="LLaVA model evaluation")
    
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
    parser.add_argument("--mm_use_im_start_end", action="store_true")
    parser.add_argument("--mm_use_im_patch_token", action="store_true")
    
    # Training/Hardware Configuration
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--model_max_length", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    
    # Data Configuration
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--image_aspect_ratio", type=str, default="pad")
    parser.add_argument("--input_json_file", type=str, required=True)
    parser.add_argument("--answers_file", type=str, required=True)
    
    # Classifier Configuration
    parser.add_argument("--classifier_model_name", type=str, default="eva-x")
    parser.add_argument("--classifier_pretrained", type=str, required=True)
    
    # Generation Configuration
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    
    # Weight Loading
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str,
                       help="Path to the trained mm_projector.bin weights")
    parser.add_argument("--lora_output_dir", type=str, 
                       help="Directory path to the trained lora weights")

    # Miscellaneous
    parser.add_argument("--convert_to_csv", action="store_true", default=True)

    args = parser.parse_args()
    
    # Convert to Arguments dataclass for better type checking
    args_obj = Arguments(
        model_name_or_path=args.model_name_or_path,
        model_class_name=args.model_class_name,
        version=args.version,
        vision_tower=args.vision_tower,
        vision_tower_pretrained=args.vision_tower_pretrained,
        mm_vision_select_layer=args.mm_vision_select_layer,
        mm_projector_type=args.mm_projector_type,
        mm_use_im_start_end=args.mm_use_im_start_end,
        mm_use_im_patch_token=args.mm_use_im_patch_token,
        bf16=args.bf16,
        fp16=args.fp16,
        device=args.device,
        local_rank=args.local_rank,
        cache_dir=args.cache_dir,
        model_max_length=args.model_max_length,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        image_folder=args.image_folder,
        image_aspect_ratio=args.image_aspect_ratio,
        input_json_file=args.input_json_file,
        answers_file=args.answers_file,
        classifier_model_name=args.classifier_model_name,
        classifier_pretrained=args.classifier_pretrained,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        pretrain_mm_mlp_adapter=args.pretrain_mm_mlp_adapter,
        lora_output_dir=args.lora_output_dir,
        convert_to_csv=args.convert_to_csv
    )
    
    # Disable initialization
    disable_torch_init()
    
    # Set up model manager and load models
    model_manager = ModelManager(args_obj)
    model, tokenizer, image_processor = model_manager.setup_model()
    
    # Create evaluator and run evaluation
    evaluator = ModelEvaluator(args_obj, model, tokenizer, image_processor)
    evaluator.evaluate()


if __name__ == "__main__":
    main()