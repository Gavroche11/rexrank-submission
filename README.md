# DD-LLaVA-X ReXrank Submission

This repository contains the submission of **DD-LLaVA-X** to [ReXrank](https://rexrank.ai/), developed by [Inhyeok Baek](https://github.com/Gavroche11) of [iRAIL](http://irail.snu.ac.kr/).

## Prerequisites

- CUDA 12.1.1 with cuDNN 8
- Python 3.10
- Conda
- Git

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Gavroche11/rexrank-submission.git
cd rexrank-submission
```

2. **Set up the Python environment:**
```bash
conda create -n llava python=3.10 -y
conda activate llava
```

3. **Install dependencies:**
```bash
pip install --upgrade pip
pip install -e ".[train]"
pip install transformers==4.47.0
pip install xformers==0.0.23.post1
pip install flash-attn==2.7.2.post1 --no-build-isolation --no-cache-dir
```

4. **Get access to huggingface models**

Get access to [`meta-llama/Meta-Llama-3-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) and [`meta-llama/Llama-3.2-1B`](https://huggingface.co/meta-llama/Llama-3.2-1B).

## Model Weights

Download the model weights from [here](https://drive.google.com/drive/folders/1SlP4XyGo73JAI74x9TvVWbQcMHI88snM?usp=sharing). You will have the file `vision_encoder.pth` and the folder `language_model_checkpoint`.

After downloading the weights, define `CLASSIFIER_PRETRAINED`, `VISION_TOWER_PRETRAINED`, and `LORA_OUTPUT_DIR` in `inference.py` as follows:
```python
CLASSIFIER_PRETRAINED = "/path/to/vision_encoder.pth"
...
VISION_TOWER_PRETRAINED = "/path/to/vision_encoder.pth"
...
LORA_OUTPUT_DIR = "/path/to/language_model_checkpoint"
```

## Running Inference

To run inference on your data:
```bash
python inference.py \
    --input_json_file <path_to_input.json> \
    --output_json_file <path_to_output.json> \
    --img_root_dir <path_to_image_directory>
```

### Parameters:
- `input_json_file`: Path to the input JSON file containing queries
- `output_json_file`: Path where the results will be saved
- `img_root_dir`: Directory containing the input images