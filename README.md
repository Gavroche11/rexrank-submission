# LLaVA-EVA-X ReXrank Submission

This repository contains the submission of **LLaVA-iRAIL** to ReXrank, developed by Inhyeok Baek of [iRAIL](http://irail.snu.ac.kr/).

## Prerequisites

- CUDA 12.1.1 with cuDNN 8
- Python 3.10
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

## Model Weights

Download the model weights:
```bash
# Instructions for downloading model weights will be added
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