# ReXrank Submission

This repository is to submit **LLaVA-EVA-X** to ReXrank, developed by Inhyeok Baek of iRAIL.

## Installation

### 1. **You must work on cuda:12.1.1-cudnn8**

### 2. **Clone this repository and navigate to the folder**
```bash
git clone https://github.com/Gavroche11/rexrank-submission.git
cd rexrank-submission
```

### 3. **Install the relevant packages:**
```bash
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip
pip install -e ".[train]"
pip install transformers==4.47.0
pip install xformers==0.0.23.post1
pip install flash-attn==2.7.2.post1 --no-build-isolation --no-cache-dir
```

## Get model weights
```bash
TODO
```

## Inference
```bash
python inference.py --input_json_file <input_json_file> --output_json_file <output_json_file> --img_root_dir <img_root_dir>
```