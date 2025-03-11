import os
import json
import numpy as np
import pandas as pd

import yaml

from tqdm.auto import tqdm
from typing import Literal, Optional

from clean_datasets import clean_mimic, clean_chexpert, clean_openi
from prompts import generate_prompt_from_diagnosis

import argparse


def generate_outputs(dataset: Literal["mimic", "chexpert", "openi"],
                     split: Literal["train", "val", "test"],
                     ann_path: str,
                     img_dir: str,
                     preds: pd.DataFrame,
                     prompt_template: dict,
                     thresholds: Optional[list[float]] = None,
                     debug: bool = False) -> list[dict]:
    
    assert dataset in ["mimic", "chexpert", "openi"], dataset
    assert split in ["train", "val", "test"], split

    if dataset == "mimic":
        data = clean_mimic(ann_path, split, debug)
    elif dataset == "chexpert":
        data = clean_chexpert(ann_path, split, debug)
    elif dataset == "openi":
        data = clean_openi(ann_path, split, debug)

    outputs = []

    print(dataset, split, len(data))

    for idx, row in tqdm(data.iterrows(), total=len(data)):
        output_format = {
            "id": "",
            "image": "",
            "conversations": [
                {
                    "from": "human",
                    "value": ""
                },
                {
                    "from": "gpt",
                    "value": ""
                },
            ]
        }
        
        output_format["id"] = row["object_id"]
        output_format["image"] = os.path.join(img_dir, row["object_id"] + ".png")

        # Prompt
        output_format['conversations'][0]['value'] = generate_prompt_from_diagnosis(row=row, preds=preds, prompt_template=prompt_template, thresholds=thresholds)

        # Answer
        output_format['conversations'][1]['value'] = row['answer'].strip()
    
        outputs.append(output_format)

    return outputs

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--train_datasets", "--train", nargs='*', default=['mimic', 'chexpert'])
    parser.add_argument("--val_datasets", "--val", nargs='*', default=['mimic'])
    parser.add_argument("--test_datasets", "--test", nargs='*', default=['mimic', 'chexpert', 'openi'])

    args = parser.parse_args()
    
    # Load the config
    cfg = yaml.safe_load(open(args.cfg, 'r'))
    model_version = cfg['model_version']
    preds_version = cfg['preds_version']
    preds_dir = cfg['preds_dir']
    prompt_version = cfg['prompt_version']
    save_base_dir = cfg['save_base_dir']
    save_dir = os.path.join(save_base_dir, model_version)
    os.makedirs(save_dir, exist_ok=True)

    if 'thresholds_path' in cfg:
        thresholds = list(np.load(cfg['thresholds_path']))
    else:
        thresholds = None

    dataset_paths = yaml.safe_load(open('preprocessing/datasets.yaml', 'r'))
    prompt_templates = yaml.safe_load(open('preprocessing/prompt_templates.yaml', 'r'))
    cur_prompt_template = prompt_templates[prompt_version]

    ## Train datasets
    if len(args.train_datasets) != 0:
        total_outputs = []
        for dataset_name in args.train_datasets:
            preds = pd.read_csv(os.path.join(preds_dir, f"{dataset_name}_diagnoses.csv"))
            outputs = generate_outputs(
                dataset=dataset_name,
                split='train',
                ann_path=dataset_paths[dataset_name]['ann_path'],
                img_dir=dataset_paths[dataset_name]['img_dir'],
                preds=preds,
                prompt_template=cur_prompt_template,
                thresholds=thresholds,
                debug=args.debug
            )
            total_outputs.extend(outputs)

        # basename = "+".join(args.train_datasets) + f"--model-{model_version}--preds-{preds_version}--prompt-{prompt_version}--train.json"
        basename = "+".join(args.train_datasets) + "-train.json"

        save_path = os.path.join(save_dir, basename)
        with open(save_path, "w") as f:
            json.dump(total_outputs, f)

        print(f'Saved the train output json file to "{save_path}".')
    else:
        print("No train datasets are selected.")

    ## Val datasets
    if len(args.val_datasets) != 0:
        total_outputs = []
        for dataset_name in args.val_datasets:
            preds = pd.read_csv(os.path.join(preds_dir, f"{dataset_name}_diagnoses.csv"))
            outputs = generate_outputs(
                dataset=dataset_name,
                split='val',
                ann_path=dataset_paths[dataset_name]['ann_path'],
                img_dir=dataset_paths[dataset_name]['img_dir'],
                preds=preds,
                prompt_template=cur_prompt_template,
                thresholds=thresholds,
                debug=args.debug
            )
            total_outputs.extend(outputs)

        # basename = "+".join(args.train_datasets) + f"--model-{model_version}--preds-{preds_version}--prompt-{prompt_version}--val.json"
        basename = "+".join(args.val_datasets) + "-val.json"

        save_path = os.path.join(save_dir, basename)
        with open(save_path, "w") as f:
            json.dump(total_outputs, f)

        print(f'Saved the val output json file to "{save_path}".')
    else:
        print("No val datasets are selected.")

    ## Test datasets
    if len(args.test_datasets) != 0:
        total_outputs = []
        for dataset_name in args.test_datasets:
            preds = pd.read_csv(os.path.join(preds_dir, f"{dataset_name}_diagnoses.csv"))
            outputs = generate_outputs(
                dataset=dataset_name,
                split='test',
                ann_path=dataset_paths[dataset_name]['test_ann_path'],
                img_dir=dataset_paths[dataset_name]['img_dir'],
                preds=preds,
                prompt_template=cur_prompt_template,
                thresholds=thresholds,
                debug=args.debug
            )

            # basename = f"{dataset_name}--model-{model_version}--preds-{preds_version}--prompt-{prompt_version}--test.json"
            basename = f"{dataset_name}-test.json"

            save_path = os.path.join(save_dir, basename)
            with open(save_path, "w") as f:
                json.dump(outputs, f)

            print(f'Saved the test output json file to "{save_path}".')
            total_outputs.extend(outputs)

        if len(args.test_datasets) > 1:
            basename = "+".join(args.test_datasets) + "-test.json"
            save_path = os.path.join(save_dir, basename)
            with open(save_path, "w") as f:
                json.dump(total_outputs, f)

            print(f'Saved the total test output json file to "{save_path}".')
        
    else:
        print("No test datasets are selected.")


if __name__ == "__main__":
    main()