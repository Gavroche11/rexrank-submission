from rexrank.utils.preprocessing import make_right_format
from rexrank.utils.build_dataset import MIMICInferenceDataset
from rexrank.classifier import load_model

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

from tqdm import tqdm

import json

CHEXPERT_CLASSES = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
                'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices', 'No Finding']

def get_right_input_json(input_json_file: str,
                         preprocessed_json_file: str,
                         img_root_dir: str,
                         classifier_model_name: str,
                         classifier_pretrained: str):

    new_dataset = make_right_format(input_json_file)

    # Get diagnoses
    model = load_model(model_name=classifier_model_name, pretrained=classifier_pretrained)
    inference_dataset = MIMICInferenceDataset(cleaned_dataset=new_dataset,
                                              img_root_dir=img_root_dir)

    print("Inference dataset:", len(inference_dataset))

    inference_dataloader = DataLoader(
        dataset=inference_dataset,
        batch_size=64,
        shuffle=False,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        study_ids = []
        preds = []

        for study_id, X in tqdm(inference_dataloader):
            X = X.to(device)
            pred = model(X)

            study_ids.extend(list(study_id))
            preds.append(F.sigmoid(pred).cpu().numpy())

    model = model.cpu()
    total_preds = np.concatenate(preds, axis=0)
    
    # Get df
    df = pd.DataFrame(new_dataset)
    df[CHEXPERT_CLASSES] = total_preds

    # make final input json
    total = []
    for _, row in tqdm(df.iterrows()):
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

        output_format["id"] = row["study_id"]
        output_format["image"] = row["image_path"]
        
        # prompt
        prompt = "Image: <image>"
        prompt += f"\nAge: {row['age']}"
        prompt += f"\nGender: {row['gender']}"
        prompt += f"\nDiagnostic Probabilities:"
        for cls in CHEXPERT_CLASSES:
            prompt += f"\n- {cls}: {row[cls]:.3f}"

        output_format['conversations'][0]['value'] = prompt

        # gt
        output_format['conversations'][1]['value'] = row['gt']

        total.append(output_format)
    
    with open(preprocessed_json_file, 'w') as f:
        json.dump(total, f)

    print(f"Saved to {preprocessed_json_file}")