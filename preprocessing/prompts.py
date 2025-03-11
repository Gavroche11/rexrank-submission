import numpy as np
import pandas as pd
import yaml

from typing import Optional, Union, Any

prompt_templates = yaml.safe_load(open('preprocessing/prompt_templates.yaml', 'r'))

ABBREVIATIONS = {
    'Enlarged Cardiomediastinum': 'EC',
    'Cardiomegaly': 'CM',
    'Lung Opacity': 'LO',
    'Lung Lesion': 'LL',
    'Edema': 'ED',
    'Consolidation': 'CS',
    'Pneumonia': 'PN',
    'Atelectasis': 'AT',
    'Pneumothorax': 'PX',
    'Pleural Effusion': 'PE',
    'Pleural Other': 'PO',
    'Fracture': 'FX',
    'Support Devices': 'SD',
    'No Finding': 'NF',
}

def normalize_probabilities(probs: float, best_threshold: float) -> float:
    # odds ratio로 변환
    odds = probs / (1 - probs)
    best_odds = best_threshold / (1 - best_threshold)
    
    # best threshold가 0.5가 되도록 스케일 조정
    scaled_odds = odds * (1 / best_odds)
    
    # 다시 확률로 변환
    normalized_probs = scaled_odds / (1 + scaled_odds)
    
    return normalized_probs

def diagnosis_to_str(cur_preds: pd.Series,
                     style: Optional[str] = None,
                     thresholds: Optional[list[float]] = None) -> str:
    
    if style is None:
        out = ""
    
    elif style == 'plain':
        diagnosis_dict = {}
        for idx in cur_preds.index:
            finding = idx.replace('_pred', '')
            diagnosis_dict[finding] = round(cur_preds[idx], 3)
        out = '{}'.format(diagnosis_dict)

    elif style == 'itemize-prob':
        out = ''
        for idx in cur_preds.index:
            finding = idx.replace('_pred', '')
            prob = cur_preds[idx]
            out += f'\n- {finding}: {prob:.3f}'
    
    elif style.startswith('itemize-prob-except:'):
        exceptions = style.split(':')[-1].split(',')
        out = ''
        for idx in cur_preds.index:
            finding = idx.replace('_pred', '')
            if ABBREVIATIONS[finding] in exceptions:
                continue
            prob = round(cur_preds[idx], 3)
            out += f'\n- {finding}: {prob}'

    elif style.startswith('itemize-normalized-prob'):
        out = ''
        for i, (idx, thres) in enumerate(zip(cur_preds.index, thresholds)):
            finding = idx.replace('_pred', '')
            normalized_prob = round(normalize_probabilities(cur_preds[idx], thres), 3)
            out += f'\n- {finding}: {normalized_prob}'

    elif style == 'itemize-binary':
        assert thresholds is not None, "Thresholds must be provided for binary style"
        out = ''
        for i, idx in enumerate(cur_preds.index):
            finding = idx.replace('_pred', '')
            prob = cur_preds[idx]
            if prob >= thresholds[i]:
                out += f'\n- {finding}: [POSITIVE]'
            else:
                out += f'\n- {finding}: [NEGATIVE]'
    elif style == 'sentence':
        assert thresholds is not None
        positive_findings = []
        for i, idx in enumerate(cur_preds.index):
            finding = idx.replace('_pred', '')
            if finding == 'No Finding':
                continue
            prob = cur_preds[idx]
            if prob >= thresholds[i]:
                positive_findings.append(finding)
        
        if len(positive_findings) == 0:
            out = 'The image model indicates that this is a normal chest X-ray (CXR) with no detectable abnormalities.'
        elif len(positive_findings) == 1:
            out = f"The image model identifies {positive_findings[0]} in this chest X-ray (CXR)."
        elif len(positive_findings) == 2:
            out = f"The image model identifies {positive_findings[0]} and {positive_findings[1]} in this chest X-ray (CXR)."
        else:
            foo = ', '.join(positive_findings[:-1]) + f", and {positive_findings[-1]}"
            out = f"The image model identifies {foo} in this chest X-ray (CXR)."

    else:
        raise ValueError(f"Invalid style name: {style}")

    return out

def generate_prompt_from_diagnosis(row: pd.Series,
                                   preds: pd.DataFrame,
                                   prompt_template: dict,
                                   thresholds: Optional[list[float]] = None) -> str:
    object_id = row["object_id"]
    cols = preds.columns[preds.columns != 'object_id']
    cur_preds = preds.loc[preds['object_id'] == object_id, cols].squeeze()
    template, style = prompt_template['template'], prompt_template['diagnosis_style']

    others = prompt_template.get("others")

    if others == "age-gender":
        # Age
        age = str(int(row["age"]))
        if age == "-1":
            age = "N/A"
        template = template.replace("<age>", age)
        
        # Gender
        gender = row["gender"]
        if gender == "M":
            template = template.replace("<gender>", "Male")
        elif gender == "F":
            template = template.replace("<gender>", "Female")
        else:
            template = template.replace("<gender>", "N/A")

    diagnosis = diagnosis_to_str(cur_preds, style, thresholds)
    
    prompt = template.format(diagnosis)

    return prompt