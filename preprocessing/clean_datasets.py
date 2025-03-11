import numpy as np
import pandas as pd
from typing import Literal, Union

DEBUG_LEN = 20

def is_empty(x: Union[str, int, float]) -> bool:
    if isinstance(x, str):
        x = x.strip()
        return x == '-1'
    elif isinstance(x, int):
        return x == -1
    else:
        return x == -1.0 or np.isnan(x)
    
def clean_mimic(ann_path: str,
                split: Literal["train", "val", "test"],
                debug: bool = False) -> pd.DataFrame:
    
    assert split in ['train', 'val', 'test'], f"Invalid split: {split}"
    data = pd.read_csv(ann_path)

    if split != 'test':
        # Select the values that satisfying the target mode
        data = data[data.loc[:, 'mm_split'] == split].reset_index(drop=True)
        
        # Select the values only view_position is AP or PA (i.e. frontal values)
        data = data[(data.loc[:, "view_position"] == "AP") | (data.loc[:, "view_position"] == "PA")]

        # Drop the empty rows
        data = data.loc[~(data['cxr_lite_findings_split'].apply(is_empty) & data['cxr_lite_impression_split'].apply(is_empty))].reset_index(drop=True)

        # Set the answer column
        data['answer'] = data['cxr_lite_findings_split']
        data.loc[data['answer'].apply(is_empty), 'answer'] = data.loc[data['answer'].apply(is_empty), 'cxr_lite_impression_split']
    else:
        assert 'object_id' in data.columns and 'answer' in data.columns, "Test data should have 'object_id' and 'answer' columns"

    # debug
    if debug:
        data = data.head(DEBUG_LEN)
    
    return data

def clean_chexpert(ann_path: str,
                    split: Literal["train", "val", "test"],
                    debug: bool = False) -> pd.DataFrame:
    
    assert split in ['train', 'val', 'test'], f"Invalid split: {split}"

    if split == 'val':
        split = 'valid'
    
    data = pd.read_csv(ann_path)

    if split != 'test':
        # Select the values that satisfying the target mode
        data = data[data.loc[:, "mm_split"] == split]
        
        # Select the values only view_position is AP or PA (i.e. frontal values)
        if 'view_position' in data.columns:
            data = data[(data.loc[:, "view_position"] == "AP") | (data.loc[:, "view_position"] == "PA")]
        else:
            data = data[(data.loc[:, "view_position_pred"] == "AP") | (data.loc[:, "view_position_pred"] == "PA")]

        # Drop the empty rows
        data = data.loc[~(data['cxr_lite_impression_split'].apply(is_empty))].reset_index(drop=True)

        # Set the answer column
        data['answer'] = data['cxr_lite_impression_split']
    else:
        assert 'object_id' in data.columns and 'answer' in data.columns, "Test data should have 'object_id' and 'answer' columns"

    # debug
    if debug:
        data = data.head(DEBUG_LEN)
    
    return data

def clean_openi(ann_path: str,
                split: Literal['test'],
                debug: bool = False) -> pd.DataFrame:
    
    assert split == 'test', f"Invalid mode: {split}"

    data = pd.read_csv(ann_path)
    assert 'object_id' in data.columns and 'answer' in data.columns, "Test data should have 'object_id' and 'answer' columns"

    # # Select the values that satisfying the target mode
    # data = data[data.loc[:, 'mm_split'] == split].reset_index(drop=True)

    # # Select the values only view_position is Frontal
    # data = data[data.loc[:, "view_position"] == "Frontal"]

    # # Drop the empty rows
    # data = data.loc[~(data['findings'].apply(is_empty) & data['impression'].apply(is_empty))].reset_index(drop=True)

    # # Set the answer column
    # data['answer'] = data['findings']
    # data.loc[data['answer'].apply(is_empty), 'answer'] = data.loc[data['answer'].apply(is_empty), 'impression']

    # debug
    if debug:
        data = data.head(DEBUG_LEN)

    return data