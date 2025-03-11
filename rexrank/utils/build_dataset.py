import os

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from typing import List, Dict

def build_transform(transform_name: str = 'eva-x',
                    img_size: int = 448):
    
    if transform_name == 'eva-x':
        transform = build_eva_x_transform(img_size=img_size)
    else:
        raise ValueError(f"Invalid transform name: {transform_name}")
    
    return transform

def build_eva_x_transform(img_size: int):
    
    mean=(0.49185243, 0.49185243, 0.49185243)
    std=(0.28509309, 0.28509309, 0.28509309)

    t = []
    
    t.append(transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC))
    t.append(transforms.CenterCrop(img_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))

    return transforms.Compose(t)


class MIMICInferenceDataset(Dataset):
    def __init__(self,
                 cleaned_dataset: List[Dict[str, str]],
                 img_root_dir: str,
                 img_size: int = 448,
                 transform: str = 'eva-x'):

        self.data = cleaned_dataset
        self.img_root_dir = img_root_dir
        self.transform = build_transform(transform_name=transform, img_size=img_size)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        cur_data = self.data[idx]
        study_id = cur_data["study_id"]
        img_path = os.path.join(self.img_root_dir, cur_data["image_path"])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        return study_id, img