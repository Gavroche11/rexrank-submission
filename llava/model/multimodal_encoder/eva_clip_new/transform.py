from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torchvision.transforms.functional as F

import torchvision
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, \
    CenterCrop

from transformers.image_processing_utils import BatchFeature

from PIL import Image

from .constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD


class ResizeMaxSize(nn.Module):

    def __init__(self, max_size, interpolation=InterpolationMode.BICUBIC, fn='max', fill=0):
        super().__init__()
        if not isinstance(max_size, int):
            raise TypeError(f"Size should be int. Got {type(max_size)}")
        self.max_size = max_size
        self.interpolation = interpolation
        self.fn = min if fn == 'min' else min
        self.fill = fill

    def forward(self, img):
        if isinstance(img, torch.Tensor):
            height, width = img.shape[:2]
        else:
            width, height = img.size
        scale = self.max_size / float(max(height, width))
        if scale != 1.0:
            new_size = tuple(round(dim * scale) for dim in (height, width))
            img = F.resize(img, new_size, self.interpolation)
            pad_h = self.max_size - new_size[0]
            pad_w = self.max_size - new_size[1]
            img = F.pad(img, padding=[pad_w//2, pad_h//2, pad_w - pad_w//2, pad_h - pad_h//2], fill=self.fill)
        return img


def _convert_to_rgb(image):
    return image.convert('RGB')


class BaseProcessor:
    def __init__(self):
        self.transform = lambda x: x
        return

    def __call__(self, item):
        return self.transform(item)


class EvaClipImageBaseProcessor(BaseProcessor):
    def __init__(self, mean=None, std=None):
        self.mean = OPENAI_DATASET_MEAN if mean is None else mean
        if not isinstance(self.mean, (list, tuple)):
            self.mean = (self.mean,) * 3

        self.std = OPENAI_DATASET_STD if std is None else std
        if not isinstance(self.std, (list, tuple)):
            self.std = (self.std,) * 3

        self.normalize = torchvision.transforms.Normalize(self.mean, self.std)

    @property
    def image_mean(self):
        return self.mean
    
    @property
    def image_std(self):
        return self.std


class EvaClipImageTrainProcessor(EvaClipImageBaseProcessor):
    def __init__(self,
                 image_size: int = 448,
                 mean: Optional[Tuple[float, ...]] = None,
                 std: Optional[Tuple[float, ...]] = None,
                 resize_longest_max: bool = False,
                 fill_color: int = 0
                 ):
        
        super().__init__(mean=mean, std=std)

        if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
            image_size = image_size[0]

        self.transform = Compose(
            [
                RandomResizedCrop(image_size, scale=(0.9, 1.0), interpolation=InterpolationMode.BICUBIC),
                ToTensor(),
            ]
        )

        self.image_size = image_size

    def preprocess(self, images, return_tensors):
        if isinstance(images, Image.Image):
            images = [images]
        else:
            assert isinstance(images, list)

        transformed_images = [self.transform(image).numpy() for image in images]
        data = {"pixel_values": transformed_images}

        return BatchFeature(data=data, tensor_type=return_tensors)

    def __call__(self, item):
        return self.transform(item)

    @property
    def crop_size(self):
        return {"height": self.image_size, "width": self.image_size}

    @property
    def size(self):
        return {"shortest_edge": self.image_size}



# class CatGen(nn.Module):
#     def __init__(self, num=4):
#         self.num = num
#     def mixgen_batch(image, text):
#         batch_size = image.shape[0]
#         index = np.random.permutation(batch_size)

#         cat_images = []
#         for i in range(batch_size):
#             # image mixup
#             image[i,:] = lam * image[i,:] + (1 - lam) * image[index[i],:]
#             # text concat
#             text[i] = tokenizer((str(text[i]) + " " + str(text[index[i]])))[0]
#         text = torch.stack(text)
#         return image, text


def image_transform(
        image_size: int,
        is_train: bool,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        resize_longest_max: bool = False,
        fill_color: int = 0,
):
    mean = mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3

    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
        image_size = image_size[0]

    normalize = Normalize(mean=mean, std=std)
    if is_train:
        return Compose([
            RandomResizedCrop(image_size, scale=(0.9, 1.0), interpolation=InterpolationMode.BICUBIC),
            ToTensor(),
        ])
    else:
        if resize_longest_max:
            transforms = [
                ResizeMaxSize(image_size, fill=fill_color)
            ]
        else:
            transforms = [
                Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(image_size),
            ]
        transforms.extend([
            ToTensor(),
        ])
        return Compose(transforms)
