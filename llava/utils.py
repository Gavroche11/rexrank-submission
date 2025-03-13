import datetime
import logging
import logging.handlers
import os
import sys
import numpy as np
from PIL import Image
import numpy as np
from typing import Tuple, Optional
import requests

from llava.constants import LOGDIR

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "I am sorry. Your input may violate our content moderation guidelines. Please avoid using harmful or offensive content."

handler = None

import torch.distributed as dist

try:
    import av
    from decord import VideoReader, cpu
except ImportError:
    print("Please install pyav to use video processing functions.")



class MedicalImagePreprocessor:
    def __init__(self, width_param: float = 4.0, target_size: int = 512):
        """
        Initialize the medical image preprocessor.
        
        Args:
            width_param: Parameter for windowing, default is 4.0
            target_size: Target size for the longest dimension, default is 512
        """
        self.width_param = width_param
        self.target_size = target_size

    def apply_windowing(self, image: np.ndarray, do_windowing: bool = False) -> np.ndarray:
        """
        Apply windowing to the medical image.
        
        Args:
            image: Input image as numpy array
            do_windowing: Whether to apply windowing transformation
            
        Returns:
            Windowed image
        """
        if not do_windowing:
            img_min = np.min(image)
            img_max = np.max(image)
            image = (image - img_min) / (img_max - img_min + 1e-8)
            return image

        # Convert to float
        image = image.astype(float)
        
        # Calculate window parameters
        mean = np.mean(image)
        std = np.std(image)
        window_center = mean
        window_width = self.width_param * std
        
        # Apply windowing
        img_min = window_center - window_width/2
        img_max = window_center + window_width/2
        image = np.clip(image, img_min, img_max)
        
        # Normalize to [0, 1]
        image = (image - img_min) / (img_max - img_min + 1e-8)
        return image

    def remove_black_padding(self, image: Image.Image) -> Image.Image:
        """
        Removes black padded space from an X-ray image.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Cropped PIL Image without black padding
        """
        # Convert to grayscale if not already
        if image.mode != 'L':
            image = image.convert('L')
            
        # Get bounding box of non-zero pixels
        bbox = image.getbbox()
        
        if bbox is None:
            return image
            
        # Add margin if desired
        margin = 0
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(image.width, x2 + margin)
        y2 = min(image.height, y2 + margin)
        
        # Crop image
        return image.crop((x1, y1, x2, y2))

    def preprocess(self, image_path: str, do_windowing: bool = True) -> np.ndarray:
        """
        Performs complete preprocessing pipeline on a medical image.
        
        Args:
            image_path: Path to the input image
            do_windowing: Whether to apply windowing transformation
            
        Returns:
            Preprocessed image as numpy array
        """
        # Read image
        try:
            image = Image.open(image_path)
        except Exception as e:
            raise ValueError(f"Could not read image at {image_path}: {str(e)}")

        # Convert to numpy array for windowing
        image_array = np.array(image)
        
        # Apply windowing
        image_array = self.apply_windowing(image_array, do_windowing)
        
        # Convert back to PIL Image for padding removal
        image = Image.fromarray((image_array * 255).astype(np.uint8))
        
        # Remove padding
        processed = self.remove_black_padding(image)
        # Convert back to numpy array and normalize
        
        return processed.convert('RGB')

def process_video_with_decord(video_file, data_args):
    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    avg_fps = round(vr.get_avg_fps() / data_args.video_fps)
    frame_idx = [i for i in range(0, total_frame_num, avg_fps)]
    
    if data_args.frames_upbound > 0:
        if len(frame_idx) > data_args.frames_upbound:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, data_args.frames_upbound, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
    
    video = vr.get_batch(frame_idx).asnumpy()
    # https://github.com/dmlc/decord/issues/208
    vr.seek(0)
    return video

def process_video_with_pyav(video_file, data_args):
    container = av.open(video_file)
    # !!! This is the only difference. Using auto threading
    container.streams.video[0].thread_type = "AUTO"

    video_frames = []
    for packet in container.demux():
        if packet.stream.type == 'video':
            for frame in packet.decode():
                video_frames.append(frame)
    total_frame_num = len(video_frames)
    video_time = video_frames[-1].time
    avg_fps = round(total_frame_num / video_time / data_args.video_fps)
    frame_idx = [i for i in range(0, total_frame_num, avg_fps)]

    if data_args.frames_upbound > 0:
        if len(frame_idx) > data_args.frames_upbound:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, data_args.frames_upbound, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()


    frames = [video_frames[i] for i in frame_idx]
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)


def rank_print(*args):
    if dist.is_initialized():
        print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)

def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(filename, when="D", utc=True)
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ""
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == "\n":
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != "":
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ""


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def violates_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}
    text = text.replace("\n", "")
    data = "{" + '"input": ' + f'"{text}"' + "}"
    data = data.encode("utf-8")
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        print(f"######################### Moderation Error: {e} #########################")
        flagged = False
    except KeyError as e:
        print(f"######################### Moderation Error: {e} #########################")
        flagged = False

    return flagged


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"
