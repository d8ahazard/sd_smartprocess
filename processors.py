import gc
import random
from typing import List, Any

import torch
from PIL import Image
from tqdm import tqdm

from extensions.sd_dreambooth_extension.dreambooth.utils.utils import printm
from extensions.sd_smartprocess import super_resolution
from extensions.sd_smartprocess.clipcrop import CropClip
from extensions.sd_smartprocess.interrogators.clip_interrogator import CLIPInterrogator
from extensions.sd_smartprocess.interrogators.booru_interrogator import BooruInterrogator
from modules import shared

# Base processor
class Processor:
    def __init__(self):
        printm("Model loaded.")

    # Unload models
    def unload(self):
        if torch.has_cuda:
            torch.cuda.empty_cache()
        gc.collect()
        printm("Model unloaded.")

    # Process images
    def process(self, images: List[Image.Image]) -> List[Any]:
        raise Exception("Not Implemented")

# CLIP Processing
class ClipProcessor(Processor):
    def __init__(
            self,
            clip_use_v2,
            clip_append_artist,
            clip_append_medium,
            clip_append_movement,
            clip_append_flavor,
            clip_append_trending,
            num_beams,
            min_clip_tokens,
            max_clip_tokens,
            max_flavors
    ):
        self.description = "Processing CLIP"
        if shared.interrogator is not None:
            shared.interrogator.unload()

        self.max_flavors = max_flavors
        shared.state.textinfo = "Loading CLIP Model..."
        self.model = CLIPInterrogator(
            clip_use_v2,
            clip_append_artist,
            clip_append_medium,
            clip_append_movement,
            clip_append_flavor,
            clip_append_trending,
            num_beams,
            min_clip_tokens,
            max_clip_tokens
        )
        super().__init__()

    def process(self, images: List[Image.Image], short:bool=False) -> List[str]:
        output = []
        shared.state.job_count = len(images)
        shared.state.textinfo = f"{self.description}..."
        for img in tqdm(images, desc=self.description):
            short_caption = self.model.interrogate(img, short=short, max_flavors=self.max_flavors)
            output.append(short_caption)
            shared.state.current_image = img
            shared.state.job_no += 1
        return output

    def unload(self):
        if self.model.clip_model:
            del self.model.clip_model
        if self.model.blip_model:
            del self.model.blip_model
        super().unload()

# Danbooru Processing
class BooruProcessor(Processor):
    def __init__(self, min_score: float):
        self.description = "Processing Danbooru"
        shared.state.textinfo = "Loading DeepDanbooru Model..."
        self.model = BooruInterrogator()
        self.min_score = min_score
        super().__init__()

    def process(self, images: List[Image.Image]) -> List[List[str]]:
        output = []
        shared.state.job_count = len(images)
        shared.state.textinfo = f"{self.description}..."
        for img in tqdm(images, desc=self.description):
            out_tags = []
            tags = self.model.interrogate(img)
            for tag in sorted(tags, key=tags.get, reverse=True):
                if tags[tag] >= self.min_score:
                    out_tags.append(tag)
            output.append(out_tags)
            shared.state.job_count += 1

    def unload(self):
        self.model.unload()
        super().unload()

# WD14 Processing

# Crop Processing
class CropProcessor(Processor):
    def __init__(self, subject_class: str, pad: bool, crop: bool):
        self.description = "Cropping"
        if crop:
            shared.state.textinfo = "Loading CROP Model..."
        self.model = CropClip() if crop else None
        self.subject_class = subject_class
        self.pad = pad
        self.crop = crop
        super().__init__()

    def process(self, images: List[Image.Image], captions: List[str] = None) -> List[Image.Image]:
        output = []
        shared.state.job_count = len(images)
        shared.state.textinfo = f"{self.description}..."
        for img, caption in tqdm(zip(images, captions), desc=self.description):
            cropped = self._process_img(img, caption)
            output.append(cropped)
            shared.state.job_no += 1
        return output


    def _process_img(self, img, short_caption):
        if self.subject_class is not None and self.subject_class != "":
            short_caption = self.subject_class

        src_ratio = img.width / img.height

        # Pad image before cropping?
        if src_ratio != 1 and self.pad:
            if img.width > img.height:
                pad_width = img.width
                pad_height = img.width
            else:
                pad_width = img.height
                pad_height = img.height
            res = Image.new("RGB", (pad_width, pad_height))
            res.paste(img, box=(pad_width // 2 - img.width // 2, pad_height // 2 - img.height // 2))
            img = res

        if self.crop:
            # Do the actual crop clip
            im_data = self.model.get_center(img, prompt=short_caption)
            crop_width = im_data[1] - im_data[0]
            center_x = im_data[0] + (crop_width / 2)
            crop_height = im_data[3] - im_data[2]
            center_y = im_data[2] + (crop_height / 2)
            crop_ratio = crop_width / crop_height
            dest_ratio = 1
            tgt_width = crop_width
            tgt_height = crop_height

            if crop_ratio != dest_ratio:
                if crop_width > crop_height:
                    tgt_height = crop_width / dest_ratio
                    tgt_width = crop_width
                else:
                    tgt_width = crop_height / dest_ratio
                    tgt_height = crop_height

                # Reverse the above if dest is too big
                if tgt_width > img.width or tgt_height > img.height:
                    if tgt_width > img.width:
                        tgt_width = img.width
                        tgt_height = tgt_width / dest_ratio
                    else:
                        tgt_height = img.height
                        tgt_width = tgt_height / dest_ratio

            tgt_height = int(tgt_height)
            tgt_width = int(tgt_width)
            left = max(center_x - (tgt_width / 2), 0)
            right = min(center_x + (tgt_width / 2), img.width)
            top = max(center_y - (tgt_height / 2), 0)
            bottom = min(center_y + (tgt_height / 2), img.height)
            img = img.crop((left, top, right, bottom))
        return img

    def unload(self):
        if self.model is not None:
            self.model.unload()
            super().unload()

# Upscale Processing
class UpscaleProcessor(Processor):
    def __init__(self):
        self.description = "Upscaling"
        shared.state.textinfo = "Loading Stable-Diffusion Upscaling Model..."
        self.sampler, self.model = super_resolution.initialize_model()
        super().__init__()

    def process(self, images: List[Image.Image], captions: List[str] = None) -> List[Image.Image]:
        output = []
        shared.state.job_count = len(images)
        shared.state.textinfo = f"{self.description}..."
        for img, caption in tqdm(zip(images, captions), desc=self.description):
            seed = int(random.randrange(2147483647))
            img = super_resolution.predict(self.sampler, img, caption, 75, 1, 10, seed, 0, 20)
            output.append(img)
            shared.state.job_no += 1
        return output

    def unload(self):
        del self.sampler
        del self.model
        super().unload()

