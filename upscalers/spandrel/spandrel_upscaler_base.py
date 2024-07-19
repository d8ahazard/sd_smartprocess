from __future__ import annotations

import logging
import os
import re
import time
import traceback
import zipfile
from abc import abstractmethod
from pathlib import Path
from urllib.request import urlretrieve

import PIL
import cv2
import numpy as np
import torch
from PIL import Image
from spandrel import (
    ImageModelDescriptor,
    ModelDescriptor, ModelLoader,
)

from modules import shared
from modules.upscaler import Upscaler, UpscalerData, NEAREST

logger = logging.getLogger(__name__)


def convert_google_drive_link(url: str) -> str:
    pattern = re.compile(
        r"^https://drive.google.com/file/d/([a-zA-Z0-9_\-]+)/view(?:\?.*)?$"
    )
    m = pattern.match(url)
    if not m:
        return url
    file_id = m.group(1)
    return "https://drive.google.com/uc?export=download&confirm=1&id=" + file_id


def download_file(url: str, filename: Path | str) -> None:
    filename = Path(filename)
    filename.parent.mkdir(exist_ok=True)

    url = convert_google_drive_link(url)

    temp_filename = filename.with_suffix(f".part-{int(time.time())}")

    try:
        logger.info("Downloading %s to %s", url, filename)
        path, _ = urlretrieve(url, filename=temp_filename)
        temp_filename.rename(filename)
    finally:
        try:
            temp_filename.unlink()
        except FileNotFoundError:
            pass


def extract_file_from_zip(
        zip_path: Path | str,
        rel_model_path: str,
        filename: Path | str,
):
    filename = Path(filename)
    filename.parent.mkdir(exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        with open(filename, "wb") as f:
            f.write(zip_ref.read(rel_model_path))


def image_to_tensor(img: np.ndarray, device: str, half) -> torch.Tensor:
    img = img.astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    if img.shape[2] == 1:
        pass
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img).to(device)
    if half is not None:
        tensor = tensor.to(half)
    return tensor.unsqueeze(0)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.cpu().squeeze().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = np.clip((image * 255.0).round(), 0, 255)
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def image_inference_tensor(
        model: ImageModelDescriptor, tensor: torch.Tensor
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(tensor)


def image_inference(model: ImageModelDescriptor, image: np.ndarray, device: str, half: str) -> np.ndarray:
    return tensor_to_image(image_inference_tensor(model, image_to_tensor(image, device, half)))


def get_h_w_c(image: np.ndarray) -> tuple[int, int, int]:
    if len(image.shape) == 2:
        return image.shape[0], image.shape[1], 1
    return image.shape[0], image.shape[1], image.shape[2]


class SpandrelUpscaler(Upscaler):
    model_url = ""
    model_type = ""
    model_file = ""
    scale = 4

    def __init__(self, create_dirs=False):
        super().__init__(create_dirs)
        self.name = "Spandrel"
        self.scale = 1
        self.scalers = []

    def do_upscale(self, img: PIL.Image, selected_model: str):
        self.load_model(selected_model)
        return self.internal_upscale(img)

    def load_model(self, path: str):
        self.model = ModelLoader().load_from_file(path)
        print(f"Model size reqs: {self.model.size_requirements}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        if self.model.supports_half:
            self.model.to(torch.half)
        self.model.eval()

    def preprocess(self, image: Image) -> Image:
        square = self.model.size_requirements.square
        minimum = self.model.size_requirements.minimum
        multiple_of = self.model.size_requirements.multiple_of
        if square:
            # Pad the shorter side to make the image square
            size = max(image.width, image.height)
            new_image = Image.new("RGB", (size, size))
            new_image.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
            image = new_image
        if minimum > 1:
            size = max(image.width, image.height)
            if size < minimum:
                new_width = int(minimum * image.width / size)
                new_height = int(minimum * image.height / size)
                image = image.resize((new_width, new_height), resample=NEAREST)
        if multiple_of > 1:
            new_width = int(multiple_of * image.width // multiple_of)
            new_height = int(multiple_of * image.height // multiple_of)
            image = image.resize((new_width, new_height), resample=NEAREST)
        return image

    def postprocess(self, image: Image, original_width: int, original_height: int) -> Image:
        square = self.model.size_requirements.square
        if square:
            original_aspect_ratio = original_width / original_height
            current_aspect_ratio = image.width / image.height

            if current_aspect_ratio > original_aspect_ratio:
                # Image is wider than the original, crop width
                new_width = int(original_aspect_ratio * image.height)
                left = (image.width - new_width) // 2
                image = image.crop((left, 0, left + new_width, image.height))
            elif current_aspect_ratio < original_aspect_ratio:
                # Image is taller than the original, crop height
                new_height = int(image.width / original_aspect_ratio)
                top = (image.height - new_height) // 2
                image = image.crop((0, top, image.width, top + new_height))

        return image

    def internal_upscale(self, image: Image):
        original_width = image.width
        original_height = image.height
        needs_preprocess = self.model.size_requirements.check(image.width, image.height)
        if not needs_preprocess:
            image = self.preprocess(image)
        # Convert image to cv2 format
        image = np.array(image)
        image_h, image_w, image_c = get_h_w_c(image)

        if self.model.input_channels == 1 and image_c == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # If the image size is already greater than 2048, we'll likely OOM on GPU, so do it on CPU
        if image_h > 2048 or image_w > 2048:
            device = "cpu"

        try:
            self.model.to(device)
            half = None
            if self.model.supports_half:
                half = torch.half
            output = image_inference(self.model, image, device, half)
            # Convert output to PIL format
            output = Image.fromarray(output)
            if needs_preprocess:
                output = self.postprocess(output, original_width, original_height)
            return output
        except Exception as e:
            print(f"Failed to upscale image: {e}")
            traceback.print_exc()
            return Image.fromarray(image)

    def unload(self):
        try:
            del self.model
        except:
            pass
        self.model = None

    def load(self):
        pass
