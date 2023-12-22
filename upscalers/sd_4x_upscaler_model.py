import os

import PIL.Image
import torch
from diffusers import StableDiffusionUpscalePipeline

from modules import modelloader, shared
from modules.upscaler import Upscaler, UpscalerData


class Sd4xUpscaler(Upscaler):
    def __init__(self, create_dirs=False):
        super().__init__()
        self.name = "SD4x"
        self.model_path = os.path.join(shared.models_path, self.name)
        self.prompt = None
        self.scalers = [self.scaler_data()]

    def load_model(self, path: str):
        model_url = "https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler/resolve/main/x4-upscaler-ema.safetensors?download=true"
        model_path = modelloader.load_models(model_url=model_url, model_path=self.model_path,
                                             download_name="x4-upscaler-ema.safetensors")
        pipeline = StableDiffusionUpscalePipeline.from_single_file(model_path, torch_dtype=torch.float16)
        pipeline = pipeline.to("cuda")
        return pipeline

    def do_upscale(self, img: PIL.Image, selected_model: str):
        pipeline = self.load_model(selected_model)
        prompt = self.prompt
        upscaled_image = pipeline(prompt=prompt, image=img).images[0]
        return upscaled_image

    @classmethod
    def scaler_data(cls):
        upscaler_data = UpscalerData("SD4x", None, cls)
        return upscaler_data
