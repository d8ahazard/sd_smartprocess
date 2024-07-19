from PIL.Image import Image

from extensions.sd_smartprocess.interrogators.interrogator import Interrogator
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

from extensions.sd_smartprocess.model_download import fetch_model
from extensions.sd_smartprocess.process_params import ProcessParams


class BLIPInterrogator(Interrogator):
    _instance = None  # Class variable to hold the singleton instance
    params = {"max_tokens": 75, "load_in_8bit": False}

    def __init__(self, params: ProcessParams):
        super().__init__(params)
        self.processor = None
        self.model = None
        self.current_device = None
        self.load_8bit = params.load_in_8bit

    def __new__(cls, params: ProcessParams):
        if cls._instance is None:
            cls._instance = super(BLIPInterrogator, cls).__new__(cls)
            try:
                cls._instance._init(params)
            except Exception as e:
                cls._instance = None
                raise e
        cls.initial_prompt = params.blip_initial_prompt
        cls._load_8bit = params.load_in_8bit
        return cls._instance

    def _init(self, params: ProcessParams):
        super().__init__(params)
        self.model = None
        self.processor = None
        self.load_8bit = params.load_in_8bit

    def interrogate(self, image: Image, params: ProcessParams, unload: bool = False) -> str:
        self.load()
        self.params = params
        max_tokens = params.max_tokens
        inputs = self.processor(image, return_tensors="pt").to(self.device, torch.float16)
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        if unload:
            self.unload()
        return generated_text

    def unload(self):
        if self.current_device != "cpu":
            try:
                self.model.to("cpu")
                self.current_device = "cpu"
            except:
                pass

    def load(self):
        try:
            if self.model is None:
                model_path = fetch_model('Salesforce/blip2-opt-6.7b', "blip2")
                self.processor = AutoProcessor.from_pretrained(model_path)
                self.model = Blip2ForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, load_in_8bit=self.load_8bit)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.device != self.current_device:
                self.model.to(self.device)
                self.current_device = self.device
        except:
            pass
