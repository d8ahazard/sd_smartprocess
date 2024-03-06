from PIL.Image import Image

from extensions.sd_smartprocess.interrogators.interrogator import Interrogator
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

from extensions.sd_smartprocess.model_download import fetch_model
from extensions.sd_smartprocess.process_params import ProcessParams


class BLIPInterrogator(Interrogator):
    _instance = None  # Class variable to hold the singleton instance

    def __new__(cls, params: ProcessParams):
        if cls._instance is None:
            cls._instance = super(BLIPInterrogator, cls).__new__(cls)
            try:
                cls._instance._init(params)
            except Exception as e:
                cls._instance = None
                raise e
        cls.initial_prompt = params.blip_initial_prompt
        return cls._instance

    def _init(self, params: ProcessParams):
        super().__init__(params)
        self.model = None
        self.processor = None

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
        try:
            self.model.to("cpu")
        except:
            pass

    def load(self):
        try:
            model_path = fetch_model('Salesforce/blip2-opt-6.7b', "blip2")
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.model = Blip2ForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, load_in_8bit=False)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
        except:
            pass
