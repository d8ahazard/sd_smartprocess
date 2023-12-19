from extensions.sd_smartprocess.interrogator import Interrogator
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

from extensions.sd_smartprocess.model_download import fetch_model


class BlipInterrogator(Interrogator):
    _instance = None  # Class variable to hold the singleton instance

    def __new__(cls, initial_prompt):
        if cls._instance is None:
            cls._instance = super(BlipInterrogator, cls).__new__(cls)
            try:
                cls._instance._init(initial_prompt)
            except Exception as e:
                cls._instance = None
                raise e
        cls._instance.initial_prompt = initial_prompt
        return cls._instance

    def _init(self, initial_prompt):
        model_path = fetch_model('Salesforce/blip2-opt-2.7b', "blip2")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.initial_prompt = initial_prompt
        self.model.to(self.device)

    def interrogate(self, image):
        prompt = self.initial_prompt
        inputs = self.processor(image, text=prompt, return_tensors="pt").to(self.device, torch.float16)
        generated_ids = self.model.generate(**inputs, max_new_tokens=20)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text

    def unload(self):
        self.model.to("cpu")

    def load(self):
        self.model.to(self.device)
