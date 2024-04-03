from PIL.Image import Image

from extensions.sd_smartprocess.interrogators.interrogator import Interrogator
from model_download import fetch_model
from process_params import ProcessParams
from torch import float16
from transformers import AutoModelForCausalLM, AutoTokenizer


class MoonDreamInterrogator(Interrogator):
    params = {"interrogation_prompt": "Describe this image in one detailed sentence."}  # Class-level attribute

    def __init__(self, params: ProcessParams):
        super().__init__(params)
        self.interrogation_prompt = params.interrogation_prompt
        self.model = None
        self.tokenizer = None
        print("Initializing Moondream...")

    def interrogate(self, image: Image, params: ProcessParams = None, unload: bool = False) -> str:
        self.load()
        enc_image = self.model.encode_image(image)
        caption = self.model.answer_question(enc_image, "Describe this image in one detailed sentence.", self.tokenizer)
        caption = caption.replace(",", "").replace(".", "").replace("?", "").replace("!", "").strip()
        if unload:
            self.unload()
        return caption

    def _to_gpu(self):
        self.model = self.model.to("cuda")

    def _to_cpu(self):
        self.model = self.model.to("cpu")

    def load(self):
        if self.model is None:
            model_id = "vikhyatk/moondream2"
            model_path = fetch_model(model_id, "llm")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True,
                torch_dtype=float16).to("cuda")
