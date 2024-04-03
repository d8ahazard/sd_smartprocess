from PIL.Image import Image

from interrogators.interrogator import Interrogator
from model_download import fetch_model
from process_params import ProcessParams
from torch import float16
from transformers import AutoModelForCausalLM, AutoTokenizer


class MoondreamInterrogator(Interrogator):
    model = None
    tokenizer = None
    params = {"interrogation_prompt": "Describe this image in one detailed sentence."}

    def __init__(self, params: ProcessParams):
        super().__init__(params)
        self.params = params
        self.model = None
        self.tokenizer = None

    def interrogate(self, image: Image, params: ProcessParams = None, unload: bool = False) -> str:
        self.load()
        enc_image = self.model.encode_image(image)
        caption = self.model.answer_question(enc_image, "Describe this image in one detailed sentence.", self.tokenizer)
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
            revision = "2024-04-02"
            model_path = fetch_model(model_id, "llm")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True,
                torch_dtype=float16, attn_implementation="flash_attention_2"
            ).to("cuda")
