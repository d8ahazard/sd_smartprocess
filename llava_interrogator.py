import gc
import logging
import os
from typing import Dict

import torch
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from extensions.sd_smartprocess.interrogator import Interrogator
from extensions.sd_smartprocess.model_download import fetch_model
from modules.paths_internal import models_path
from mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor, MplugOwlForConditionalGeneration

logger = logging.getLogger(__name__)


class LLAVAInterrogator(Interrogator):
    model = None
    processor = None

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Initializing LLM model...")
        model_path = fetch_model('MAGAer13/mplug-owl-llama-7b', "llm")
        model_config = os.path.join(model_path, "config.json")
        print(f"Loading model from {model_path}...")
        self.model = MplugOwlForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            config=model_config,
        )
        print("Loading tokenizer...")
        self.image_processor = MplugOwlImageProcessor.from_pretrained(model_path)
        print("Loading processor...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("Initializing processor...")
        self.processor = MplugOwlProcessor(self.image_processor, self.tokenizer)
        logger.debug("Initialized LLM model.")

    def interrogate(self, image: Image, params: Dict = None, unload: bool = False):
        self._to_gpu()
        raw_image = image.convert('RGB')
        if params is None:
            params = {}
        character = params.get("character", "nsfw")
        char_file = os.path.join(os.path.dirname(__file__), "mplug_owl", "characters", f"{character}.txt")
        with open(char_file, "r") as f:
            character = f.read().strip()

        generate_kwargs = {
            'do_sample': True,
            'top_k': 5,
            'max_length': 77
        }

        images = [raw_image]
        prompts = [character]

        logger.debug("Processing inputs...")
        inputs = self.processor(text=prompts, images=images, return_tensors='pt')
        inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        logger.debug("Generating response...")
        with torch.no_grad():
            res = self.model.generate(**inputs, **generate_kwargs)
        caption = self.tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
        if "," in caption:
            parts = caption.split(",")
            caption = " ".join([part.strip() for part in parts if part.strip() != ""])
        logger.debug(f"Caption: {caption}")
        if unload:
            self._to_cpu()

        return caption

    def _to_cpu(self):
        self.model.to('cpu')

    def _to_gpu(self):
        self.model.to(self.device)

    def unload(self):
        self._to_cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def load(self):
        self._to_gpu()
