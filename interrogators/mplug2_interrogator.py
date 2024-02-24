import gc
import logging
import os
from typing import Dict

import torch
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import TextStreamer

from extensions.sd_smartprocess.interrogators.interrogator import Interrogator
from extensions.sd_smartprocess.model_download import fetch_model
from extensions.sd_smartprocess.process_params import ProcessParams
from modules.paths_internal import models_path
from extensions.sd_smartprocess.mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from extensions.sd_smartprocess.mplug_owl2.conversation import conv_templates
from extensions.sd_smartprocess.mplug_owl2.mm_utils import KeywordsStoppingCriteria, tokenizer_image_token, \
    process_images, \
    get_model_name_from_path
from extensions.sd_smartprocess.mplug_owl2.model.builder import load_pretrained_model

# This is basically broken until we can update transformers in AUTO past the current version supported

logger = logging.getLogger(__name__)


class MPLUG2Interrogator(Interrogator):
    model = None
    processor = None
    params = {"max_tokens": 75}

    def __init__(self, params: ProcessParams):
        super().__init__(params)
        logger.debug("Initializing LLM model...")
        model_path = fetch_model('MAGAer13/mplug-owl2-llama2-7b', "llm")
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, None,
                                                                                                   model_name,
                                                                                                   load_8bit=False,
                                                                                                   load_4bit=False,
                                                                                                   device="cuda")

        self._to_cpu()
        logger.debug("Initialized LLM model.")

    def interrogate(self, image: Image, params=None, unload: bool = False) -> str:
        self.load()
        if params is None:
            params = {}
        query = "Describe the image with a caption that can be used to generate a similar image."

        conv = conv_templates["mplug_owl2"].copy()
        roles = conv.roles

        max_edge = max(image.size)  # We recommand you to resize to squared image for BEST performance.
        image = image.resize((max_edge, max_edge))

        image_tensor = process_images([image], self.image_processor)
        image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        inp = DEFAULT_IMAGE_TOKEN + query
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
            0).to(
            self.model.device)
        stop_str = conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        temperature = 0.7
        max_new_tokens = 512

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        caption = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        caption = caption.replace(",", "").replace(".", "").replace("?", "").replace("!", "").strip()
        return caption

    def _to_cpu(self):
        self.model.to('cpu')
        #self.image_processor.to('cpu')
        #self.tokenizer.to('cpu')

    def _to_gpu(self):
        self.model.to(self.device)
        #self.image_processor.to(self.device)
        #self.tokenizer.to(self.device)

    def unload(self):
        self._to_cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def load(self):
        self._to_gpu()
