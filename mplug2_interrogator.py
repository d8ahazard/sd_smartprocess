import gc
import logging
import os
from typing import Dict

import torch
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import TextStreamer

from extensions.sd_smartprocess.interrogator import Interrogator
from modules.paths_internal import models_path
from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates
from mplug_owl2.mm_utils import KeywordsStoppingCriteria, tokenizer_image_token, process_images, \
    get_model_name_from_path
from mplug_owl2.model.builder import load_pretrained_model

logger = logging.getLogger(__name__)


class MPLUG2Interrogator(Interrogator):
    model = None
    processor = None

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug("Initializing LLM model...")
        pretrained_ckpt = 'MAGAer13/mplug-owl2-llama2-7b'
        scripts_dir = os.path.join(models_path, "llm")
        os.makedirs(scripts_dir, exist_ok=True)
        model_name = "mplug-owl2-llama2-7b"
        model_path = os.path.join(scripts_dir, model_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)
            snapshot_download(pretrained_ckpt, repo_type="model", local_dir=model_path, local_dir_use_symlinks=False)

        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, None, model_name,
                                                                               load_8bit=False, load_4bit=False,
                                                                               device="cuda")

        self._to_cpu()
        logger.debug("Initialized LLM model.")

    def interrogate(self, image: Image, params: Dict = None, unload: bool = False):
        self._to_gpu()
        query = "Describe the image."

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

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(
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

        return caption

    def _to_cpu(self):
        self.model.to('cpu')
        self.image_processor.to('cpu')
        self.tokenizer.to('cpu')

    def _to_gpu(self):
        self.model.to(self.device)
        self.image_processor.to(self.device)
        self.tokenizer.to(self.device)

    def unload(self):
        self._to_cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def load(self):
        self._to_gpu()
