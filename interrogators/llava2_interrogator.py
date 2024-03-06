import gc
import logging
from datetime import datetime

import torch
from PIL import Image
from transformers import TextStreamer

from extensions.sd_smartprocess.interrogators.interrogator import Interrogator
from extensions.sd_smartprocess.model_download import fetch_model
from extensions.sd_smartprocess.mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from extensions.sd_smartprocess.mplug_owl2.conversation import conv_templates
from extensions.sd_smartprocess.mplug_owl2.mm_utils import KeywordsStoppingCriteria, tokenizer_image_token, \
    process_images, \
    get_model_name_from_path
from extensions.sd_smartprocess.mplug_owl2.model.builder import load_pretrained_model
from extensions.sd_smartprocess.process_params import ProcessParams

# This is basically broken until we can update transformers in AUTO past the current version supported

logger = logging.getLogger(__name__)

NO_CAP_PROMPT = """
Generate a concise caption describing the key elements and context of the image in one sentence, 
focusing on the medium, subject, style, clothing, pose, action, and location. Ensure the sentence is accurate and devoid
 of assumptions, prose, etc. Keep it direct and relevant to the image.

Follow the caption with a list of specific tags (keywords) detailing smaller key elements like clothing, poses,
actions, and other notable features. 
"""

EX_CAP_PROMPT = """
Here is a caption consisting of a sentence and a list of tags (keywords) that describe the image.

Refine the provided caption to more accurately and vividly capture the essence and key details visible in the image,
focusing on encapsulating the medium, subject, style, clothing, pose, action, and location in one sentence. 
 
Update the accompanying tags to reflect only the elements present in the image, ensuring precision and relevance.
Avoid adding new information not supported by the existing caption or the image.
"""


class LLAVA2Interrogator(Interrogator):
    model = None
    processor = None
    params = {"max_tokens": 75, "load_in_8bit": False, "replace_blip_caption": True}

    def __init__(self, params: ProcessParams):
        super().__init__(params)
        logger.debug("Initializing LLM model...")
        model_path = fetch_model('MAGAer13/mplug-owl2-llama2-7b', "llm")
        model_name = get_model_name_from_path(model_path)
        self.load_8bit = params.load_mplug_8bit
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.context_len = None
        self.current_device = None
        logger.debug("Initialized LLM model.")

    def interrogate(self, image: Image, params: ProcessParams = None, unload: bool = False) -> str:
        self.load()

        query = NO_CAP_PROMPT
        if params is not None:
            if params.new_caption != "":
                existing_caption_txt = params.new_caption
                query = f"{EX_CAP_PROMPT}: {existing_caption_txt}"
                logger.debug(f"Existing caption query: {query}")

        conv = conv_templates["mplug_owl2"].copy()

        max_edge = max(image.size)
        image = image.resize((max_edge, max_edge))
        image_tensor = process_images([image], self.image_processor)
        image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        inp = query + " " + DEFAULT_IMAGE_TOKEN
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
        if params.txt_action != "include":
            caption = caption.replace(",", "").replace(".", "").replace("?", "").replace("!", "").strip()
        if unload:
            self.unload()
        return caption

    def _to_cpu(self):
        if self.load_8bit:
            print("Model is loaded in 8bit, can't move to CPU.")
            return
        from extensions.sd_smartprocess.smartprocess import vram_usage
        used, total = vram_usage()
        print(f"VRAM: {used}/{total}")
        free = total - used
        # If we have over 16GB of VRAM, we can use the GPU
        if free > 16:
            print("VRAM is over 16GB, moving to GPU")
            self._to_gpu()
            return
        print("Moving to CPU")
        if self.current_device != "cpu":
            self.model.to('cpu')
            self.current_device = "cpu"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _to_gpu(self):
        if self.model is None:
            self.load()
            return
        if self.current_device != "cuda" and torch.cuda.is_available():
            print("Moving to GPU")
            time = datetime.now()
            self.model.to(self.device)
            print(f"Model to GPU: {datetime.now() - time}")
            self.current_device = "cuda"
        # self.image_processor.to(self.device)
        # self.tokenizer.to(self.device)

    def unload(self):
        if self.model is not None:
            print("Unloading LLAVA2 model")
            self._to_cpu()

    def load(self):
        if self.model is None:
            model_path = fetch_model('MAGAer13/mplug-owl2-llama2-7b', "llm")
            model_name = get_model_name_from_path(model_path)
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.current_device = self.device
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, None,
                                                                                                       model_name,
                                                                                                       load_8bit=self.load_8bit,
                                                                                                       load_4bit=False,
                                                                                                       device="cuda")

        self._to_gpu()
