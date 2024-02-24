import gc
import logging

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
from extensions.sd_smartprocess.smartprocess import read_caption

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

    def interrogate(self, image: Image, params: ProcessParams = None, unload: bool = False) -> str:
        self.load()

        query = "Analyze the provided image carefully and generate a comprehensive caption in natural language that succinctly summarizes the key elements and the overall context of the image. Following the narrative caption, provide a list of descriptive tags that detail significant components of the image. Focus on identifying and describing any people, animals, objects, and scenery. Include specifics such as hair colors, dog breeds, styles of clothing or footwear, and any other noteworthy features that stand out in the image. Ensure the language is clear, engaging, and directly relevant to what is visibly present in the image."
        if params is not None:
            if params.new_caption != "":
                existing_caption_txt = params.new_caption
                query = f"Review the provided image along with its existing caption. First, refine the existing natural-language caption to better capture the essence and key details of the image, ensuring the language is vivid, precise, and comprehensive. Then, examine the current list of descriptive tags accompanying the caption. Update these tags to accurately reflect the contents of the image, adding new tags for previously unmentioned but relevant details and removing any tags that are no longer applicable. Pay special attention to details such as the appearance of people, animal breeds, specific styles or designs of clothing and accessories, and the overall setting. The goal is to enhance the clarity and relevance of both the caption and the tags, providing a coherent and detailed description of the image: {existing_caption_txt}"
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
        return caption

    def _to_cpu(self):
        self.model.to('cpu')
        # self.image_processor.to('cpu')
        # self.tokenizer.to('cpu')

    def _to_gpu(self):
        self.model.to(self.device)
        # self.image_processor.to(self.device)
        # self.tokenizer.to(self.device)

    def unload(self):
        self._to_cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def load(self):
        self._to_gpu()
