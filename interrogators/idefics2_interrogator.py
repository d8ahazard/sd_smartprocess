import gc
import logging
from datetime import datetime

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig

from extensions.sd_smartprocess.interrogators.interrogator import Interrogator
from extensions.sd_smartprocess.model_download import fetch_model
from extensions.sd_smartprocess.mplug_owl2.mm_utils import get_model_name_from_path
from extensions.sd_smartprocess.process_params import ProcessParams


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


class Idefics2Interrogator(Interrogator):
    model = None
    processor = None
    params = {"max_tokens": 75, "load_in_4bit": True, "replace_blip_caption": True, "idefics2_prompt": "Describe this image in one detailed sentence, include the subject, location, style, and type of image."}

    def __init__(self, params: ProcessParams):
        super().__init__(params)
        logger.debug("Initializing LLM model...")
        model_path = fetch_model("HuggingFaceM4/idefics2-8b", "llm")
        model_name = get_model_name_from_path(model_path)
        self.load_4bit = params.load_in_4bit
        self.prompt = params.interrogation_prompt
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.context_len = None
        self.current_device = None
        logger.debug("Initialized LLM model.")

    def interrogate(self, image: Image, params: ProcessParams = None, unload: bool = False) -> str:
        self.load()
        self.prompt = NO_CAP_PROMPT
        if params is not None:
            if params.new_caption != "":
                existing_caption_txt = params.new_caption
                self.prompt = f"{EX_CAP_PROMPT}: {existing_caption_txt}"
                logger.debug(f"Existing caption query: {self.prompt}")

        # Create inputs
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.prompt},
                ]
            }
        ]

        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        generated_ids = self.model.generate(**inputs, max_new_tokens=500)
        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        caption = caption[0].strip()
        # Find "Assistant: " in the string and return everything after it
        caption = caption[caption.find("Assistant: ") + len("Assistant: "):].strip()
        if params.txt_action != "include":
            caption = caption.replace(",", "").replace(".", "").replace("?", "").replace("!", "").strip()
        if unload:
            self.unload()
        return caption

    def _to_cpu(self):
        if self.load_4bit:
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
        if self.load_4bit:
            print("Model is loaded in 4bit, can't move to GPU.")
            return
        if self.current_device != "cuda" and torch.cuda.is_available():
            print("Moving to GPU")
            time = datetime.now()
            self.model.to(self.device)
            print(f"Model to GPU: {datetime.now() - time}")
            self.current_device = "cuda"

    def unload(self):
        if self.model is not None:
            print("Unloading Idefics2 model")
            self._to_cpu()

    def load(self):
        if self.model is None:
            model_path = fetch_model("HuggingFaceM4/idefics2-8b", "llm")
            # model_name = get_model_name_from_path(model_path)
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.current_device = self.device
            print(f"Loading processor on {self.device}")
            self.processor = AutoProcessor.from_pretrained(model_path)
            if self.load_4bit:
                print(f"Loading model.")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    quantization_config=quantization_config
                )
            else:
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16
                ).to(self.device)
            print(f"Model loaded on {self.device}")
        self._to_gpu()
