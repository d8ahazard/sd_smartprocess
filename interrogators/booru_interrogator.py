import re

import numpy as np
import torch
from PIL.Image import Image

import modules.deepbooru
from extensions.sd_smartprocess.interrogators.interrogator import Interrogator, re_special
from extensions.sd_smartprocess.process_params import ProcessParams
from modules import images, devices


class BooruInterrogator(Interrogator):
    params = {"min_score": 0.75}

    def __init__(self, params: ProcessParams) -> None:
        super().__init__(params)
        self.tags = None
        self.booru = modules.deepbooru.DeepDanbooru()
        self.booru.start()
        self.model = self.booru.model

    def unload(self):
        self.booru.stop()

    def load(self):
        pass

    def interrogate(self, image: Image, params: ProcessParams, unload: bool = False) -> str:
        self.load()
        self.params = params
        min_score = params.booru_min_score
        pic = images.resize_image(2, image, 512, 512)
        a = np.expand_dims(np.array(pic, dtype=np.float32), 0) / 255

        with torch.no_grad(), devices.autocast():
            # Move the model to the same device as the input tensor
            self.model.to(devices.device)
            # Convert input to the correct type (half-precision if needed)
            x = torch.from_numpy(a).to(devices.device).type_as(self.model.n_Conv_0.weight)
            # Forward pass through the model
            y = self.model(x)[0].detach().cpu().numpy()

        probability_dict = {}

        for tag, probability in zip(self.model.tags, y):
            if tag.startswith("rating:"):
                continue

            probability_dict[tag] = probability

        tags = [tag for tag, _ in sorted(probability_dict.items(), key=lambda x: -x[1])]

        output = {}
        for tag in tags:
            probability = probability_dict[tag]
            tag_outformat = tag
            tag_outformat = re.sub(re_special, r'\\\1', tag_outformat)
            output[tag_outformat] = probability
        out_tags = []
        for tag in sorted(output, key=output.get, reverse=True):
            if output[tag] >= min_score:
                # print(f"DBTag {tag} score is {tags[tag]}")
                out_tags.append(tag)
        output = ", ".join(out_tags)
        if unload:
            self.unload()
        return output
