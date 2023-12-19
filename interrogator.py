# Borrowed from https://github.com/toriato/stable-diffusion-webui-wd14-tagger/blob/master/tagger/interrogator.py

import os
import re
from collections import namedtuple
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from huggingface_hub import hf_hub_download

import modules.deepbooru
import modules.shared as shared
from extensions.sd_smartprocess import dbimutils
from modules import devices
from modules import images
from modules.deepbooru import re_special as tag_escape_pattern
from modules.paths_internal import models_path

blip_image_eval_size = 384
clip_model_name = 'ViT-L/14'

Category = namedtuple("Category", ["name", "topn", "items"])

re_topn = re.compile(r"\.top(\d+)\.")

use_cpu = shared.cmd_opts.use_cpu == 'all' or shared.cmd_opts.use_cpu == 'interrogate'
onyx_providers = []
if use_cpu:
    tf_device_name = '/cpu:0'
    onyx_providers = ['CPUExecutionProvider']
else:
    tf_device_name = '/gpu:0'
    onyx_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    if shared.cmd_opts.device_id is not None:
        try:
            tf_device_name = f'/gpu:{int(shared.cmd_opts.device_id)}'
        except ValueError:
            print('--device-id is not a integer')


class Interrogator:
    @staticmethod
    def postprocess_tags(
            tags: Dict[str, float],
            threshold=0.35,
            additional_tags=None,
            exclude_tags=None,
            sort_by_alphabetical_order=False,
            add_confident_as_weight=False,
            replace_underscore=False,
            replace_underscore_excludes=None,
            escape_tag=False
    ) -> Dict[str, float]:

        if replace_underscore_excludes is None:
            replace_underscore_excludes = []
        if exclude_tags is None:
            exclude_tags = []
        if additional_tags is None:
            additional_tags = []
        tags = {
            **{t: 1.0 for t in additional_tags},
            **tags
        }

        # those lines are totally not "pythonic" but looks better to me
        tags = {
            t: c

            # sort by tag name or confident
            for t, c in sorted(
                tags.items(),
                key=lambda i: i[0 if sort_by_alphabetical_order else 1],
                reverse=not sort_by_alphabetical_order
            )

            # filter tags
            if (
                    c >= threshold
                    and t not in exclude_tags
            )
        }

        for tag in list(tags):
            new_tag = tag

            if replace_underscore and tag not in replace_underscore_excludes:
                new_tag = new_tag.replace('_', ' ')

            if escape_tag:
                new_tag = tag_escape_pattern.sub(r'\\\1', new_tag)

            if add_confident_as_weight:
                new_tag = f'({new_tag}:{tags[tag]})'

            if new_tag != tag:
                tags[new_tag] = tags.pop(tag)

        return tags

    def interrogate(
            self,
            image: Image
    ) -> Tuple[
        Dict[str, float],  # rating confidence
        Dict[str, float]  # tag confidence
    ]:
        pass

    def unload(self):
        pass


re_special = re.compile(r'([\\()])')


class BooruInterrogator(Interrogator):
    def __init__(self, min_score) -> None:
        self.tags = None
        self.booru = modules.deepbooru.DeepDanbooru()
        self.min_score = min_score
        self.booru.start()
        self.model = self.booru.model

    def unload(self):
        self.booru.stop()

    def load(self):
        pass

    def interrogate(self, pil_image) -> str:
        pic = images.resize_image(2, pil_image, 512, 512)
        a = np.expand_dims(np.array(pic, dtype=np.float32), 0) / 255

        with torch.no_grad(), devices.autocast():
            x = torch.from_numpy(a).to(devices.device)
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
            if output[tag] >= self.min_score:
                # print(f"DBTag {tag} score is {tags[tag]}")
                out_tags.append(tag)
        output = ", ".join(out_tags)
        return output


class WaifuDiffusionInterrogator(Interrogator):
    def __init__(
            self,
            repo='SmilingWolf/wd-v1-4-vit-tagger',
            model_path='model.onnx',
            tags_path='selected_tags.csv',
            min_score=0.35
    ) -> None:
        self.tags = None
        self.model = None
        self.repo = repo
        self.model_path = model_path
        self.tags_path = tags_path
        self.min_score = min_score
        self.load()

    def download(self) -> Tuple[os.PathLike, os.PathLike]:
        print(f'Loading Waifu Diffusion tagger model file from {self.repo}')
        model_dir = os.path.join(models_path, "wd14")
        model_path = Path(hf_hub_download(self.repo, filename=self.model_path, local_dir=model_dir, local_dir_use_symlinks=False))
        tags_path = Path(hf_hub_download(self.repo, filename=self.tags_path, local_dir=model_dir, local_dir_use_symlinks=False))
        return model_path, tags_path

    def load(self) -> None:
        model_path, tags_path = self.download()
        from launch import is_installed, run_pip
        if not is_installed('onnxruntime'):
            package_name = 'onnxruntime-gpu'

            if use_cpu or not torch.cuda.is_available():
                package_name = 'onnxruntime'

            package = os.environ.get(
                'ONNXRUNTIME_PACKAGE',
                package_name
            )

            run_pip(f'install {package}', package_name)

        from onnxruntime import InferenceSession

        self.model = InferenceSession(str(model_path), providers=onyx_providers)

        print(f'Loaded Waifu Diffusion tagger model from {model_path}')
        self.tags = pd.read_csv(tags_path)

    def unload(self):
        pass

    def interrogate(
            self,
            image: Image
    ) -> str:
        # code for converting the image and running the model is taken from the link below
        # thanks, SmilingWolf!
        # https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags/blob/main/app.py

        # convert an image to fit the model
        _, height, _, _ = self.model.get_inputs()[0].shape

        # alpha to white
        image = image.convert('RGBA')
        new_image = Image.new('RGBA', image.size, 'WHITE')
        new_image.paste(image, mask=image)
        image = new_image.convert('RGB')
        image = np.asarray(image)

        # PIL RGB to OpenCV BGR
        image = image[:, :, ::-1]

        image = dbimutils.make_square(image, height)
        image = dbimutils.smart_resize(image, height)
        image = image.astype(np.float32)
        image = np.expand_dims(image, 0)

        # evaluate model
        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        confidence = self.model.run([label_name], {input_name: image})[0]

        tags = self.tags[:][['name']]
        tags['confidence'] = confidence[0]

        # first 4 items are for rating (general, sensitive, questionable, explicit)
        ratings = dict(tags[:4].values)

        # rest are regular tags
        tags = dict(tags[4:].values)
        out_tags = []
        for tag in sorted(tags, key=tags.get, reverse=True):
            if tags[tag] >= self.min_score:
                # print(f"WDTag {tag} score is {tags[tag]}")
                out_tags.append(tag)

        return ", ".join(out_tags)
