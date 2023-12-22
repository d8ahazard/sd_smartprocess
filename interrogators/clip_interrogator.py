import hashlib
import math
import os
import pickle
import time
from dataclasses import dataclass
from typing import List

import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm

from extensions.sd_smartprocess.interrogators.blip_interrogator import BLIPInterrogator
from extensions.sd_smartprocess.interrogators.interrogator import Interrogator
from extensions.sd_smartprocess.process_params import ProcessParams


@dataclass
class Config:
    # models can optionally be passed in directly
    clip_model = None
    clip_preprocess = None

    # clip settings
    clip_model_name: str = 'ViT-L-14/openai'
    clip_model_path: str = None

    # interrogator settings
    cache_path: str = 'cache'
    chunk_size: int = 2048
    data_path: str = os.path.join(os.path.dirname(__file__), 'data')
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    flavor_intermediate_count: int = 2048
    quiet: bool = False  # when quiet progress bars are not shown


class CLIPInterrogator(Interrogator):
    params = {
        "min_clip_tokens": 32,
        "max_clip_tokens": 75,
        "num_beams": 1,
        "max_flavors": 32,
        "use_v2": False,
        "append_artist": False,
        "append_medium": False,
        "append_movement": False,
        "append_flavor": False,
        "append_trending": False
    }

    def __init__(self, params: ProcessParams):
        super().__init__(params)
        if params.clip_use_v2:
            model_name = "ViT-H-14/laion2b_s32b_b79k"
        else:
            model_name = "ViT-L-14/openai"
        print(f"Loading CLIP model from {model_name}")
        self.artists = None
        self.flavors = None
        self.mediums = None
        self.movements = None
        self.tokenize = None
        self.trendings = None
        self.clip_model = None
        self.clip_preprocess = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.append_artist = params.clip_append_artist
        self.append_medium = params.clip_append_medium
        self.append_movement = params.clip_append_movement
        self.append_trending = params.clip_append_trending
        self.append_flavor = params.clip_append_flavor
        self.blip_initial_prompt = params.blip_initial_prompt
        self.min_clip_tokens = params.min_clip_tokens
        self.max_clip_tokens = params.max_clip_tokens
        self.max_flavors = params.clip_max_flavors
        config = Config
        config.clip_model_name = model_name
        config.blip_min_length = self.min_clip_tokens
        config.blip_max_length = self.max_clip_tokens
        config.blip_num_beams = params.num_beams
        config.device = self.device
        self.config = config
        self.blip_interrogator = BLIPInterrogator(params.blip_initial_prompt)
        self.load_clip_model()

    def set_model_type(self, use_v2):
        if use_v2:
            model_name = "ViT-H-14/laion2b_s32b_b79k"
        else:
            model_name = "ViT-L-14/openai"
        if self.config.clip_model_name != model_name:
            if self.clip_model is not None:
                del self.clip_model
            print(f"Loading CLIP model from {model_name}")
            self.config.clip_model_name = model_name
            self.load_clip_model()

    def load_clip_model(self):
        start_time = time.time()
        config = self.config

        if config.clip_model is None:
            if not config.quiet:
                print("Loading CLIP model...")

            clip_model_name, clip_model_pretrained_name = config.clip_model_name.split('/', 2)
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                clip_model_name,
                pretrained=clip_model_pretrained_name,
                precision='fp16' if config.device == 'cuda' else 'fp32',
                device=config.device,
                jit=False,
                cache_dir=config.clip_model_path
            )
            self.clip_model.to(config.device).eval()
        else:
            self.clip_model = config.clip_model
            self.clip_preprocess = config.clip_preprocess
            clip_model_name = config.clip_model_name

        self.tokenize = open_clip.get_tokenizer(clip_model_name)

        sites = ['Artstation', 'behance', 'cg society', 'cgsociety', 'deviantart', 'dribble', 'flickr', 'instagram',
                 'pexels', 'pinterest', 'pixabay', 'pixiv', 'polycount', 'reddit', 'shutterstock', 'tumblr', 'unsplash',
                 'zbrush central']
        trending_list = [site for site in sites]
        trending_list.extend(["trending on " + site for site in sites])
        trending_list.extend(["featured on " + site for site in sites])
        trending_list.extend([site + " contest winner" for site in sites])

        raw_artists = _load_list(config.data_path, 'artists.txt')
        artists = [f"by {a}" for a in raw_artists]
        artists.extend([f"inspired by {a}" for a in raw_artists])
        if self.append_artist:
            self.artists = LabelTable(artists, "artists", self.clip_model, self.tokenize, config)
        if self.append_flavor:
            self.flavors = LabelTable(_load_list(config.data_path, 'flavors.txt'), "flavors", self.clip_model,
                                      self.tokenize, config)
        if self.append_medium:
            self.mediums = LabelTable(_load_list(config.data_path, 'mediums.txt'), "mediums", self.clip_model,
                                      self.tokenize, config)
        if self.append_movement:
            self.movements = LabelTable(_load_list(config.data_path, 'movements.txt'), "movements", self.clip_model,
                                        self.tokenize, config)
        if self.append_trending:
            self.trendings = LabelTable(trending_list, "trendings", self.clip_model, self.tokenize, config)

        end_time = time.time()
        if not config.quiet:
            print(f"Loaded CLIP model and data in {end_time - start_time:.2f} seconds.")

    def generate_caption(self, image: Image) -> str:
        self.blip_interrogator.initial_prompt = self.blip_initial_prompt
        blip_caption = self.blip_interrogator.interrogate(image, self.params)
        return blip_caption

    def image_to_features(self, image: Image) -> torch.Tensor:
        images = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

    def interrogate(self, image: Image, params: ProcessParams, unload: bool = False, short: bool = False) -> str:
        self.load()
        self.params = params
        self.append_artist = params.clip_append_artist
        self.append_medium = params.clip_append_medium
        self.append_movement = params.clip_append_movement
        self.append_trending = params.clip_append_trending
        self.append_flavor = params.clip_append_flavor
        self.blip_initial_prompt = params.blip_initial_prompt
        self.min_clip_tokens = params.min_clip_tokens
        self.max_clip_tokens = params.max_clip_tokens
        self.max_flavors = params.clip_max_flavors
        config = Config
        config.blip_min_length = self.min_clip_tokens
        config.blip_max_length = self.max_clip_tokens
        config.blip_num_beams = params.num_beams
        config.device = self.device
        self.config = config

        caption = self.generate_caption(image)
        image_features = self.image_to_features(image)
        best_prompt = caption
        best_sim = self.similarity(image_features, best_prompt)

        def check(addition: str) -> bool:
            nonlocal best_prompt, best_sim
            prompt = best_prompt + ", " + addition
            sim = self.similarity(image_features, prompt)
            if sim > best_sim:
                best_sim = sim
                best_prompt = prompt
                return True
            return False

        def check_multi_batch(opts: List[str]):
            nonlocal best_prompt, best_sim
            prompts = []
            for i in range(2 ** len(opts)):
                prompt = best_prompt
                for bit in range(len(opts)):
                    if i & (1 << bit):
                        prompt += ", " + opts[bit]
                prompts.append(prompt)

            t = LabelTable(prompts, None, self.clip_model, self.tokenize, self.config)
            best_prompt = t.rank(image_features, 1)[0]
            best_sim = self.similarity(image_features, best_prompt)

        batch = []

        if not short:
            if self.append_artist:
                batch.append(self.artists.rank(image_features, 1)[0])
            if self.append_flavor:
                best_flavors = self.flavors.rank(image_features, self.config.flavor_intermediate_count)
                extended_flavors = set(best_flavors)
                for _ in tqdm(range(self.max_flavors), desc="Flavor chain", disable=self.config.quiet):
                    best = self.rank_top(image_features, [f"{best_prompt}, {f}" for f in extended_flavors])
                    flave = best[len(best_prompt) + 2:]
                    if not check(flave):
                        break
                    if _prompt_at_max_len(best_prompt, self.tokenize):
                        break
                    extended_flavors.remove(flave)
            if self.append_medium:
                batch.append(self.mediums.rank(image_features, 1)[0])
            if self.append_trending:
                batch.append(self.trendings.rank(image_features, 1)[0])
            if self.append_movement:
                batch.append(self.movements.rank(image_features, 1)[0])

            check_multi_batch(batch)
            tags = best_prompt.split(",")
        else:
            tags = [best_prompt]
        if unload:
            self.unload()
        return tags

    def rank_top(self, image_features: torch.Tensor, text_array: List[str]) -> str:
        text_tokens = self.tokenize([text for text in text_array]).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = text_features @ image_features.T
        return text_array[similarity.argmax().item()]

    def similarity(self, image_features: torch.Tensor, text: str) -> float:
        text_tokens = self.tokenize([text]).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = text_features @ image_features.T
        return similarity[0][0].item()

    def unload(self):
        if self.clip_model is not None:
            self.clip_model.to('cpu')
        if self.blip_interrogator is not None:
            self.blip_interrogator.unload()

    def load(self):
        if self.clip_model is not None:
            self.clip_model.to(self.device)
        if self.blip_interrogator is not None:
            self.blip_interrogator.load()


class LabelTable:
    def __init__(self, labels: List[str], desc: str, clip_model, tokenize, config: Config):
        self.chunk_size = config.chunk_size
        self.config = config
        self.device = config.device
        self.embeds = []
        self.labels = labels
        self.tokenize = tokenize

        hash = hashlib.sha256(",".join(labels).encode()).hexdigest()

        cache_filepath = None
        if config.cache_path is not None and desc is not None:
            os.makedirs(config.cache_path, exist_ok=True)
            sanitized_name = config.clip_model_name.replace('/', '_').replace('@', '_')
            cache_filepath = os.path.join(config.cache_path, f"{sanitized_name}_{desc}.pkl")
            if desc is not None and os.path.exists(cache_filepath):
                with open(cache_filepath, 'rb') as f:
                    try:
                        data = pickle.load(f)
                        if data.get('hash') == hash:
                            self.labels = data['labels']
                            self.embeds = data['embeds']
                    except Exception as e:
                        print(f"Error loading cached table {desc}: {e}")

        if len(self.labels) != len(self.embeds):
            self.embeds = []
            chunks = np.array_split(self.labels, max(1, len(self.labels) / config.chunk_size))
            for chunk in tqdm(chunks, desc=f"Preprocessing {desc}" if desc else None, disable=self.config.quiet):
                text_tokens = self.tokenize(chunk).to(self.device)
                with torch.no_grad(), torch.cuda.amp.autocast():
                    text_features = clip_model.encode_text(text_tokens)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    text_features = text_features.half().cpu().numpy()
                for i in range(text_features.shape[0]):
                    self.embeds.append(text_features[i])

            if cache_filepath is not None:
                with open(cache_filepath, 'wb') as f:
                    pickle.dump({
                        "labels": self.labels,
                        "embeds": self.embeds,
                        "hash": hash,
                        "model": config.clip_model_name
                    }, f)

        if self.device == 'cpu' or self.device == torch.device('cpu'):
            self.embeds = [e.astype(np.float32) for e in self.embeds]

    def _rank(self, image_features: torch.Tensor, text_embeds: torch.Tensor, top_count: int = 1) -> str:
        top_count = min(top_count, len(text_embeds))
        text_embeds = torch.stack([torch.from_numpy(t) for t in text_embeds]).to(self.device)
        with torch.cuda.amp.autocast():
            similarity = image_features @ text_embeds.T
        _, top_labels = similarity.float().cpu().topk(top_count, dim=-1)
        return [top_labels[0][i].numpy() for i in range(top_count)]

    def rank(self, image_features: torch.Tensor, top_count: int = 1) -> List[str]:
        if len(self.labels) <= self.chunk_size:
            tops = self._rank(image_features, self.embeds, top_count=top_count)
            return [self.labels[i] for i in tops]

        num_chunks = int(math.ceil(len(self.labels) / self.chunk_size))
        keep_per_chunk = int(self.chunk_size / num_chunks)

        top_labels, top_embeds = [], []
        for chunk_idx in tqdm(range(num_chunks), disable=self.config.quiet):
            start = chunk_idx * self.chunk_size
            stop = min(start + self.chunk_size, len(self.embeds))
            tops = self._rank(image_features, self.embeds[start:stop], top_count=keep_per_chunk)
            top_labels.extend([self.labels[start + i] for i in tops])
            top_embeds.extend([self.embeds[start + i] for i in tops])

        tops = self._rank(image_features, top_embeds, top_count=top_count)
        return [top_labels[i] for i in tops]


def _load_list(data_path: str, filename: str) -> List[str]:
    with open(os.path.join(data_path, filename), 'r', encoding='utf-8', errors='replace') as f:
        items = [line.strip() for line in f.readlines()]
    return items


def _merge_tables(tables: List[LabelTable], config: Config) -> LabelTable:
    m = LabelTable([], None, None, None, config)
    for table in tables:
        m.labels.extend(table.labels)
        m.embeds.extend(table.embeds)
    return m


def _prompt_at_max_len(text: str, tokenize) -> bool:
    tokens = tokenize([text])
    return tokens[0][-1] != 0


def _truncate_to_fit(text: str, tokenize) -> str:
    parts = text.split(', ')
    new_text = parts[0]
    for part in parts[1:]:
        if _prompt_at_max_len(new_text + part, tokenize):
            break
        new_text += ', ' + part
    return new_text
