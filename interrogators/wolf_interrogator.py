import logging

import PIL
import cv2
import huggingface_hub
import numpy as np
import pandas as pd
from PIL.Image import Image
from onnxruntime import InferenceSession

from extensions.sd_smartprocess.interrogators.interrogator import Interrogator
from extensions.sd_smartprocess.model_download import fetch_model
from extensions.sd_smartprocess.process_params import ProcessParams

logger = logging.getLogger(__name__)

CONV2_MODEL_REPO = "SmilingWolf/wd-v1-4-convnextv2-tagger-v2"
MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"
WOLF_PARAMS = {
    "group": "WOLF",
    "threshold": 0.75,
    "char_threshold": 0.75
}


class MoatInterrogator(Interrogator):
    params = WOLF_PARAMS

    def __init__(self, params: ProcessParams) -> None:
        super().__init__(params)
        self._setup()

    def _setup(self):
        model_path = "SmilingWolf/wd-v1-4-moat-tagger-v2"
        self.model = load_model(model_path, self.device)

    def interrogate(self, image: Image, params: ProcessParams, unload: bool = False) -> str:
        threshold = params.threshold
        char_threshold = params.char_threshold
        a, c, rating, character_res, general_res = predict(image, self.model, threshold, char_threshold)
        return a


class SwinInterrogator(Interrogator):
    params = WOLF_PARAMS

    def __init__(self, params: ProcessParams) -> None:
        super().__init__(params)
        self._setup()

    def _setup(self):
        model_path = "SmilingWolf/wd-v1-4-swinv2-tagger-v2"
        self.model = load_model(model_path, self.device)

    def interrogate(self, image: Image, params: ProcessParams, unload: bool = False) -> str:
        threshold = params.threshold
        char_threshold = params.char_threshold
        a, c, rating, character_res, general_res = predict(image, self.model, threshold, char_threshold)
        return a


class ConvInterrogator(Interrogator):
    params = WOLF_PARAMS

    def __init__(self, params: ProcessParams) -> None:
        super().__init__(params)
        self._setup()

    def _setup(self):
        model_path = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
        self.model = load_model(model_path, self.device)

    def interrogate(self, image: Image, params: ProcessParams, unload: bool = False) -> str:
        threshold = params.threshold
        char_threshold = params.char_threshold
        a, c, rating, character_res, general_res = predict(image, self.model, threshold, char_threshold)
        return a


class Conv2Interrogator(Interrogator):
    params = WOLF_PARAMS

    def __init__(self, params: ProcessParams) -> None:
        super().__init__(params)
        self._setup()

    def _setup(self):
        model_path = "SmilingWolf/wd-v1-4-convnextv2-tagger-v2"
        self.model = load_model(model_path, self.device)

    def interrogate(self, image: Image, params: ProcessParams, unload: bool = False) -> str:
        threshold = params.threshold
        char_threshold = params.char_threshold
        a, c, rating, character_res, general_res = predict(image, self.model, threshold, char_threshold)
        return a


class VitInterrogator(Interrogator):
    params = WOLF_PARAMS

    def __init__(self, params: ProcessParams) -> None:
        super().__init__(params)
        self._setup()

    def _setup(self):
        model_path = "SmilingWolf/wd-v1-4-vit-tagger-v2"
        self.model = load_model(model_path, self.device)

    def interrogate(self, image: Image, params: ProcessParams, unload: bool = False) -> str:
        threshold = params.threshold
        char_threshold = params.char_threshold
        a, c, rating, character_res, general_res = predict(image, self.model, threshold, char_threshold)
        return a


def predict(
        image: Image,
        model,
        general_threshold: float = 0.5,
        character_threshold: float = 0.5
):
    tag_names, rating_indexes, general_indexes, character_indexes = load_labels()

    _, height, width, _ = model.get_inputs()[0].shape

    # Alpha to white
    image = image.convert("RGBA")
    new_image = PIL.Image.new("RGBA", image.size, "WHITE")
    new_image.paste(image, mask=image)
    image = new_image.convert("RGB")
    image = np.asarray(image)

    # PIL RGB to OpenCV BGR
    image = image[:, :, ::-1]

    image = make_square(image, height)
    image = smart_resize(image, height)
    image = image.astype(np.float32)
    image = np.expand_dims(image, 0)

    input_name = model.get_inputs()[0].name
    label_name = model.get_outputs()[0].name
    probs = model.run([label_name], {input_name: image})[0]

    labels = list(zip(tag_names, probs[0].astype(float)))

    # First 4 labels are actually ratings: pick one with argmax
    ratings_names = [labels[i] for i in rating_indexes]
    rating = dict(ratings_names)

    # Then we have general tags: pick any where prediction confidence > threshold
    general_names = [labels[i] for i in general_indexes]
    general_res = [x for x in general_names if x[1] > general_threshold]
    general_res = dict(general_res)

    # Everything else is characters: pick any where prediction confidence > threshold
    character_names = [labels[i] for i in character_indexes]
    character_res = [x for x in character_names if x[1] > character_threshold]
    character_res = dict(character_res)

    b = dict(sorted(general_res.items(), key=lambda item: item[1], reverse=True))
    a = (
        ", ".join(list(b.keys()))
        .replace("_", " ")
        .replace("(", "\(")
        .replace(")", "\)")
    )
    c = ", ".join(list(b.keys()))

    return a, c, rating, character_res, general_res


def load_model(model_path, device):
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    if device == "cpu":
        providers.pop(0)
    path = fetch_model(model_path, "wolf", True)
    model = InferenceSession(path, providers=providers)
    return model


def smart_imread(img, flag=cv2.IMREAD_UNCHANGED):
    if img.endswith(".gif"):
        img = PIL.Image.open(img)
        img = img.convert("RGB")
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(img, flag)
    return img


def smart_24bit(img):
    if img.dtype is np.dtype(np.uint16):
        img = (img / 257).astype(np.uint8)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        trans_mask = img[:, :, 3] == 0
        img[trans_mask] = [255, 255, 255, 255]
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def make_square(img, target_size):
    old_size = img.shape[:2]
    desired_size = max(old_size)
    desired_size = max(desired_size, target_size)

    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [255, 255, 255]
    new_im = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return new_im


def smart_resize(img, size):
    # Assumes the image has already gone through make_square
    if img.shape[0] > size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    elif img.shape[0] < size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    return img


def load_labels() -> list[str]:
    path = huggingface_hub.hf_hub_download(CONV2_MODEL_REPO, LABEL_FILENAME)
    df = pd.read_csv(path)
    tag_names = df["name"].tolist()
    rating_indexes = list(np.where(df["category"] == 9)[0])
    general_indexes = list(np.where(df["category"] == 0)[0])
    character_indexes = list(np.where(df["category"] == 4)[0])
    return tag_names, rating_indexes, general_indexes, character_indexes
