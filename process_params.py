from dataclasses import dataclass, field
from typing import List

from extensions.sd_smartprocess.interrogator import Interrogator


@dataclass
class ProcessParams:
    rename: bool = False
    src_files: List[str] = field(default_factory=lambda: [])
    dst: str = ""
    pad: bool = False
    crop: bool = False
    max_size: int = 1024
    txt_action: str = "prepend"
    flip: bool = False
    blip_initial_prompt = "a caption for this image is: "
    caption: bool = False
    captioners: List[str] = field(default_factory=lambda: [])
    caption_length: int = 77
    num_beams: int = 5
    min_clip: float = 0.0
    max_clip: float = 1.0
    clip_use_v2: bool = False
    clip_append_flavor: bool = False
    clip_max_flavors: int = 3
    clip_append_medium: bool = False
    clip_append_movement: bool = False
    clip_append_artist: bool = False
    clip_append_trending: bool = False
    wd14_min_score: float = 0.75
    booru_min_score: float = 0.75
    tags_to_ignore: List[str] = field(default_factory=lambda: [])
    subject_class: str = ""
    subject: str = ""
    replace_class: bool = False
    restore_faces: bool = False
    face_model: str = "Codeformers"
    upscale: bool = False
    upscale_ratio: float = 2.0
    scaler = None
    save_image: bool = False
    save_caption: bool = False

    def clip_params(self):
        return {
            "min_clip": self.min_clip,
            "max_clip": self.max_clip,
            "use_v2": self.clip_use_v2,
            "append_flavor": self.clip_append_flavor,
            "max_flavors": self.clip_max_flavors,
            "append_medium": self.clip_append_medium,
            "append_movement": self.clip_append_movement,
            "append_artist": self.clip_append_artist,
            "append_trending": self.clip_append_trending,
            "num_beams": self.num_beams,
            "clip_max_flavors": self.clip_max_flavors,
            "blip_initial_prompt": self.blip_initial_prompt
        }
