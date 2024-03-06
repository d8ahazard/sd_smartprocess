from dataclasses import dataclass, field
from typing import List, Dict

from extensions.sd_smartprocess.file_manager import ImageData


@dataclass
class ProcessParams:
    auto_save: bool = False
    blip_initial_prompt = "a caption for this image is: "
    booru_min_score: float = 0.75
    caption: bool = False
    captioners: Dict[str, bool] = field(default_factory=lambda: [])
    clip_append_artist: bool = False
    clip_append_flavor: bool = False
    clip_append_medium: bool = False
    clip_append_movement: bool = False
    clip_append_trending: bool = False
    clip_max_flavors: int = 3
    clip_use_v2: bool = False
    crop: bool = False
    crop_mode: str = "smart"
    do_backup: bool = False
    do_rename: bool = False
    dst: str = ""
    face_model: str = "Codeformers"
    flip: bool = False
    load_mplug_8bit: bool = True
    max_clip_tokens: float = 1.0
    max_size: int = 1024
    max_tokens: int = 75
    min_clip_tokens: float = 0.0
    new_caption: str = ""
    nl_captioners: Dict[str, bool] = field(default_factory=lambda: [])
    num_beams: int = 5
    pad: bool = False
    replace_class: bool = False
    restore_faces: bool = False
    save_caption: bool = False
    save_image: bool = False
    src_files: List[ImageData] = field(default_factory=lambda: [])
    subject: str = ""
    subject_class: str = ""
    tags_to_ignore: List[str] = field(default_factory=lambda: [])
    threshold: float = 0.5
    char_threshold: float = 0.5
    txt_action: str = "ignore"
    upscale: bool = False
    upscale_max: int = 4096
    upscale_mode: str = "ratio"
    upscale_ratio: float = 2.0
    upscaler_1 = None
    upscaler_2 = None
    wd14_min_score: float = 0.75
    image_path = None

    def clip_params(self):
        return {
            "min_clip_tokens": self.min_clip_tokens,
            "max_clip_tokens": self.max_clip_tokens,
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

    def pre_only(self):
        self.caption = False
        self.upscale = False
        self.restore_faces = False

    def cap_only(self):
        self.upscale = False
        self.restore_faces = False
        self.crop = False
        self.pad = False

    def post_only(self):
        self.caption = False
        self.crop = False
        self.pad = False

    @classmethod
    def from_dict(cls, d):
        instance = cls()  # Get the singleton instance
        for k, v in d.items():
            k = k.replace("sp_", "")  # Adjust the attribute name
            if k == "class":
                k = "subject_class"
            if hasattr(instance, k):
                setattr(instance, k, v)
        return instance
