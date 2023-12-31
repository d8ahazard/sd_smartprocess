import os

import torch
from huggingface_hub import snapshot_download

from modules.paths_internal import models_path
from modules.safe import unsafe_torch_load, load


def fetch_model(model_repo, model_type, single_file=False):
    model_dir = os.path.join(models_path, model_type)
    dest_dir = os.path.join(model_dir, model_repo.split("/")[1])
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
        dest_dir = snapshot_download(model_repo, repo_type="model", local_dir=dest_dir, local_dir_use_symlinks=False)
    if single_file:
        dest_dir = os.path.join(dest_dir, "model.onnx")
    return dest_dir


disable_safe_unpickle_count = 0


def disable_safe_unpickle():
    global disable_safe_unpickle_count
    try:
        from modules import shared as auto_shared
        if not auto_shared.cmd_opts.disable_safe_unpickle:
            auto_shared.cmd_opts.disable_safe_unpickle = True
            torch.load = unsafe_torch_load
        disable_safe_unpickle_count += 1
    except:
        pass


def enable_safe_unpickle():
    global disable_safe_unpickle_count
    try:
        from modules import shared as auto_shared
        if disable_safe_unpickle_count > 0:
            disable_safe_unpickle_count -= 1
            if disable_safe_unpickle_count == 0 and auto_shared.cmd_opts.disable_safe_unpickle:
                auto_shared.cmd_opts.disable_safe_unpickle = False
                torch.load = load
    except:
        pass
