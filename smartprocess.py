import gc
import os
import re
import sys
import traceback
from io import StringIO
from math import sqrt
from pathlib import Path
from typing import Union, List, Tuple

import numpy as np
import torch
from PIL import Image, features
from tqdm import tqdm

import modules.codeformer_model
import modules.gfpgan_model
from clipcrop import CropClip
from extensions.sd_smartprocess.file_manager import ImageData
from extensions.sd_smartprocess.interrogators.blip_interrogator import BLIPInterrogator
from extensions.sd_smartprocess.interrogators.interrogator import InterrogatorRegistry
from extensions.sd_smartprocess.model_download import disable_safe_unpickle, enable_safe_unpickle
from extensions.sd_smartprocess.process_params import ProcessParams
from modules import shared, images

blip_interrogator = None
crop_clip = None
image_interrogators = {}
global_unpickler = None
image_features = None


def printi(message):
    shared.state.textinfo = message
    print(message)


def get_backup_path(file_path, params: ProcessParams):
    backup_path = file_path
    if params.do_backup:
        file_base = os.path.splitext(file_path)[0]
        file_ext = os.path.splitext(file_path)[1]
        backup_index = 0
        backup_path = f"{file_base}_backup{backup_index}{file_ext}"
        if os.path.exists(backup_path):
            while os.path.exists(backup_path):
                backup_index += 1
                backup_path = f"{file_base}_backup{backup_index}{file_ext}"
    return file_path, backup_path


def save_pic(img, src_name, img_index, params: ProcessParams):
    dest_dir = os.path.dirname(src_name)
    src_name, backup_name = get_backup_path(src_name, params)
    if src_name != backup_name and os.path.exists(src_name):
        os.rename(src_name, backup_name)
    basename = os.path.splitext(os.path.basename(src_name))[0]
    shared.state.current_image = img
    dest = os.path.join(dest_dir, f"{basename}.png")
    img.save(dest)
    return dest


def save_img_caption(image_path: str, img_caption: str, params: ProcessParams):
    basename = os.path.splitext(image_path)[0]
    dest = f"{basename}.txt"
    src_name, backup_name = get_backup_path(dest, params)
    if src_name != backup_name and os.path.exists(src_name):
        os.rename(src_name, backup_name)
    if img_caption is not None and len(img_caption) > 0:
        with open(src_name, "w", encoding="utf8") as file:
            file.write(src_name)
    return src_name


def list_features():
    global image_features
    if image_features is None:
        # Create buffer for pilinfo() to write into rather than stdout
        buffer = StringIO()
        features.pilinfo(out=buffer)
        pil_features = []
        # Parse and analyse lines
        for line in buffer.getvalue().splitlines():
            if "Extensions:" in line:
                ext_list = line.split(": ")[1]
                extensions = ext_list.split(", ")
                for extension in extensions:
                    if extension not in pil_features:
                        pil_features.append(extension)
        image_features = pil_features
    else:
        pil_features = image_features
    return pil_features


def is_image(path: Union[Path, str], feats=None):
    if feats is None:
        feats = []
    if not len(feats):
        feats = list_features()
    if isinstance(path, str):
        path = Path(path)
    is_img = path.is_file() and path.suffix.lower() in feats
    return is_img


def cleanup():
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
    except:
        print("cleanup exception")


def vram_usage():
    if torch.cuda.is_available():
        used_vram = torch.cuda.memory_allocated(0) / 1024 ** 3  # Convert bytes to GB
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3  # Convert bytes to GB
        return used_vram, total_vram
    else:
        return 0.0, 0.0


def unload_system():
    disable_safe_unpickle()
    if shared.interrogator is not None:
        shared.interrogator.unload()
    try:
        shared.sd_model.to("cpu")
    except:
        pass
    for former in modules.shared.face_restorers:
        try:
            former.send_model_to("cpu")
        except:
            pass
    cleanup()
    used, total = vram_usage()
    print(f"System unloaded, current VRAM usage: {used}/{total} GB")


def load_system():
    enable_safe_unpickle()
    if shared.interrogator is not None:
        shared.interrogator.send_blip_to_ram()
    try:
        if modules.shared.sd_model is not None:
            modules.shared.sd_model.to(shared.device)
    except:
        pass


def get_crop_clip():
    global crop_clip
    if crop_clip is None:
        try:
            del sys.modules['models']
        except:
            pass

        crop_clip = CropClip()
    return crop_clip


def get_image_interrogators(params: ProcessParams, all_captioners):
    global image_interrogators
    all_interrogators = InterrogatorRegistry.get_all_interrogators()
    interrogators = all_captioners
    caption_agents = []
    print(f"Interrogators: {interrogators}")
    for interrogator_name in interrogators:
        if interrogator_name not in image_interrogators:
            printi(f"\rLoading {interrogator_name} interrogator...")
            interrogator = all_interrogators[f"{interrogator_name}Interrogator"](params)
            image_interrogators[interrogator_name] = interrogator
        else:
            interrogator = image_interrogators[interrogator_name]
        interrogator.unload()
        caption_agents.append(interrogator)

    return caption_agents


def clean_string(s):
    """
    Remove non-alphanumeric characters except spaces, and normalize spacing.
    Args:
        s: The string to clean.

    Returns: A cleaned string.
    """
    # Strip any HTML tags
    cleaned = re.sub(r'<[^>]+>', '', s)
    # Remove non-alphanumeric characters except spaces
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', cleaned)
    # Check for a sentence with just the same word repeated
    if len(set(cleaned.split())) == 1:
        cleaned = cleaned.split()[0]
    words = cleaned.split()
    words_out = []
    for word in words:
        if word == "y":
            word = "a"
        words_out.append(word)
    cleaned = " ".join(words_out)
    # Replace multiple spaces with a single space
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def read_caption(image):
    existing_caption_txt_filename = os.path.splitext(image)[0] + '.txt'
    if os.path.exists(existing_caption_txt_filename):
        with open(existing_caption_txt_filename, 'r', encoding="utf8") as file:
            existing_caption_txt = file.read()
    else:
        image_name = os.path.splitext(os.path.basename(image))[0]
        existing_caption_txt = clean_string(image_name)
    return existing_caption_txt


def build_caption(image, captions_list, tags_to_ignore, caption_length, subject_class, subject, replace_class, insert_subject,
                  txt_action="ignore"):
    """
    Build a caption from an array of captions, optionally ignoring tags, optionally replacing a class name with a subject name.
    Args:
        image: the image path, used for existing caption txt file
        captions_list: A list of generated captions
        tags_to_ignore: A comma-separated list of tags to ignore
        caption_length: The maximum number of tags to include in the caption
        subject_class: The class name to replace
        subject: The subject name to replace the class name with
        replace_class: Whether to replace the class name with the subject name
        txt_action: What to do with the existing caption, if any

    Returns: A string containing the caption
    """

    all_tags = set()
    for cap in captions_list:
        all_tags.update({clean_string(tag) for tag in cap.split(",") if tag.strip()})

    if isinstance(tags_to_ignore, str):
        tags_to_ignore = tags_to_ignore.split(",")
    # Filter out ignored tags
    ignore_tags = set(clean_string(tag) for tag in tags_to_ignore if tag.strip())
    all_tags.difference_update(ignore_tags)

    # Handling existing caption based on txt_action
    if txt_action == "include":
        # Read existing caption from path/txt file
        existing_caption_txt = read_caption(image)

        existing_tags = set(clean_string(tag) for tag in existing_caption_txt.split(",") if tag.strip())
    else:
        existing_tags = set()
    all_tags = all_tags.union(existing_tags)

    # Replace class with subject
    if replace_class and subject is not "" and subject_class is not "":
        phrases = ["a", "an", "the", "this", "that"]
        cleaned_tags = []
        for tag in all_tags:
            replaced = False
            lower_tag = tag.lower()
            for phrase in phrases:
                conjunction = f"{phrase} {subject_class.lower()}"
                if conjunction in lower_tag:
                    tag = lower_tag.replace(conjunction, f"{subject.lower()}")
                    replaced = True
                    break
            if not replaced:
                tag = lower_tag.replace(subject_class, subject)
            cleaned_tags.append(tag)
        all_tags = set(cleaned_tags)
    # Limiting caption length
    tags_list = list(all_tags)
    # Sort tags list by length, with the longest caption first
    tags_list.sort(key=len, reverse=True)
    # if caption_length and len(tags_list) > caption_length:
    #     tags_list = tags_list[:caption_length]

    caption_txt = ", ".join(tags_list)
    if subject != "" and insert_subject:
        if subject not in caption_txt:
            caption_txt = f"{subject}, {caption_txt}"
    return caption_txt


def calculate_job_length(files, crop, caption, captioners, flip, restore_faces, upscale):
    num_files = len(files)
    job_length = 0
    if crop:
        job_length += num_files
    if caption:
        job_length += num_files * len(captioners)
    if flip:
        job_length += num_files
    if restore_faces:
        job_length += num_files
    if upscale:
        job_length += num_files
    return job_length


def crop_smart(img: Image, interrogator: BLIPInterrogator, cc: CropClip, params: ProcessParams):
    short_caption = interrogator.interrogate(img, params)
    im_data = cc.get_center(img, prompt=short_caption)
    crop_width = im_data[1] - im_data[0]
    center_x = im_data[0] + (crop_width / 2)
    crop_height = im_data[3] - im_data[2]
    center_y = im_data[2] + (crop_height / 2)
    crop_ratio = crop_width / crop_height
    dest_ratio = 1
    tgt_width = crop_width
    tgt_height = crop_height

    if crop_ratio != dest_ratio:
        if crop_width > crop_height:
            tgt_height = crop_width / dest_ratio
            tgt_width = crop_width
        else:
            tgt_width = crop_height / dest_ratio
            tgt_height = crop_height

        # Reverse the above if dest is too big
        if tgt_width > img.width or tgt_height > img.height:
            if tgt_width > img.width:
                tgt_width = img.width
                tgt_height = tgt_width / dest_ratio
            else:
                tgt_height = img.height
                tgt_width = tgt_height / dest_ratio

    tgt_height = int(tgt_height)
    tgt_width = int(tgt_width)
    left = max(center_x - (tgt_width / 2), 0)
    right = min(center_x + (tgt_width / 2), img.width)
    top = max(center_y - (tgt_height / 2), 0)
    bottom = min(center_y + (tgt_height / 2), img.height)
    img = img.crop((left, top, right, bottom))
    return img


def crop_center(img: Image, max_size: int):
    ratio = 1
    src_ratio = img.width / img.height

    src_w = max_size if ratio < src_ratio else img.width * max_size // img.height
    src_h = max_size if ratio >= src_ratio else img.height * max_size // img.width

    resized = images.resize_image(0, img, src_w, src_h)
    res = Image.new("RGB", (max_size, max_size))
    res.paste(resized, box=(max_size // 2 - src_w // 2, max_size // 2 - src_h // 2))
    img = res
    return img


def crop_empty(img: Image):
    # Convert PIL Image to OpenCV format
    open_cv_image = np.array(img)
    open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR

    # Function to check if the border is uniform
    def is_uniform_border(border_slice):
        # Check if all pixels in the slice are the same
        return np.all(border_slice == border_slice[0, 0, :])

    # Check top, bottom, left, right borders
    h, w, _ = open_cv_image.shape
    top_border = open_cv_image[0:1, :]
    bottom_border = open_cv_image[h - 1:h, :]
    left_border = open_cv_image[:, 0:1]
    right_border = open_cv_image[:, w - 1:w]

    # Compare opposite borders to ensure they match
    top_matches_bottom = is_uniform_border(top_border) and is_uniform_border(bottom_border) and np.all(
        top_border == bottom_border)
    left_matches_right = is_uniform_border(left_border) and is_uniform_border(right_border) and np.all(
        left_border == right_border)

    # Find the padding sizes
    top_pad = 0
    bottom_pad = 0
    left_pad = 0
    right_pad = 0

    if top_matches_bottom:
        while top_pad < h and is_uniform_border(open_cv_image[top_pad:top_pad + 1, :]):
            top_pad += 1
        while bottom_pad < h and is_uniform_border(open_cv_image[h - bottom_pad - 1:h - bottom_pad, :]):
            bottom_pad += 1

    if left_matches_right:
        while left_pad < w and is_uniform_border(open_cv_image[:, left_pad:left_pad + 1]):
            left_pad += 1
        while right_pad < w and is_uniform_border(open_cv_image[:, w - right_pad - 1:w - right_pad]):
            right_pad += 1

    # Crop the image
    cropped_image = open_cv_image[top_pad:h - bottom_pad, left_pad:w - right_pad]

    # Convert back to PIL Image
    cropped_image = Image.fromarray(cropped_image[:, :, ::-1])  # Convert BGR to RGB

    return cropped_image


def crop_contain(img, params: ProcessParams):
    ratio = 1
    src_ratio = img.width / img.height

    src_w = params.max_size if ratio < src_ratio else img.width * params.max_size // img.height
    src_h = params.max_size if ratio >= src_ratio else img.height * params.max_size // img.width

    resized = images.resize_image(0, img, src_w, src_h)
    res = Image.new("RGB", (params.max_size, params.max_size))
    res.paste(resized, box=(params.max_size // 2 - src_w // 2, params.max_size // 2 - src_h // 2))
    img = res
    return img


def get_blip_interrogator(params: ProcessParams):
    global blip_interrogator

    if blip_interrogator is None:
        blip_interrogator = BLIPInterrogator(params)

    else:

        blip_interrogator.unload()

    return blip_interrogator


def process_pre(files: List[ImageData], params: ProcessParams) -> List[ImageData]:
    output = []
    interrogator = None
    cc = None
    if params.crop and params.crop_mode == "smart":
        interrogator = get_blip_interrogator(params.clip_params())
        cc = get_crop_clip()
    total_files = len(files)
    crop_length = 0
    if params.crop:
        crop_length += total_files
    if params.pad:
        crop_length += total_files
    pbar = tqdm(total=crop_length, desc="Processing images")
    for image_data in files:
        img = image_data.get_image()
        if params.crop:
            if params.crop_mode == "smart":
                img = crop_smart(img, interrogator, cc, params)
            elif params.crop_mode == "center":
                img = crop_center(img, params.max_size)
            elif params.crop_mode == "empty":
                img = crop_empty(img)
            elif params.crop_mode == "contain":
                img = crop_contain(img, params)
            shared.state.current_image = img
            pbar.update(1)
            shared.state.job_no += 1

        if params.pad:
            ratio = 1
            src_ratio = img.width / img.height

            src_w = params.max_size if ratio < src_ratio else img.width * params.max_size // img.height
            src_h = params.max_size if ratio >= src_ratio else img.height * params.max_size // img.width

            resized = images.resize_image(0, img, src_w, src_h)
            res = Image.new("RGB", (params.max_size, params.max_size))
            res.paste(resized, box=(params.max_size // 2 - src_w // 2, params.max_size // 2 - src_h // 2))
            img = res
            pbar.update(1)
            shared.state.job_no += 1

        image_data.update_image(img)

        if params.save_image:
            img_path = save_pic(img, image_data.image_path, len(output), params)
            image_data.image_path = img_path
            output.append(image_data)
        else:
            output.append(image_data)
    if interrogator is not None:
        interrogator.unload()
    if cc is not None:
        cc.unload()
    return output


def process_captions(files: List[ImageData], params: ProcessParams, all_captioners) -> List[ImageData]:
    output = []
    caption_dict = {}
    caption_length = params.max_tokens
    tags_to_ignore = params.tags_to_ignore
    subject_class = params.subject_class
    subject = params.subject
    replace_class = params.replace_class
    txt_action = params.txt_action
    save_captions = params.save_caption or params.auto_save
    insert_subject = params.insert_subject
    agents = get_image_interrogators(params, all_captioners)
    total_files = len(files)
    total_captions = total_files * len(agents)
    pbar = tqdm(total=total_captions, desc="Captioning images")
    blip_captions = {}
    for caption_agent in agents:
        print(f"Captioning with {caption_agent.__class__.__name__}...")
        caption_agent.load()
        is_llava = False
        for image_data in files:
            temp_params = params
            img = image_data.get_image()
            temp_params.image_path = image_data.image_path
            image_path = image_data.image_path
            if image_path not in caption_dict:
                caption_dict[image_path] = []
            try:
                # If the agent is LLAVA2, build the current caption
                if caption_agent.__class__.__name__ == "LLAVA2Interrogator":
                    is_llava = True
                    print("Building caption for LLAVA2")
                    temp_params.new_caption = build_caption(image_path, caption_dict[image_path], tags_to_ignore,
                                                            caption_length,
                                                            subject_class, subject, replace_class, insert_subject,
                                                            txt_action)
                caption_out = caption_agent.interrogate(img, temp_params)
                if caption_agent.__class__.__name__ == "BLIPInterrogator":
                    blip_captions[image_path] = caption_out
                print(f"Caption for {image_path}: {caption_out}")
                caption_dict[image_path].append(caption_out)
                if is_llava and params.replace_blip_caption:
                    captions = caption_dict[image_path]
                    blip_caption = blip_captions.get(image_path, None)
                    if blip_caption is not None:
                        if blip_caption in captions:
                            captions.remove(blip_caption)
                            caption_dict[image_path] = captions

                pbar.update(1)
                shared.state.job_no += 1
            except Exception as e:
                print(f"Exception captioning {image_data}: {e}")
                traceback.print_exc()
        caption_agent.unload()

    output_dict = {}
    insert_subject = params.insert_subject
    for image_path, captions in caption_dict.items():
        caption_string = build_caption(image_path, captions, tags_to_ignore, caption_length, subject_class, subject,
                                       replace_class, insert_subject, txt_action)
        output_dict[image_path] = caption_string
        if save_captions:
            save_img_caption(image_path, caption_string, params)
        # Find the image data object in files with the path matching image_path
        image_data = next((image_data for image_data in files if image_data.image_path == image_path), None)
        if image_data is not None:
            image_data.update_caption(caption_string, False)
            output.append(image_data)
    return output


def process_post(files: ImageData, params: ProcessParams, all_files: bool = False) -> List[ImageData]:
    output = []
    total_files = len(files)
    total_post = 0
    if params.restore_faces:
        total_post += total_files

    if params.upscale:
        total_post += total_files
    pbar = tqdm(total=total_post, desc="Post-processing images")
    upscalers = []

    if params.upscale:
        shared.state.textinfo = "Upscaling..."
        if params.upscaler_1 is not None and params.upscaler_1 != "None":
            upscalers.append(params.upscaler_1)
        if params.upscaler_2 is not None and params.upscaler_2 != "None":
            upscalers.append(params.upscaler_2)
    img_index = 0
    for file in files:
        img = file.get_image()
        if params.restore_faces:
            shared.state.textinfo = f"Restoring faces using {params.face_model}..."
            if params.face_model == "gfpgan":
                restored_img = modules.gfpgan_model.gfpgan_fix_faces(np.array(img, dtype=np.uint8))
                img = Image.fromarray(restored_img)
            else:
                restored_img = modules.codeformer_model.codeformer.restore(np.array(img, dtype=np.uint8),
                                                                           w=1.0)
                img = Image.fromarray(restored_img)
            pbar.update(1)
            shared.state.job_no += 1
            shared.state.current_image = img

        if params.upscale:
            shared.state.textinfo = "Upscaling..."
            used, total = vram_usage()
            print(f"Upscaling, current VRAM usage: {used}/{total} GB")
            scaler_dims = {}
            for scaler_name in upscalers:
                for scaler in shared.sd_upscalers:
                    print(f"Scaler: {scaler.name}")
                    if scaler.name == scaler_name:
                        if scaler.name != "none":
                            scaler_dims[scaler_name] = scaler.scale
                        break

            # Calculate the upscale factor
            if params.upscale_mode == "Size":
                upscale_to_max = params.max_size
                desired_upscale = max(upscale_to_max / img.width, upscale_to_max / img.height)
            else:
                desired_upscale = params.upscale_ratio

            # Adjust the upscale factor if two upscalers are used
            if len(upscalers) == 2:
                upscale_by = sqrt(desired_upscale)
            else:
                upscale_by = desired_upscale
            print(f"Upscalers: {upscalers}")
            # Apply each upscaler sequentially
            img_prompt = None
            for scaler_name in upscalers:
                upscaler = None
                for scaler in shared.sd_upscalers:
                    print(f"Scaler: {scaler.name}")
                    if scaler.name == scaler_name:
                        upscaler = scaler
                        if scaler.name == "SD4x":
                            img_prompt = file.caption
                        break
                if upscaler:
                    scaler = upscaler.scaler
                    if img_prompt:
                        scaler.prompt = img_prompt
                    img = scaler.upscale(img, upscale_by, upscaler.data_path)
                    try:
                        scaler.unload()
                    except:
                        pass
                    pbar.update(1)
                    shared.state.job_no += 1
                    shared.state.current_image = img

        if params.save_image:
            img_path = save_pic(img, file, img_index, params)
            file.image_path = img_path
        file.update_image(img, False)
        output.append(file)
        img_index += 1

    if params.do_rename and all_files:
        rand_addon = np.random.randint(1000, 9999)
        img_idx = 0
        updated_files = []
        for file in output:
            src = file.image_path
            base_name = os.path.splitext(src)[0]
            img_ext = os.path.splitext(src)[1]
            cap_file = base_name + '.txt'
            new_img_name = f"{img_idx}_{rand_addon}.{img_ext}"
            new_cap_file = f"{img_idx}_{rand_addon}.txt"
            new_img_path = os.path.join(os.path.dirname(src), new_img_name)
            new_cap_path = os.path.join(os.path.dirname(src), new_cap_file)
            os.rename(src, new_img_path)
            if os.path.exists(cap_file):
                os.rename(cap_file, new_cap_path)
            file.image_path = new_img_path
            updated_files.append(new_img_path)
            img_idx += 1

        output = []
        for file in updated_files:
            image_path = file.image_path
            cap_path = os.path.splitext(image_path)[0] + '.txt'
            new_image_path = image_path.replace(f"_{rand_addon}", "")
            new_cap_path = cap_path.replace(f"_{rand_addon}", "")
            os.rename(image_path, new_image_path)
            if os.path.exists(cap_path):
                os.rename(cap_path, new_cap_path)
            file.image_path = new_image_path
            output.append(file)
    return output


def do_process(params: ProcessParams, all_files=False) -> Tuple[List[ImageData], str]:
    print(f"Processing with params: {params}")
    output = params.src_files
    try:
        global blip_interrogator
        global image_interrogators
        # combine params.captioners and params.nl_captioners
        all_captioners = params.captioners
        for nl_captioner in params.nl_captioners:
            all_captioners.append(nl_captioner)

        job_length = calculate_job_length(params.src_files, params.crop, params.caption, all_captioners, params.flip,
                                          params.restore_faces, params.upscale)

        if job_length == 0:
            msg = "Nothing to do."
            printi(msg)
            return output, msg

        unload_system()
        do_preprocess = params.pad or params.crop or params.flip
        do_postprocess = params.restore_faces or params.upscale

        shared.state.textinfo = "Initializing smart processing..."
        shared.state.job_count = job_length
        shared.state.job_no = 0
        if do_preprocess:
            output = process_pre(output, params)

        if params.caption:
            output = process_captions(output, params, all_captioners)

        if do_postprocess:
            output = process_post(output, params, all_files)

        return output, f"Successfully processed {len(output)} images."
    except Exception as e:
        traceback.print_exc()
        msg = f"Error processing images: {e}"
        printi(msg)

    return output, msg
