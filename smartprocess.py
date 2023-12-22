import os
import re
import sys
import traceback
from io import StringIO
from math import sqrt
from pathlib import Path
from typing import Union, Dict, List

import numpy as np
from PIL import Image, features
from tqdm import tqdm

import modules.codeformer_model
import modules.gfpgan_model
from clipcrop import CropClip
from extensions.sd_smartprocess.interrogators.clip_interrogator import CLIPInterrogator
from extensions.sd_smartprocess.interrogators.interrogator import InterrogatorRegistry
from extensions.sd_smartprocess.model_download import disable_safe_unpickle, enable_safe_unpickle
from extensions.sd_smartprocess.process_params import ProcessParams
from modules import shared, images

clip_interrogator = None
crop_clip = None
image_interrogators = {}
global_unpickler = None


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
    if params.do_rename:
        basename = f"{img_index:05}"
    else:
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


def unload_system():
    disable_safe_unpickle()
    if shared.interrogator is not None:
        shared.interrogator.unload()
    try:
        shared.sd_model.to("cpu")
    except:
        pass


def load_system():
    enable_safe_unpickle()
    if shared.interrogator is not None:
        shared.interrogator.send_blip_to_ram()
    try:
        shared.sd_model.to(shared.device)
    except:
        pass


def get_clip_interrogator(params: ProcessParams):
    global clip_interrogator
    if clip_interrogator is None:
        clip_interrogator = CLIPInterrogator(params)
    else:
        clip_interrogator.unload()
    return clip_interrogator


def get_crop_clip():
    global crop_clip
    if crop_clip is None:
        try:
            del sys.modules['models']
        except:
            pass

        crop_clip = CropClip()
    return crop_clip


def get_image_interrogators(params: ProcessParams):
    global image_interrogators
    all_interrogators = InterrogatorRegistry.get_all_interrogators()
    interrogators = params.captioners
    caption_agents = []
    print(f"Interrogators: {interrogators}")
    for interrogator_name, enable in interrogators.items():
        if enable:
            if interrogator_name not in image_interrogators:
                printi(f"\rLoading {interrogator_name} interrogator...")
                if interrogator_name == "Clip":
                    interrogator = get_clip_interrogator(params.clip_params())
                else:
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
    # Remove non-alphanumeric characters except spaces
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', s)
    # Check for a sentence with just the same word repeated
    if len(set(cleaned.split())) == 1:
        cleaned = cleaned.split()[0]
    # Replace multiple spaces with a single space
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def build_caption(image, captions_list, tags_to_ignore, caption_length, subject_class, subject, replace_class,
                  txt_action="append"):
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
    # Read existing caption from path/txt file
    existing_caption_txt_filename = os.path.splitext(image)[0] + '.txt'
    existing_caption_txt = ''
    if os.path.exists(existing_caption_txt_filename):
        with open(existing_caption_txt_filename, 'r', encoding="utf8") as file:
            existing_caption_txt = file.read()
    else:
        existing_caption_txt = clean_string(os.path.basename(image).rstrip('.jpg.jpeg.png.gif'))

    all_tags = set()
    for cap in captions_list:
        all_tags.update({clean_string(tag) for tag in cap.split(",") if tag.strip()})

    if isinstance(tags_to_ignore, str):
        tags_to_ignore = tags_to_ignore.split(",")
    # Filter out ignored tags
    ignore_tags = set(clean_string(tag) for tag in tags_to_ignore if tag.strip())
    all_tags.difference_update(ignore_tags)

    # Handling existing caption based on txt_action
    existing_tags = set(clean_string(tag) for tag in existing_caption_txt.split(",") if tag.strip())
    if txt_action == "prepend":
        all_tags = existing_tags.union(all_tags)
    elif txt_action == 'append':
        all_tags = all_tags.union(existing_tags)
    elif txt_action == 'copy':
        all_tags = existing_tags

    # Replace class with subject
    if replace_class and subject and subject_class:
        all_tags = {tag.replace(clean_string(subject_class), clean_string(subject)) for tag in all_tags}

    # Limiting caption length
    tags_list = list(all_tags)
    if caption_length and len(tags_list) > caption_length:
        tags_list = tags_list[:caption_length]

    caption_txt = ", ".join(tags_list)
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


def crop_smart(img: Image, interrogator: CLIPInterrogator, cc: CropClip, params: ProcessParams):
    short_caption = interrogator.interrogate(img, params, short=True)
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


def process_pre(files, params: ProcessParams) -> Union[List[str], List[Image.Image]]:
    output = []
    interrogator = None
    cc = None
    if params.crop and params.crop_mode == "smart":
        interrogator = get_clip_interrogator(params.clip_params())
        cc = get_crop_clip()
    total_files = len(files)
    crop_length = 0
    if params.crop:
        crop_length += total_files
    if params.pad:
        crop_length += total_files
    pbar = tqdm(total=crop_length, desc="Processing images")
    for file in files:
        img = Image.open(file).convert("RGB")
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

        if params.save_image:
            img_path = save_pic(img, file, len(output), params)
            output.append(img_path)
        else:
            output.append(img)
    interrogator.unload()
    cc.unload()
    return output


def process_captions(files, params: ProcessParams) -> Dict[str, str]:
    caption_dict = {}
    caption_length = params.max_tokens
    tags_to_ignore = params.tags_to_ignore
    subject_class = params.subject_class
    subject = params.subject
    replace_class = params.replace_class
    txt_action = params.txt_action
    save_captions = params.save_caption

    agents = get_image_interrogators(params)
    total_files = len(files)
    total_captions = total_files * len(agents)
    pbar = tqdm(total=total_captions, desc="Captioning images")
    for caption_agent in agents:
        print(f"Captioning with {caption_agent.__class__.__name__}...")
        caption_agent.load()
        for image_to_caption in files:
            img = Image.open(image_to_caption).convert("RGB")
            if image_to_caption not in caption_dict:
                caption_dict[image_to_caption] = []
            try:
                caption_out = caption_agent.interrogate(img, params)
                print(f"Caption for {image_to_caption}: {caption_out}")
                caption_dict[image_to_caption].append(caption_out)
                pbar.update(1)
                shared.state.job_no += 1
            except Exception as e:
                print(f"Exception captioning {image_to_caption}: {e}")
                traceback.print_exc()
        caption_agent.unload()

    output_dict = {}
    for image_path, captions in caption_dict.items():
        caption_string = build_caption(image_path, captions, tags_to_ignore, caption_length, subject_class, subject,
                                       replace_class, txt_action)
        output_dict[image_path] = caption_string
        if save_captions:
            save_img_caption(image_path, caption_string, params)
    return output_dict


def process_post(files, params: ProcessParams) -> Union[
    List[str], List[Image.Image]]:
    output = []
    total_files = len(files)
    total_post = 0
    if params.restore_faces:
        total_post += total_files

    if params.upscale:
        total_post += total_files
    pbar = tqdm(total=total_post, desc="Post-processing images")
    params.do_rename = False
    upscalers = []

    if params.upscale:
        shared.state.textinfo = "Upscaling..."
        if params.upscaler_1 is not None:
            upscalers.append(params.upscaler_1)
        if params.upscaler_2 is not None:
            upscalers.append(params.upscaler_2)

    for file in files:
        img = Image.open(file).convert("RGB")
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

            # Apply each upscaler sequentially
            for scaler_name in upscalers:
                upscaler = shared.sd_upscalers[scaler_name]
                img = upscaler.upscale(img, upscale_by, upscaler.data_path)
                upscaler.unload()
                pbar.update(1)
                shared.state.job_no += 1
                shared.state.current_image = img

        if params.save_image:
            img_path = save_pic(img, file, 0, params)
            output.append(img_path)
        else:
            output.append(img)
    for scaler in upscalers:
        try:
            del scaler
        except:
            pass
    return output


def do_process(params: ProcessParams):
    print(f"Processing with params: {params}")
    outputs = []
    caption_dict = {}

    try:
        global clip_interrogator
        global image_interrogators

        job_length = calculate_job_length(params.src_files, params.crop, params.caption, params.captioners, params.flip,
                                          params.restore_faces, params.upscale)
        clip_params = params.clip_params()

        if job_length == 0:
            msg = "Nothing to do."
            printi(msg)
            return outputs, {}, msg

        unload_system()
        do_preprocess = params.pad or params.crop or params.flip
        do_postprocess = params.restore_faces or params.upscale

        shared.state.textinfo = "Initializing smart processing..."
        shared.state.job_count = job_length
        shared.state.job_no = 0
        out_images = params.src_files
        if do_preprocess:
            out_images = process_pre(out_images, params)
        else:
            out_images = params.src_files

        if params.caption:
            caption_dict = process_captions(out_images, params)

        if do_postprocess:
            out_images = process_post(out_images, params)
        outputs = out_images

        return out_images, caption_dict, f"Successfully processed {len(out_images)} images."
    except Exception as e:
        traceback.print_exc()
        msg = f"Error processing images: {e}"
        printi(msg)

    return outputs, caption_dict, msg
