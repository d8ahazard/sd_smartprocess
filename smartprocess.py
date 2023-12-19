import os
import sys
import traceback
from io import StringIO
from pathlib import Path
from typing import Union, Dict, List

import numpy as np
from PIL import Image, features, ImageOps
from tqdm import tqdm

import modules.codeformer_model
import modules.gfpgan_model
from clipcrop import CropClip
from extensions.sd_smartprocess.blipinterrogator import BlipInterrogator
from extensions.sd_smartprocess.clipinterrogator import ClipInterrogator
from extensions.sd_smartprocess.interrogator import WaifuDiffusionInterrogator, BooruInterrogator
from extensions.sd_smartprocess.llava_interrogator import LLAVAInterrogator
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


def get_clip_interrogator(clip_params: Dict):
    clip_use_v2 = clip_params["use_v2"]
    clip_append_artist = clip_params["append_artist"]
    clip_append_medium = clip_params["append_medium"]
    clip_append_movement = clip_params["append_movement"]
    clip_append_flavor = clip_params["append_flavor"]
    clip_append_trending = clip_params["append_trending"]
    num_beams = clip_params["num_beams"]
    min_clip = clip_params["min_clip"]
    max_clip = clip_params["max_clip"]
    clip_max_flavors = clip_params["clip_max_flavors"]
    blip_initial_prompt = clip_params["blip_initial_prompt"]
    global clip_interrogator
    if clip_interrogator is None:
        clip_interrogator = ClipInterrogator(clip_use_v2,
                                             clip_append_artist,
                                             clip_append_medium,
                                             clip_append_movement,
                                             clip_append_flavor,
                                             clip_append_trending,
                                             num_beams,
                                             min_clip,
                                             max_clip,
                                             clip_max_flavors,
                                             blip_initial_prompt)
    else:
        clip_interrogator.use_v2 = clip_use_v2
        clip_interrogator.append_artist = clip_append_artist
        clip_interrogator.append_medium = clip_append_medium
        clip_interrogator.append_movement = clip_append_movement
        clip_interrogator.append_flavor = clip_append_flavor
        clip_interrogator.append_trending = clip_append_trending
        clip_interrogator.num_beams = num_beams
        clip_interrogator.min_clip = min_clip
        clip_interrogator.max_clip = max_clip
        clip_interrogator.max_flavors = clip_max_flavors
        clip_interrogator.blip_initial_prompt = blip_initial_prompt
        clip_interrogator.set_model_type(clip_use_v2)
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


def get_image_interrogators(interrogators: Dict[str, bool], clip_params: Dict, booru_score: float, wd14_score: float):
    global image_interrogators
    print(f"Interrogators: {interrogators}")
    caption_blip = interrogators.get("BLIP", False)
    caption_clip = interrogators.get("CLIP", False)
    caption_deepbooru = interrogators.get("DeepDanbooru", False)
    caption_wd14 = interrogators.get("WD14", False)
    caption_llava = interrogators.get("LLAVA", False)
    # caption_mplug_2 = interrogators.get("MPLUG2", False)
    caption_agents = []
    if caption_blip:
        if "BLIP" not in image_interrogators:
            printi("\rLoading BLIP interrogator...")
            blip_interrogator = BlipInterrogator(
                clip_params.get("blip_initial_prompt", "a caption for this image is: "))
            image_interrogators["BLIP"] = blip_interrogator
        else:
            blip_interrogator = image_interrogators["BLIP"]
        caption_agents.append(blip_interrogator)
    if caption_clip:
        caption_agents.append(get_clip_interrogator(clip_params))
    if caption_deepbooru:
        if "DeepDanbooru" not in image_interrogators:
            printi("\rLoading Deepbooru interrogator...")
            db_interrogator = BooruInterrogator(booru_score)
            image_interrogators["DeepDanbooru"] = db_interrogator
        else:
            image_interrogators["DeepDanbooru"].min_score = booru_score
            db_interrogator = image_interrogators["DeepDanbooru"]
        db_interrogator.unload()
        caption_agents.append(db_interrogator)
    if caption_wd14:
        if "WD14" not in image_interrogators:
            printi("\rLoading wd14 interrogator...")
            wd_interrogator = WaifuDiffusionInterrogator(min_score=wd14_score)
            image_interrogators["WD14"] = wd_interrogator
        else:
            image_interrogators["WD14"].min_score = wd14_score
            wd_interrogator = image_interrogators["WD14"]
        wd_interrogator.unload()
        caption_agents.append(wd_interrogator)
    if caption_llava:
        if "LLAVA" not in image_interrogators:
            printi("\rLoading LLAVA interrogator...")
            llava_interrogator = LLAVAInterrogator()
            image_interrogators["LLAVA"] = llava_interrogator
        else:
            llava_interrogator = image_interrogators["LLAVA"]
        llava_interrogator.unload()
        caption_agents.append(llava_interrogator)
    # if caption_mplug_2:
    #     if "MPLUG2" not in image_interrogators:
    #         printi("\rLoading MPLUG2 interrogator...")
    #         mplug_2_interrogator = MPLUG2Interrrogator()
    #     else:
    #         mplug_2_interrogator = image_interrogators["MPLUG2"]
    #     mplug_2_interrogator.unload()
    #     caption_agents.append(mplug_2_interrogator)
    return caption_agents


def build_caption(image, caption_array, tags_to_ignore, caption_length, subject_class, subject, replace_class,
                  txt_action="append"):
    # Read existing caption from path/txt file
    existing_caption_txt_filename = os.path.splitext(image)[0] + '.txt'
    if os.path.exists(existing_caption_txt_filename):
        with open(existing_caption_txt_filename, 'r', encoding="utf8") as file:
            existing_caption_txt = file.read()
    else:
        existing_caption_txt = ''.join(c for c in image if c.isalpha() or c in [" ", ", "])

    all_tags = []
    for cap in caption_array:
        tags = cap.split(",")
        for tag in tags:
            stripped = tag.strip()
            if stripped not in all_tags:
                all_tags.append(stripped)
    # Remove duplicates, filter dumb stuff
    chars_to_strip = ["_\\("]
    unique_tags = []
    ignore_tags = []
    if tags_to_ignore != "" and tags_to_ignore is not None:
        si_tags = tags_to_ignore.split(",")
        for tag in si_tags:
            ignore_tags.append(tag.strip)

    for tag in all_tags:
        if tag not in unique_tags and "_\(" not in tag and tag not in ignore_tags:
            unique_tags.append(tag.strip())

    existing_tags = existing_caption_txt.split(",")

    if txt_action == "prepend" and len(existing_tags):
        new_tags = existing_tags
        for tag in unique_tags:
            if not tag in new_tags:
                new_tags.append(tag)
        unique_tags = new_tags

    if txt_action == 'append' and len(existing_tags):
        for tag in existing_tags:
            if not tag in unique_tags:
                unique_tags.append(tag)

    if txt_action == 'copy' and existing_caption_txt:
        for tag in existing_tags:
            unique_tags.append(tag.strip())

    caption_txt = ", ".join(unique_tags)

    if replace_class and subject is not None and subject_class is not None:
        # Find and replace "a SUBJECT CLASS" in caption_txt with subject name
        if f"a {subject_class}" in caption_txt:
            caption_txt = caption_txt.replace(f"a {subject_class}", subject)

        if subject_class in caption_txt:
            caption_txt = caption_txt.replace(subject_class, subject)

    tags = caption_txt.split(" ")

    if caption_length != 0 and len(tags) > caption_length:
        tags = tags[0:caption_length]
        tags[-1] = tags[-1].rstrip(",")
    caption_txt = " ".join(tags)
    return caption_txt


def save_img_caption(image, img_caption):
    basename = os.path.splitext(image)[0]
    dest = f"{basename}.txt"
    if img_caption is not None and len(img_caption) > 0:
        with open(dest, "w", encoding="utf8") as file:
            file.write(dest)
    return dest


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


def process_cropping(files, dest, clip_params, crop, pad, max_size, flip, do_rename, save=False) -> Union[
    List[str], List[Image.Image]]:
    output = []
    interrogator = get_clip_interrogator(clip_params)
    cc = get_crop_clip()
    total_files = len(files)
    crop_length = 0
    if crop:
        crop_length += total_files
    if flip:
        crop_length += total_files
    if pad:
        crop_length += total_files
    pbar = tqdm(total=crop_length, desc="Processing images")
    for file in files:
        img = Image.open(file).convert("RGB")
        if crop:
            short_caption = interrogator.interrogate(img, short=True)
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
            shared.state.current_image = img
            pbar.update(1)
            shared.state.job_no += 1

        if pad:
            ratio = 1
            src_ratio = img.width / img.height

            src_w = max_size if ratio < src_ratio else img.width * max_size // img.height
            src_h = max_size if ratio >= src_ratio else img.height * max_size // img.width

            resized = images.resize_image(0, img, src_w, src_h)
            res = Image.new("RGB", (max_size, max_size))
            res.paste(resized, box=(max_size // 2 - src_w // 2, max_size // 2 - src_h // 2))
            img = res
            pbar.update(1)
            shared.state.job_no += 1

        if flip:
            img = ImageOps.flip(img)
            pbar.update(1)
            shared.state.job_no += 1
        if save:
            img_path = save_pic(img, file, dest, len(output), do_rename)
            output.append(img_path)
        else:
            output.append(img)
    interrogator.unload()
    cc.unload()
    return output


def process_captions(files, captioners, caption_length, clip_params, wd14_min_score, booru_min_score, tags_to_ignore,
                     subject_class, subject, replace_class, txt_action, save_captions) -> Dict[str, str]:
    caption_dict = {}

    interrogators = get_image_interrogators(captioners, clip_params, booru_min_score, wd14_min_score)
    total_files = len(files)
    total_captions = total_files * len(interrogators)
    pbar = tqdm(total=total_captions, desc="Captioning images")
    for caption_agent in interrogators:
        print(f"Captioning with {caption_agent.__class__.__name__}...")
        caption_agent.load()
        for image_to_caption in files:
            img = Image.open(image_to_caption).convert("RGB")
            if image_to_caption not in caption_dict:
                caption_dict[image_to_caption] = []
            try:
                caption_out = caption_agent.interrogate(img)
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
            save_img_caption(image_path, caption_string)
    return output_dict


def process_post(files, dest, restore_faces, face_model, upscale, upscale_ratio, scaler, save=False) -> Union[
    List[str], List[Image.Image]]:
    output = []
    total_files = len(files)
    total_post = 0
    if restore_faces:
        total_post += total_files

    if upscale:
        total_post += total_files
    pbar = tqdm(total=total_post, desc="Post-processing images")

    for file in files:
        img = Image.open(file).convert("RGB")
        if restore_faces:
            shared.state.textinfo = f"Restoring faces using {face_model}..."
            if face_model == "gfpgan":
                restored_img = modules.gfpgan_model.gfpgan_fix_faces(np.array(img, dtype=np.uint8))
                img = Image.fromarray(restored_img)
            else:
                restored_img = modules.codeformer_model.codeformer.restore(np.array(img, dtype=np.uint8),
                                                                           w=1.0)
                img = Image.fromarray(restored_img)
            pbar.update(1)
            shared.state.job_no += 1
            shared.state.current_image = img

        if upscale:
            shared.state.textinfo = "Upscaling..."
            upscaler = shared.sd_upscalers[scaler]
            res = upscaler.scaler.upscale(img, upscale_ratio, upscaler.data_path)
            img = res
            pbar.update(1)
            shared.state.job_no += 1
            shared.state.current_image = img

        if save:
            img_path = save_pic(img, file, dest, len(output))
            output.append(img_path)
        else:
            output.append(img)
    return output


def save_pic(img, src_name, dst, img_index, do_rename=False, flipped=False):
    if do_rename:
        basename = f"{img_index:05}"
    else:
        basename = os.path.splitext(src_name)[0]
        if flipped:
            basename += "_flipped"

    shared.state.current_image = img
    dest = os.path.join(dst, f"{basename}.png")
    img.save(dest)
    return dest


def caption_images(src_files,
                   captioners,
                   blip_initial_prompt,
                   caption_length,
                   txt_action,
                   num_beams,
                   min_clip,
                   max_clip,
                   clip_use_v2,
                   clip_append_flavor,
                   clip_max_flavors,
                   clip_append_medium,
                   clip_append_movement,
                   clip_append_artist,
                   clip_append_trending,
                   wd14_min_score,
                   booru_min_score,
                   tags_to_ignore,
                   subject_class,
                   subject,
                   replace_class,
                   save_output: bool = False
                   ):
    params = ProcessParams()
    params.src_files = src_files
    params.caption = True
    params.captioners = captioners
    params.blip_initial_prompt = blip_initial_prompt
    params.caption_length = caption_length
    params.txt_action = txt_action
    params.num_beams = num_beams
    params.min_clip = min_clip
    params.max_clip = max_clip
    params.clip_use_v2 = clip_use_v2
    params.clip_append_flavor = clip_append_flavor
    params.clip_max_flavors = clip_max_flavors
    params.clip_append_medium = clip_append_medium
    params.clip_append_movement = clip_append_movement
    params.clip_append_artist = clip_append_artist
    params.clip_append_trending = clip_append_trending
    params.wd14_min_score = wd14_min_score
    params.booru_min_score = booru_min_score
    params.tags_to_ignore = tags_to_ignore
    params.subject_class = subject_class
    params.subject = subject
    params.replace_class = replace_class
    params.save_caption = save_output

    try:
        return do_process(params)
    except Exception as e:
        traceback.print_exc()
        msg = f"Error processing images: {e}"
        printi(msg)
        return [], {}, msg


def process_images(rename, src_files, dst, pad, crop, max_size, flip, save_output: bool = False):
    params = ProcessParams()
    params.rename = rename
    params.src_files = src_files
    params.dst = dst
    params.pad = pad
    params.crop = crop
    params.max_size = max_size
    params.flip = flip
    params.save_image = save_output
    try:
        return do_process(params)
    except Exception as e:
        traceback.print_exc()
        msg = f"Error processing images: {e}"
        printi(msg)
        return [], {}, msg


def postprocess_images(src_files, dst, restore_faces, face_model, upscale, upscale_ratio, scaler,
                       save_output: bool = False):
    params = ProcessParams()
    params.src_files = src_files
    params.dst = dst
    params.restore_faces = restore_faces
    params.face_model = face_model
    params.upscale = upscale
    params.upscale_ratio = upscale_ratio
    params.scaler = scaler
    params.save_image = save_output
    try:
        return do_process(params)
    except Exception as e:
        traceback.print_exc()
        msg = f"Error processing images: {e}"
        printi(msg)
        return [], {}, msg


def process_all(rename,
                src,
                dst,
                pad,
                crop,
                size,
                txt_action,
                flip,
                caption,
                captioners,
                caption_length,
                blip_initial_prompt,
                num_beams,
                min_clip,
                max_clip,
                clip_use_v2,
                clip_append_flavor,
                clip_max_flavors,
                clip_append_medium,
                clip_append_movement,
                clip_append_artist,
                clip_append_trending,
                wd14_min_score,
                booru_min_score,
                tags_to_ignore,
                subject_class,
                subject,
                replace_class,
                restore_faces,
                face_model,
                upscale,
                upscale_ratio,
                scaler):
    params = ProcessParams()
    params.rename = rename
    # Walk the source directory and get all files
    params.src_files = []
    for root, dirs, files in os.walk(src):
        for file in files:
            if is_image(file):
                params.src_files.append(os.path.join(root, file))
    params.dst = dst
    params.pad = pad
    params.crop = crop
    params.max_size = size
    params.txt_action = txt_action
    params.flip = flip
    params.blip_initial_prompt = blip_initial_prompt
    params.caption = caption
    params.captioners = captioners
    params.caption_length = caption_length
    params.num_beams = num_beams
    params.min_clip = min_clip
    params.max_clip = max_clip
    params.clip_use_v2 = clip_use_v2
    params.clip_append_flavor = clip_append_flavor
    params.clip_max_flavors = clip_max_flavors
    params.clip_append_medium = clip_append_medium
    params.clip_append_movement = clip_append_movement
    params.clip_append_artist = clip_append_artist
    params.clip_append_trending = clip_append_trending
    params.wd14_min_score = wd14_min_score
    params.booru_min_score = booru_min_score
    params.tags_to_ignore = tags_to_ignore
    params.subject_class = subject_class
    params.subject = subject
    params.replace_class = replace_class
    params.restore_faces = restore_faces
    params.face_model = face_model
    params.upscale = upscale
    params.upscale_ratio = upscale_ratio
    params.scaler = scaler
    params.save_image = True
    params.save_caption = True
    try:
        return do_process(params)
    except Exception as e:
        traceback.print_exc()
        msg = f"Error processing images: {e}"
        printi(msg)
        return [], {}, msg


def do_process(params: ProcessParams):
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
            return outputs, msg

        unload_system()
        do_preprocess = params.pad or params.crop or params.flip
        do_postprocess = params.restore_faces or params.upscale
        src = os.path.abspath(os.path.dirname(params.src_files[0]))
        if params.dst is None or params.dst == "":
            params.dst = os.path.dirname(src) + "_processed"
        params.dst = os.path.abspath(params.dst)

        if src == params.dst:
            msg = "Source and destination are the same, returning."
            printi(msg)
            return outputs, msg

        os.makedirs(params.dst, exist_ok=True)

        shared.state.textinfo = "Initializing smart processing..."
        shared.state.job_count = job_length
        shared.state.job_no = 0

        if do_preprocess:
            out_images = process_cropping(params.src_files, params.dst, clip_params, params.crop, params.pad,
                                          params.max_size, params.flip, params.rename, save=params.save_image)
        else:
            out_images = params.src_files

        if params.caption:
            caption_dict = process_captions(out_images, params.captioners, params.caption_length, clip_params,
                                            params.wd14_min_score,
                                            params.booru_min_score, params.tags_to_ignore, params.subject_class,
                                            params.subject, params.replace_class,
                                            params.txt_action, save_captions=params.save_caption)

        if do_postprocess:
            out_images = process_post(out_images, params.dst, params.restore_faces, params.face_model, params.upscale,
                                      params.upscale_ratio, params.scaler, save=params.save_image)
        outputs = out_images

        return out_images, caption_dict, f"Successfully processed {len(out_images)} images."
    except Exception as e:
        traceback.print_exc()
        msg = f"Error processing images: {e}"
        printi(msg)

    return outputs, caption_dict, msg
