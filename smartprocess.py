import math
import os
import sys

import numpy as np
import tqdm
from PIL import Image, ImageOps

from clipcrop import CropClip
import reallysafe
from modules import shared, images, safe
import modules.gfpgan_model
import modules.codeformer_model
from modules.shared import opts, cmd_opts

if cmd_opts.deepdanbooru:
    import modules.deepbooru as deepbooru


def interrogate_image(image: Image):
    prev_artists = shared.opts.interrogate_use_builtin_artists
    prev_max = shared.opts.interrogate_clip_max_length
    prev_min = shared.opts.interrogate_clip_min_length
    shared.opts.interrogate_clip_min_length = 10
    shared.opts.interrogate_clip_max_length = 20
    shared.opts.interrogate_use_builtin_artists = False
    prompt = shared.interrogator.interrogate(image)
    shared.opts.interrogate_clip_min_length = prev_min
    shared.opts.interrogate_clip_max_length = prev_max
    shared.opts.interrogate_use_builtin_artists = prev_artists
    full_caption = shared.interrogator.interrogate(image)
    return prompt, full_caption


def preprocess(src,
               dst,
               crop,
               width,
               height,
               append_filename,
               save_txt,
               pretxt_action,
               flip,
               split,
               caption,
               caption_length,
               caption_deepbooru,
               split_threshold,
               overlap_ratio,
               subject_class,
               subject,
               replace_class,
               restore_faces,
               face_model,
               upscale,
               upscale_ratio,
               scaler
               ):
    try:
        shared.state.textinfo = "Loading models for smart processing..."
        safe.RestrictedUnpickler = reallysafe.RestrictedUnpickler
        if caption:
            shared.interrogator.load()

        if caption_deepbooru:
            db_opts = deepbooru.create_deepbooru_opts()
            db_opts[deepbooru.OPT_INCLUDE_RANKS] = False
            deepbooru.create_deepbooru_process(opts.interrogate_deepbooru_score_threshold, db_opts)

        prework(src,
                dst,
                crop,
                width,
                height,
                append_filename,
                save_txt,
                pretxt_action,
                flip,
                split,
                caption,
                caption_length,
                caption_deepbooru,
                split_threshold,
                overlap_ratio,
                subject_class,
                subject,
                replace_class,
                restore_faces,
                face_model,
                upscale,
                upscale_ratio,
                scaler)

    finally:

        if caption:
            shared.interrogator.send_blip_to_ram()

        if caption_deepbooru:
            deepbooru.release_process()

    return "Processing complete.", ""


def prework(src,
            dst,
            crop_image,
            width,
            height,
            append_filename,
            save_txt,
            pretxt_action,
            flip,
            split,
            caption_image,
            caption_length,
            caption_deepbooru,
            split_threshold,
            overlap_ratio,
            subject_class,
            subject,
            replace_class,
            restore_faces,
            face_model,
            upscale,
            upscale_ratio,
            scaler):
    try:
        del sys.modules['models']
    except:
        pass
    width = width
    height = height
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)

    if not crop_image and not caption_image and not restore_faces and not upscale:
        print("Nothing to do.")
        shared.state.textinfo = "Nothing to do!"
        return

    assert src != dst, 'same directory specified as source and destination'

    os.makedirs(dst, exist_ok=True)

    files = os.listdir(src)

    shared.state.textinfo = "Preprocessing..."
    shared.state.job_count = len(files)

    def build_caption(image, caption):
        existing_caption = None
        if not append_filename:
            existing_caption_filename = os.path.splitext(filename)[0] + '.txt'
            if os.path.exists(existing_caption_filename):
                with open(existing_caption_filename, 'r', encoding="utf8") as file:
                    existing_caption = file.read()
        else:
            existing_caption = ''.join(c for c in filename if c.isalpha() or c in [" ", ","])

        if caption_deepbooru:
            if len(caption) > 0:
                caption += ", "
            caption += deepbooru.get_tags_from_process(image)

        if pretxt_action == 'prepend' and existing_caption:
            caption = existing_caption + ' ' + caption
        elif pretxt_action == 'append' and existing_caption:
            caption = caption + ' ' + existing_caption
        elif pretxt_action == 'copy' and existing_caption:
            caption = existing_caption

        caption = caption.strip()
        if replace_class and subject is not None and subject_class is not None:
            # Find and replace "a SUBJECT CLASS" in caption with subject name
            if f"a {subject_class}" in caption:
                caption = caption.replace(f"a {subject_class}", subject)

            if subject_class in caption:
                caption = caption.replace(subject_class, subject)

        if 0 < caption_length < len(caption):
            split_cap = caption.split(" ")
            caption = ""
            cap_test = ""
            split_idx = 0
            while True and split_idx < len(split_cap):
                cap_test += f" {split_cap[split_idx]}"
                if len(cap_test < caption_length):
                    caption = cap_test
                split_idx += 1

        caption = caption.strip()
        return caption

    def save_pic_with_caption(image, img_index, existing_caption):

        if not append_filename:
            filename_part = filename
            filename_part = os.path.splitext(filename_part)[0]
            filename_part = os.path.basename(filename_part)
        else:
            filename_part = existing_caption

        basename = f"{img_index:05}-{subindex[0]}-{filename_part}"
        shared.state.current_image = img
        image.save(os.path.join(dst, f"{basename}.png"))

        if save_txt:
            if len(existing_caption) > 0:
                with open(os.path.join(dst, f"{basename}.txt"), "w", encoding="utf8") as file:
                    file.write(existing_caption)

        subindex[0] += 1

    def save_pic(image, img_index, existing_caption=None):
        save_pic_with_caption(image, img_index, existing_caption=existing_caption)

        if flip:
            save_pic_with_caption(ImageOps.mirror(image), img_index, existing_caption=existing_caption)

    def split_pic(image, img_inverse_xy):
        if img_inverse_xy:
            from_w, from_h = image.height, image.width
            to_w, to_h = height, width
        else:
            from_w, from_h = image.width, image.height
            to_w, to_h = width, height
        h = from_h * to_w // from_w
        if img_inverse_xy:
            image = image.resize((h, to_w))
        else:
            image = image.resize((to_w, h))

        split_count = math.ceil((h - to_h * overlap_ratio) / (to_h * (1.0 - overlap_ratio)))
        y_step = (h - to_h) / (split_count - 1)
        for i in range(split_count):
            y = int(y_step * i)
            if img_inverse_xy:
                split_img = image.crop((y, 0, y + to_h, to_w))
            else:
                split_img = image.crop((0, y, to_w, y + to_h))
            yield split_img

    crop_clip = None

    if crop_image:
        split_threshold = max(0.0, min(1.0, split_threshold))
        overlap_ratio = max(0.0, min(0.9, overlap_ratio))
        crop_clip = CropClip()

    for index, imagefile in enumerate(tqdm.tqdm(files)):

        if shared.state.interrupted:
            break

        subindex = [0]
        filename = os.path.join(src, imagefile)
        try:
            img = Image.open(filename).convert("RGB")
        except Exception:
            continue

        # Interrogate once
        short_caption, full_caption = interrogate_image(img)

        if subject_class is not None and subject_class != "":
            short_caption = subject_class

        # Build our caption
        if caption_image:
            full_caption = build_caption(img, full_caption)
        shared.state.current_image = img
        shared.state.textinfo = f"Processing: '{full_caption}' ({filename})"
        if crop_image:
            shared.state.textinfo = "Cropping..."
            if img.height > img.width:
                ratio = (img.width * height) / (img.height * width)
                inverse_xy = False
            else:
                ratio = (img.height * width) / (img.width * height)
                inverse_xy = True

            if split and ratio < 1.0 and ratio <= split_threshold:
                for splitted in split_pic(img, inverse_xy):
                    save_pic(splitted, index, existing_caption=full_caption)

            im_data = crop_clip.get_center(img, prompt=short_caption)
            crop_width = im_data[1] - im_data[0]
            center_x = im_data[0] + (crop_width / 2)
            crop_height = im_data[3] - im_data[2]
            center_y = im_data[2] + (crop_height / 2)
            crop_ratio = crop_width / crop_height
            dest_ratio = width / height
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
            default_resize = True
            shared.state.current_image = img
        else:
            default_resize = False

        if restore_faces:
            shared.state.textinfo = f"Restoring faces using {face_model}..."
            if face_model == "gfpgan":
                restored_img = modules.gfpgan_model.gfpgan_fix_faces(np.array(img, dtype=np.uint8))
                img = Image.fromarray(restored_img)
            else:
                restored_img = modules.codeformer_model.codeformer.restore(np.array(img, dtype=np.uint8),
                                                                           w=1.0)
                img = Image.fromarray(restored_img)
            shared.state.current_image = img

        if upscale:
            shared.state.textinfo = "Upscaling..."
            upscaler = shared.sd_upscalers[scaler]
            res = upscaler.scaler.upscale(img, upscale_ratio, upscaler.data_path)
            img = res
            default_resize = True
            shared.state.current_image = img

        if default_resize:
            img = images.resize_image(1, img, width, height)
        shared.state.current_image = img
        save_pic(img, index, existing_caption=full_caption)

        shared.state.nextjob()
