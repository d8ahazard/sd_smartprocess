import math
import os
import sys
import traceback

import numpy as np
import tqdm
from PIL import Image, ImageOps

import modules.codeformer_model
import modules.gfpgan_model
import reallysafe
from clipcrop import CropClip
from extensions.sd_smartprocess.clipinterrogator import ClipInterrogator
from extensions.sd_smartprocess.interrogator import WaifuDiffusionInterrogator, BooruInterrogator
from modules import shared, images, safe
from modules.shared import cmd_opts

if cmd_opts.deepdanbooru:
    import modules.deepbooru as deepbooru


def printi(message):
    shared.state.textinfo = message
    print(message)


def preprocess(rename,
               src,
               dst,
               pad,
               crop,
               max_size,
               txt_action,
               flip,
               caption,
               caption_length,
               caption_clip,
               clip_use_v2,
               clip_append_flavor,
               clip_append_medium,
               clip_append_movement,
               clip_append_artist,
               clip_append_trending,
               caption_wd14,
               wd14_min_score,
               caption_deepbooru,
               booru_min_score,
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
        shared.state.textinfo = "Initializing smart processing..."
        safe.RestrictedUnpickler = reallysafe.RestrictedUnpickler

        if not crop and not caption and not restore_faces and not upscale and not pad:
            msg = "Nothing to do."
            printi(msg)
            return msg, msg

        wd_interrogator = None
        db_interrogator = None
        clip_interrogator = None
        crop_clip = None

        if caption or crop:
            printi("\rLoading captioning models...")
            if caption_clip or crop:
                printi("\rLoading CLIP interrogator...")
                if shared.interrogator is not None:
                    shared.interrogator.unload()
                clip_interrogator = ClipInterrogator(clip_use_v2,
                                                     clip_append_artist,
                                                     clip_append_medium,
                                                     clip_append_movement,
                                                     clip_append_flavor,
                                                     clip_append_trending)

            if caption_deepbooru:
                printi("\rLoading Deepbooru interrogator...")
                db_interrogator = BooruInterrogator()

            if caption_wd14:
                printi("\rLoading wd14 interrogator...")
                wd_interrogator = WaifuDiffusionInterrogator()

        if crop:
            printi("Loading YOLOv5 interrogator...")
            crop_clip = CropClip()

            del sys.modules['models']

        src = os.path.abspath(src)
        dst = os.path.abspath(dst)

        if src == dst:
            msg = "Source and destination are the same, returning."
            printi(msg)
            return msg, msg

        os.makedirs(dst, exist_ok=True)

        files = os.listdir(src)

        printi("Preprocessing...")
        shared.state.job_count = len(files)

        def build_caption(image):
            # Read existing caption from path/txt file
            existing_caption_txt_filename = os.path.splitext(filename)[0] + '.txt'
            if os.path.exists(existing_caption_txt_filename):
                with open(existing_caption_txt_filename, 'r', encoding="utf8") as file:
                    existing_caption_txt = file.read()
            else:
                existing_caption_txt = ''.join(c for c in filename if c.isalpha() or c in [" ", ", "])

            out_tags = []
            if clip_interrogator is not None:
                if caption_clip:
                    tags = clip_interrogator.interrogate(img)
                    for tag in tags:
                        #print(f"CLIPTag: {tag}")
                        out_tags.append(tag)

            if wd_interrogator is not None:
                ratings, tags = wd_interrogator.interrogate(img)

                for rating in ratings:
                    #print(f"Rating {rating} score is {ratings[rating]}")
                    if ratings[rating] >= wd14_min_score:
                        out_tags.append(rating)

                for tag in sorted(tags, key=tags.get, reverse=True):
                    if tags[tag] >= wd14_min_score:
                        #print(f"WDTag {tag} score is {tags[tag]}")
                        out_tags.append(tag)
                    else:
                        break

            if caption_deepbooru:
                tags = db_interrogator.interrogate(image)
                for tag in sorted(tags, key=tags.get, reverse=True):
                    if tags[tag] >= booru_min_score:
                        #print(f"DBTag {tag} score is {tags[tag]}")
                        out_tags.append(tag)

            # Remove duplicates
            unique_tags = []
            for tag in out_tags:
                if not tag in unique_tags:
                    unique_tags.append(tag.strip())
            caption_txt = ", ".join(unique_tags)

            if txt_action == 'prepend' and existing_caption_txt:
                caption_txt = existing_caption_txt + ' ' + caption_txt
            elif txt_action == 'append' and existing_caption_txt:
                caption_txt = caption_txt + ' ' + existing_caption_txt
            elif txt_action == 'copy' and existing_caption_txt:
                caption_txt = existing_caption_txt

            caption_txt = caption_txt.strip()
            if replace_class and subject is not None and subject_class is not None:
                # Find and replace "a SUBJECT CLASS" in caption_txt with subject name
                if f"a {subject_class}" in caption_txt:
                    caption_txt = caption_txt.replace(f"a {subject_class}", subject)

                if subject_class in caption_txt:
                    caption_txt = caption_txt.replace(subject_class, subject)

            if 0 < caption_length < len(caption_txt):
                split_cap = caption_txt.split(" ")
                caption_txt = ""
                cap_test = ""
                split_idx = 0
                while True and split_idx < len(split_cap):
                    cap_test += f" {split_cap[split_idx]}"
                    if len(cap_test) < caption_length:
                        caption_txt = cap_test
                    split_idx += 1

            caption_txt = caption_txt.strip()
            return caption_txt

        def save_pic(image, src_name, img_index, existing_caption=None, flipped=False):
            if rename:
                basename = f"{img_index:05}"
            else:
                basename = src_name
                if flipped:
                    basename += "_flipped"

            shared.state.current_image = img
            image.save(os.path.join(dst, f"{basename}.png"))

            if existing_caption is not None and len(existing_caption) > 0:
                with open(os.path.join(dst, f"{basename}.txt"), "w", encoding="utf8") as file:
                    file.write(existing_caption)

        image_index = 0

        # Enumerate images
        for index, src_image in enumerate(tqdm.tqdm(files)):
            # Quit on cancel
            if shared.state.interrupted:
                msg = f"Processing interrupted, {index}/{len(files)}"
                return msg, msg

            filename = os.path.join(src, src_image)
            try:
                img = Image.open(filename).convert("RGB")
            except Exception as e:
                msg = f"Exception processing: {e}"
                printi(msg)
                traceback.print_exc()
                return msg, msg

            if crop:
                # Interrogate once
                short_caption = clip_interrogator.interrogate(img, short=True)

                if subject_class is not None and subject_class != "":
                    short_caption = subject_class

                shared.state.textinfo = f"Cropping: {short_caption}"
                src_ratio = img.width / img.height

                # Pad image before cropping?
                if src_ratio != 1 and pad:
                    if img.width > img.height:
                        pad_width = img.width
                        pad_height = img.width
                    else:
                        pad_width = img.height
                        pad_height = img.height
                    res = Image.new("RGB", (pad_width, pad_height))
                    res.paste(img, box=(pad_width // 2 - img.width // 2, pad_height // 2 - img.height // 2))
                    img = res

                # Do the actual crop clip
                im_data = crop_clip.get_center(img, prompt=short_caption)
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
                shared.state.current_image = img

            if pad:
                ratio = 1
                src_ratio = img.width / img.height

                src_w = max_size if ratio < src_ratio else img.width * max_size // img.height
                src_h = max_size if ratio >= src_ratio else img.height * max_size // img.width

                resized = images.resize_image(0, img, src_w, src_h)
                res = Image.new("RGB", (max_size, max_size))
                res.paste(resized, box=(max_size // 2 - src_w // 2, max_size // 2 - src_h // 2))
                img = res

            # Resize again if image is not at the right size.
            if img.width != max_size or img.height != max_size:
                img = images.resize_image(1, img, max_size, max_size)

            # Build a caption, if enabled
            full_caption = build_caption(img) if caption else None
            # Show our output
            shared.state.current_image = img
            printi(f"Processed: '({src_image} - {full_caption})")

            save_pic(img, src_image, image_index, existing_caption=full_caption)
            image_index += 1

            if flip:
                save_pic(ImageOps.flip(img), src_image, image_index, existing_caption=full_caption, flipped=True)
                image_index += 1

            shared.state.nextjob()

        if caption_clip or crop:
            printi("Unloading CLIP interrogator...")
            shared.interrogator.send_blip_to_ram()

        if caption_deepbooru:
            printi("Unloading Deepbooru interrogator...")
            db_interrogator.unload()

        if caption_wd14:
            printi("Unloading wd14 interrogator...")
            wd_interrogator.unload()

        return f"Successfully processed {len(files)} images.", f"Successfully processed {len(files)} images."

    except Exception as e:
        msg = f"Exception processing: {e}"
        traceback.print_exc()
        pass

    return msg, msg
