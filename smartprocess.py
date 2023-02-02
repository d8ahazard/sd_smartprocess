import os
import traceback
from io import StringIO
from pathlib import Path

from PIL import Image, features
from tqdm import tqdm

from extensions.sd_smartprocess.processors import WDProcessor, ClipProcessor, CropProcessor, BooruProcessor, \
    UpscaleProcessor, SystemUpscaleProcessor
from modules import shared


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


def is_image(path: str, feats=None):
    path = Path(path)
    if feats is None:
        feats = []
    if not len(feats):
        feats = list_features()
    is_img = path.is_file() and path.suffix.lower() in feats
    return is_img


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
               caption_wd14,
               wd14_min_score,
               caption_deepbooru,
               booru_min_score,
               tags_to_ignore,
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

        def build_captions(image_paths, all_captions, all_wd_tags, all_db_tags):
            output = []
            for i in len(image_paths):
                filename = image_paths[i]
                # Read existing caption from path/txt file
                existing_caption_txt_filename = os.path.splitext(filename)[0] + '.txt'
                if os.path.exists(existing_caption_txt_filename):
                    with open(existing_caption_txt_filename, 'r', encoding="utf8") as file:
                        existing_caption_txt = file.read()
                else:
                    existing_caption_txt = ''.join(c for c in filename if c.isalpha() or c in [" ", ", "])

                out_tags = []
                if len(all_captions) > i:
                    tags = all_captions[i]
                    for tag in tags:
                        # print(f"CLIPTag: {tag}")
                        out_tags.append(tag)

                if len(all_wd_tags) > i:
                    tags = all_wd_tags[i]
                    for tag in sorted(tags, key=tags.get, reverse=True):
                        if tags[tag] >= wd14_min_score:
                            # print(f"WDTag {tag} score is {tags[tag]}")
                            out_tags.append(tag)
                        else:
                            break

                if len(all_db_tags) > i:
                    tags = all_db_tags[i]
                    for tag in sorted(tags, key=tags.get, reverse=True):
                        if tags[tag] >= booru_min_score:
                            # print(f"DBTag {tag} score is {tags[tag]}")
                            out_tags.append(tag)

                # Remove duplicates, filter dumb stuff
                chars_to_strip = ["_\\("]
                unique_tags = []
                ignore_tags = []
                if tags_to_ignore != "" and tags_to_ignore is not None:
                    si_tags = tags_to_ignore.split(",")
                    for tag in si_tags:
                        ignore_tags.append(tag.strip())

                main_tag = out_tags.pop(0)
                for tag in out_tags:
                    if tag not in unique_tags and "_\(" not in tag and tag not in ignore_tags:
                        add = True
                        for ignore_tag in ignore_tags:
                            if ignore_tag in tag:
                                add = False
                        if add:
                            unique_tags.append(tag.strip())

                existing_tags = existing_caption_txt.split(",")

                if txt_action == "prepend" and len(existing_tags):
                    new_tags = existing_tags
                    for tag in unique_tags:
                        if tag not in new_tags:
                            new_tags.append(tag)
                    unique_tags = new_tags

                unique_tags.insert(0, main_tag)
                if txt_action == 'append' and len(existing_tags):
                    for tag in existing_tags:
                        if tag not in unique_tags:
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
                output.append(caption_txt)
            return output

        def save_images(images, image_paths, captions):
            shared.state.job_count = len(images)
            shared.state.textinfo = "Saving images..."
            for img_index in tqdm(range(len(images)), desc="Saving images"):
                image = images[img_index]
                src_name = image_paths[img_index]
                caption = captions[img_index] if len(captions) > img_index else None
                if rename:
                    basename = f"{img_index:05}"
                else:
                    basename = os.path.splitext(src_name)[0]

                shared.state.current_image = image
                image.save(os.path.join(dst, f"{basename}.png"))

                if caption is not None and len(caption) > 0:
                    with open(os.path.join(dst, f"{basename}.txt"), "w", encoding="utf8") as file:
                        file.write(caption)
                shared.state.job_no += 1

        if not crop and not caption and not restore_faces and not upscale and not pad:
            msg = "Nothing to do."
            printi(msg)
            return msg, msg

        src = os.path.abspath(src)
        dst = os.path.abspath(dst)

        if src == dst:
            msg = "Source and destination are the same, returning."
            printi(msg)
            return msg, msg

        # Move SD Model to CPU
        shared.sd_model.to("cpu")
        shared.state.textinfo = "Initializing smart processing..."

        # Turn this awful junk off
        was_safe_unpickle = shared.cmd_opts.disable_safe_unpickle
        shared.cmd_opts.disable_safe_unpickle = True

        # Make dest
        os.makedirs(dst, exist_ok=True)

        # Enumerate images
        pil_features = list_features()
        files = os.listdir(src)

        src_files = [os.path.join(src, file) for file in files if is_image(os.path.join(src, file), pil_features)]

        src_images = []

        for src_path in src_files:
            src_images.append(Image.open(src_path))

        printi("Preprocessing...")
        short_caps = []
        captions = []
        wd_tags = []
        db_tags = []

        wd_cap = None
        if crop or pad or upscale:
            wd_cap = ClipProcessor(
                clip_use_v2,
                clip_append_artist,
                clip_append_medium,
                clip_append_movement,
                clip_append_flavor,
                clip_append_trending,
                num_beams,
                min_clip,
                max_clip,
                clip_max_flavors
            )
            short_caps = wd_cap.process(src_images, True)
            if not caption_clip:
                wd_cap.unload()
            else:
                # Move this to CPU while cropping
                wd_cap.model.clip_model.to("cpu")
            if crop or pad:
                cropper = CropProcessor(subject_class, pad, crop)
                src_images = cropper.process(src_images, short_caps)
                cropper.unload()

        if caption:
            if caption_clip:
                captions = wd_cap.process(src_images, False)
                wd_cap.unload()

            if caption_wd14:
                wd_processor = WDProcessor(wd14_min_score)
                wd_tags = wd_processor.process(src_images)
                wd_processor.unload()

            if caption_deepbooru:
                db_processor = BooruProcessor(booru_min_score)
                db_tags = db_processor.process(src_images)
                db_processor.unload()

            captions = build_captions(src_files, captions, wd_tags, db_tags)

        if upscale:
            print(f"Using scaler: {scaler}")
            if scaler >= len(shared.sd_upscalers):
                up_processor = UpscaleProcessor()
                src_images = up_processor.process(src_images, short_caps)
            else:
                up_processor = SystemUpscaleProcessor(scaler, upscale_ratio)
                src_images = up_processor.process(src_images)

        save_images(src_images, src_files, captions)
        shared.cmd_opts.disable_safe_unpickle = was_safe_unpickle

        try:
            shared.sd_model.to(shared.device)
        except:
            pass
        return f"Successfully processed {len(files)} images.", f"Successfully processed {len(files)} images."

    except Exception as e:
        msg = f"Exception processing: {e}"
        traceback.print_exc()
        pass

    return msg, msg
