import os
import sys
from importlib import import_module

import gradio as gr

from extensions.sd_smartprocess import smartprocess
from extensions.sd_smartprocess.interrogators.interrogator import InterrogatorRegistry
from extensions.sd_smartprocess.process_params import ProcessParams
from extensions.sd_smartprocess.smartprocess import is_image, get_backup_path
from modules import script_callbacks, shared
from modules.call_queue import wrap_gradio_gpu_call, wrap_gradio_call
from modules.ui import setup_progressbar
from modules.upscaler import Upscaler

refresh_symbol = "\U0001f504"  # 🔄
delete_symbol = "\U0001F5D1"  # 🗑️

# This holds all the inputs for the Smart Process tab
all_inputs = []
# Input keys
input_keys = []

# This one holds all files loaded from user path, shown in the first gallery
all_files = []

# This one holds the files for the secondary gallery
selected_files = []

# This holds the filtered content of the gallery
filtered_files = []

# Captions for the first gallery
all_captions = []
# Captions for the second gallery
selected_captions = []
# This holds the files that will be processed
process_targets = []

# This holds the accordions for the captioners
accordion_keys = []
captioner_accordions = []

# Sort all the tags by frequency
tags_dict = {}

# Tags to include when filtering
filter_tags_include = []
# Tags to exclude when filtering
filter_tags_exclude = []

# Only include tags that contain these strings
filter_string_include = []
# Exclude tags that contain these strings
filter_string_exclude = []

current_image = None
current_caption = None

# The list of processors to run
registry = InterrogatorRegistry()

int_dict = registry.list_interrogators()
print(f"Found {len(int_dict.keys())} interrogators: {int_dict}")
natural_captioner_names = ["CLIP", "BLIP", "LLAVA"]
default_captioners = ["Swin"]

natural_captioners = {}
tag_captioners = {}

for interrogator, params in int_dict.items():
    enable = False
    display_name = interrogator.replace("Interrogator", "")
    if display_name in default_captioners:
        enable = True
    if display_name in natural_captioner_names:
        natural_captioners[display_name] = enable
    else:
        tag_captioners[display_name] = enable

wolf_captioners = ["Moat", "Swin", "Conv", "Conv2", "Vit"]


def list_scalers():
    scaler_dir = os.path.join(shared.script_path, "extensions", "sd_smartprocess", "upscalers")
    scalers = []
    for file in os.listdir(scaler_dir):
        if file.endswith("_model.py"):
            # Construct module path for import
            module_name = f"extensions.sd_smartprocess.upscalers.{file.replace('.py', '')}"
            imported_module = import_module(module_name)

            # Add the directory to sys.path to ensure relative imports in the module work
            module_dir = os.path.dirname(os.path.join(scaler_dir, file))
            if module_dir not in sys.path:
                sys.path.append(module_dir)

            for name, obj in vars(imported_module).items():
                # Check if the object is a class and a subclass of Upscaler, and has the scaler_data classmethod
                if isinstance(obj, type) and issubclass(obj, Upscaler) and hasattr(obj, 'scaler_data'):
                    scaler_data = obj.scaler_data()
                    scalers.append(scaler_data)

    system_scalers = shared.sd_upscalers

    for scaler in scalers:
        if scaler.name not in [x.name for x in system_scalers]:
            system_scalers.append(scaler)


def generate_caption_section():
    global captioner_accordions
    wolf_params = {}
    for gator, gator_params in int_dict.items():
        accordion_name = gator.replace("Interrogator", "")

        if accordion_name in wolf_captioners:
            wolf_params = gator_params
            continue
        show_accordion = False
        if gator in natural_captioners.keys():
            show_accordion = natural_captioners.get(gator, False)
        elif gator in tag_captioners.keys():
            show_accordion = tag_captioners.get(gator, False)
        if len(gator_params.keys()) == 0:
            continue
        print(f"Adding {accordion_name} to captioners")
        with gr.Accordion(label=accordion_name, open=False, visible=show_accordion) as temp_accordion:
            with gr.Column():
                for param, param_value in gator_params.items():
                    temp_element = None
                    label = param.replace("_", " ").title()
                    if isinstance(param_value, bool):
                        temp_element = gr.Checkbox(label=label, value=param_value, interactive=True)
                    elif isinstance(param_value, int):
                        temp_element = gr.Slider(label=label, value=param_value, step=1, minimum=0, maximum=100,
                                                 interactive=True)
                    elif isinstance(param_value, float):
                        temp_element = gr.Slider(label=label, value=param_value, step=0.01, minimum=0, maximum=1,
                                                 interactive=True)
                    elif isinstance(param_value, str):
                        if param == "group":
                            continue
                        temp_element = gr.Textbox(label=label, value=param_value, interactive=True)
                    if temp_element:
                        all_inputs.append(temp_element)
                        input_keys.append(param)
                        setattr(temp_element, "do_not_save_to_config", True)
        captioner_accordions.append(temp_accordion)
        accordion_keys.append(accordion_name)
    show_accordion = False
    for wc in wolf_captioners:
        if wc in default_captioners:
            show_accordion = True
    accordion_name = "Wolf"
    print(f"Adding {accordion_name} to captioners")

    with gr.Accordion(label=accordion_name, open=False, visible=show_accordion) as temp_accordion:
        with gr.Column():
            for param, param_value in wolf_params.items():
                temp_element = None
                label = param.replace("_", " ").title()
                if isinstance(param_value, bool):
                    temp_element = gr.Checkbox(label=label, value=param_value, interactive=True)
                elif isinstance(param_value, int):
                    temp_element = gr.Slider(label=label, value=param_value, step=1, minimum=0, maximum=100,
                                             interactive=True)
                elif isinstance(param_value, float):
                    temp_element = gr.Slider(label=label, value=param_value, step=0.01, minimum=0, maximum=1,
                                             interactive=True)
                elif isinstance(param_value, str):
                    if param == "group":
                        continue
                    temp_element = gr.Textbox(label=label, value=param_value, interactive=True)
                if temp_element:
                    all_inputs.append(temp_element)
                    input_keys.append(param)
                    setattr(temp_element, "do_not_save_to_config", True)
    captioner_accordions.append(temp_accordion)
    accordion_keys.append(accordion_name)


def sort_tags_dict():
    global tags_dict
    global filter_tags_include
    global filter_tags_exclude
    # Sort the tags into a list by frequency
    sorted_tags = sorted(tags_dict.items(), key=lambda x: x[1], reverse=True)
    tag_options = []
    for tag in sorted_tags:
        if tag[0] in filter_tags_include or tag[0] in filter_tags_exclude:
            continue
        tag_name = tag[0]
        tag_label = f"{tag_name} ({tag[1]})"
        tag_options.append(tag_label)
    return tag_options


def create_process_ui():
    """Create the UI for the Smart Process tab"""
    global tag_captioners
    global natural_captioners
    global all_inputs
    global input_keys
    global captioner_accordions
    global filter_tags_include
    global filter_tags_exclude
    global filter_string_include
    global filter_string_exclude
    list_scalers()
    with gr.Blocks() as sp_interface:
        with gr.Row(equal_height=True):
            with gr.Column():
                with gr.Row():
                    sp_src = gr.Textbox(label='Source directory', elem_id="sp_src")
                    sp_load_src = gr.Button(value=refresh_symbol, size="sm", variant='primary', elem_id="sp_load_src")
                    sp_clear_src = gr.Button(value=delete_symbol, size="sm", variant='secondary',
                                             elem_id="sp_clear_src")
            with gr.Column():
                with gr.Row(elem_id="sp_top_btn_row"):
                    sp_process_selected = gr.Button(value="Process Selected")
                    sp_cancel = gr.Button(value="Cancel", visible=False)
                    sp_process_all = gr.Button(value="Process All", variant='primary')

        with gr.Column():
            with gr.Row():
                with gr.Accordion(label="Filtering", open=False):
                    with gr.Row():
                        with gr.Column():
                            sp_filter = gr.Textbox(label="Filter", value="")
                        with gr.Column():
                            with gr.Row():
                                sp_filter_include = gr.Button(value="Include", elem_classes=["inc_exc_btn"])
                                sp_filter_exclude = gr.Button(value="Exclude", elem_classes=["inc_exc_btn"])
                    with gr.Row():
                        with gr.Column():
                            sp_tag_dropdown = gr.Dropdown(label="Tags", choices=sort_tags_dict(), value="")
                        with gr.Column():
                            with gr.Row():
                                sp_tag_include = gr.Button(value="Include", elem_classes=["inc_exc_btn"])
                                sp_tag_exclude = gr.Button(value="Exclude", elem_classes=["inc_exc_btn"])
                    with gr.Row(variant="panel"):
                        with gr.Row():
                            with gr.Column():
                                sp_filter_string_include_group = gr.CheckboxGroup(label="Included Strings",
                                                                                  choices=filter_string_include,
                                                                                  value=filter_string_include,
                                                                                  interactive=True)
                            with gr.Column():
                                sp_filter_string_exclude_group = gr.CheckboxGroup(label="Excluded Strings",
                                                                                  choices=filter_string_exclude,
                                                                                  value=filter_string_exclude,
                                                                                  interactive=True)

                        with gr.Row():
                            with gr.Column():
                                sp_include_tag_group = gr.CheckboxGroup(label="Included Tags",
                                                                        choices=filter_tags_include,
                                                                        value=filter_tags_include, interactive=True)
                            with gr.Column():
                                sp_exclude_tag_group = gr.CheckboxGroup(label="Excluded Tags",
                                                                        choices=filter_tags_exclude,
                                                                        value=filter_tags_exclude, interactive=True)

        with gr.Row(equal_height=False):
            with gr.Column(variant="panel", scale=1):
                with gr.Tab("Pre-Process") as sp_crop_tab:
                    sp_size = gr.Slider(minimum=64, maximum=4096, step=64, label="Output Size", value=1024)
                    with gr.Row():
                        sp_pad = gr.Checkbox(label="Pad Images", elem_classes=["short_check"])
                        sp_crop = gr.Checkbox(label="Crop Images", elem_classes=["short_check"])
                        sp_do_rename = gr.Checkbox(label="Rename Images", elem_classes=["short_check"])
                    sp_crop_mode = gr.Dropdown(label='Crop Mode', value="smart",
                                               choices=["smart", "square", "empty", "contain"],
                                               visible=False)

                    def toggle_mode_dropdown(evt: gr.SelectData):
                        return gr.update(visible=evt.selected)

                    sp_crop.select(
                        fn=toggle_mode_dropdown,
                        outputs=[sp_crop_mode]
                    )

                with gr.Tab("Captions") as sp_cap_tab:
                    sp_caption = gr.Checkbox(label='Generate Captions')
                    with gr.Column(visible=False) as cap_settings:
                        sp_nl_captioners = gr.CheckboxGroup(label='Natural Language Captioners',
                                                            choices=natural_captioners.keys(),
                                                            value=[x for x in natural_captioners.keys() if
                                                                   natural_captioners[x]],
                                                            interactive=True)
                        sp_captioners = gr.CheckboxGroup(label='Captioners',
                                                         choices=tag_captioners.keys(),
                                                         value=[x for x in tag_captioners.keys() if tag_captioners[x]],
                                                         interactive=True)
                        sp_txt_action = gr.Dropdown(label='Existing Caption Action', value="ignore",
                                                    choices=["ignore", "copy", "prepend", "append"])
                        generate_caption_section()
                        sp_tags_to_ignore = gr.Textbox(label="Tags To Ignore", value="")
                        sp_replace_class = gr.Checkbox(label='Replace Class with Subject in Caption', value=False)
                        sp_class = gr.Textbox(label='Subject Class', placeholder='Subject class to crop (leave '
                                                                                 'blank to auto-detect)')
                        sp_subject = gr.Textbox(label='Subject Name', placeholder='Subject Name to replace class '
                                                                                  'with in captions')

                        def update_caption_groups(evt: gr.SelectData):
                            global tag_captioners
                            global natural_captioners
                            if evt.value in natural_captioners.keys():
                                natural_captioners[evt.value] = evt.selected
                            elif evt.value in tag_captioners.keys():
                                tag_captioners[evt.value] = evt.selected
                            outputs = []
                            show_wolf = False
                            for cap_name, accordion in zip(accordion_keys, captioner_accordions):
                                if cap_name in tag_captioners.keys():
                                    outputs.append(gr.update(visible=tag_captioners.get(cap_name, False)))
                                elif cap_name in natural_captioners.keys():
                                    outputs.append(gr.update(visible=natural_captioners.get(cap_name, False)))
                            for wc in wolf_captioners:
                                if tag_captioners.get(wc, False):
                                    show_wolf = True
                            outputs.append(gr.update(visible=show_wolf))
                            return outputs

                        sp_captioners.select(
                            fn=update_caption_groups,
                            outputs=captioner_accordions
                        )

                        sp_nl_captioners.select(
                            fn=update_caption_groups,
                            outputs=captioner_accordions
                        )

                    def toggle_cap_settings(evt: gr.SelectData):
                        return gr.update(visible=evt.selected)

                    sp_caption.select(
                        fn=toggle_cap_settings,
                        outputs=[cap_settings]
                    )

                with gr.Tab("Post-Process") as sp_post_tab:
                    sp_restore_faces = gr.Checkbox(label='Restore Faces', value=False)
                    sp_face_model = gr.Dropdown(label="Face Restore Model", choices=["GFPGAN", "Codeformer"],
                                                value="GFPGAN")
                    sp_upscale = gr.Checkbox(label='Upscale and Resize', value=False)
                    with gr.Column(visible=False) as sp_upscale_settings:
                        sp_upscale_mode = gr.Radio(label="Upscale Mode", choices=["Size", "Ratio"], value="Size")
                        sp_upscale_ratio = gr.Slider(label="Upscale Ratio", value=2, step=1, minimum=2, maximum=8,
                                                     visible=False)
                        sp_upscale_size = gr.Slider(label="Upscale Size", value=1024, step=64, minimum=512,
                                                    maximum=8192)
                        sp_upscaler_1 = gr.Dropdown(label='Upscaler', elem_id="sp_scaler_1",
                                                    choices=[x.name for x in shared.sd_upscalers],
                                                    value=shared.sd_upscalers[0].name, type="index")
                        sp_upscaler_2 = gr.Dropdown(label='Upscaler', elem_id="sp_scaler_2",
                                                    choices=[x.name for x in shared.sd_upscalers],
                                                    value=shared.sd_upscalers[0].name, type="index")

                        def toggle_upscale_mode(evt: gr.SelectData):
                            return gr.update(visible=evt.value == "Ratio"), gr.update(visible=evt.value == "Size")

                        def toggle_upscale_settings(evt: gr.SelectData):
                            return gr.update(visible=evt.selected)

                        sp_upscale_mode.select(
                            fn=toggle_upscale_mode,
                            outputs=[sp_upscale_ratio, sp_upscale_size]
                        )

                        sp_upscale.select(
                            fn=toggle_upscale_settings,
                            outputs=[sp_upscale_settings]
                        )

            # Preview/progress
            with gr.Column(variant="panel", scale=1):
                sp_gallery_all = gr.Gallery(label='All', elem_id='sp_gallery', rows=2, columns=4,
                                            allow_preview=False)
                with gr.Row():
                    sp_add_all = gr.Button(value="Add All", size="sm")
                    sp_add_selected = gr.Button(value="Add Selected", size="sm")
                    sp_remove_selected = gr.Button(value="Remove Selected", size="sm")
                    sp_clear_selected = gr.Button(value="Remove All", size="sm")
                sp_gallery_selected = gr.Gallery(label='Selected', elem_id='sp_selected', rows=2, columns=4,
                                                 allow_preview=False)

            with gr.Column(variant="panel", scale=1):
                with gr.Row(elem_id="sp_file_options"):
                    sp_do_backup = gr.Checkbox(label="Backup Files", elem_classes=["short_check"])
                    sp_auto_save = gr.Checkbox(label="Auto-Save", elem_classes=["short_check"])
                with gr.Row() as sp_crop_row:
                    sp_pre_current = gr.Button(value="PreProcess Current", size="sm")
                    sp_save_current_pre = gr.Button(value="Save Pre", size="sm")
                with gr.Row(visible=False) as sp_cap_row:
                    with gr.Column():
                        with gr.Row():
                            sp_current_caption = gr.Textbox(label="Caption", value="", lines=2)
                        with gr.Row():
                            sp_caption_current = gr.Button(value="Caption Current", size="sm")
                            sp_save_current_caption = gr.Button(value="Save Caption", size="sm")
                with gr.Row(visible=False) as sp_post_row:
                    sp_post_current = gr.Button(value="PostProcess Current", size="sm")
                    sp_save_current_post = gr.Button(value="Save Post", size="sm")
                sp_progress = gr.HTML(elem_id="sp_progress", value="")
                sp_outcome = gr.HTML(elem_id="sp_error", value="")
                sp_progressbar = gr.HTML(elem_id="sp_progressbar")
                sp_preview = gr.Image(label='Preview', elem_id='sp_preview', interactive=False)
                setup_progressbar(sp_progressbar, sp_preview, 'sp', textinfo=sp_progress)

        def toggle_group_buttons(gd: gr.SelectData):
            value = gd.value
            selected = gd.selected
            show_cap = False
            show_pre = False
            show_post = False
            if value == "Pre-Process":
                show_pre = selected
            elif value == "Captions":
                show_cap = selected
            elif value == "Post-Process":
                show_post = selected
            return gr.update(visible=show_pre), gr.update(visible=show_cap), gr.update(visible=show_post)

        for elem in [sp_crop_tab, sp_cap_tab, sp_post_tab]:
            elem.select(
                fn=toggle_group_buttons,
                outputs=[sp_crop_row, sp_cap_row, sp_post_row]
            )

        def get_current_image():
            files = []
            if current_image:
                # If current image is a tuple, select the first element
                if isinstance(current_image, tuple):
                    image_path = current_image[0]
                else:
                    image_path = current_image
                files.append(image_path)
            return files

        def pre_process_current(*args):
            params = params_to_dict(*args)
            params.src_files = get_current_image()
            params.pre_only()
            params.save_image = False
            return smartprocess.do_process(params)

        def caption_current(*args):
            cap_params = params_to_dict(*args)
            cap_params.src_files = get_current_image()
            cap_params.save_caption = False
            cap_params.cap_only()
            outputs, caption_dict, msg = start_caption(cap_params)
            caption = ""
            # Get the first key from the caption dict
            if len(caption_dict) > 0:
                captions = list(caption_dict.values())
                caption = captions[0]
            return gr.update(visible=False), gr.update(value=caption), gr.update(value=msg)

        def post_process_current(*args):
            post_params = params_to_dict(*args)
            post_params.src_files = get_current_image()
            post_params.post_only()
            post_params.save_image = False
            return smartprocess.do_process(post_params)

        def process_selected(*args):
            global selected_files
            params = params_to_dict(*args)
            params.src_files = selected_files
            return smartprocess.do_process(params)

        def process_all(*args):
            global filtered_files
            params = params_to_dict(*args)
            params.src_files = filtered_files
            return smartprocess.do_process(params)

        def start_caption(params: ProcessParams):
            params.captioners = tag_captioners
            params.crop = False
            params.pad = False
            params.scale = False
            params.restore_faces = False
            params.caption = True
            if len(params.src_files) > 0:
                # Make sure files in src_files are not tuples
                files = []
                for file in params.src_files:
                    if isinstance(file, tuple):
                        files.append(file[0])
                    else:
                        files.append(file)
                params.src_files = files
                outputs, caption_dict, msg = smartprocess.do_process(params)
                return outputs, caption_dict, msg

        def start_preprocess(params: ProcessParams):
            params.captioners = {}
            params.scale = False
            params.restore_faces = False
            if len(params.src_files) > 0:
                # Make sure files in src_files are not tuples
                files = []
                for file in params.src_files:
                    if isinstance(file, tuple):
                        files.append(file[0])
                    else:
                        files.append(file)
                outputs, caption_dict, msg = smartprocess.do_process(params)
                return outputs, caption_dict, msg

        def start_postprocess(params: ProcessParams):
            params.captioners = {}
            params.crop = False
            params.pad = False
            params.caption = False
            if len(params.src_files) > 0:
                # Make sure files in src_files are not tuples
                files = []
                for file in params.src_files:
                    if isinstance(file, tuple):
                        files.append(file[0])
                    else:
                        files.append(file)
                outputs, caption_dict, msg = smartprocess.do_process(params)
                return outputs, caption_dict, msg

        def process_tags(captions):
            global tags_dict
            tags_dict = {}
            for caption in captions:
                tags = caption.split(",")
                for tag in tags:
                    tag = tag.strip()
                    if tag not in tags_dict.keys():
                        tags_dict[tag] = 1
                    else:
                        tags_dict[tag] += 1

        def clear_globals():
            global all_files
            global all_captions
            global selected_files
            global selected_captions
            global filter_tags_include
            global filter_tags_exclude
            global filter_string_include
            global filter_string_exclude
            global current_image
            global current_caption

            all_files = []
            all_captions = []
            selected_files = []
            selected_captions = []
            filter_tags_include = []
            filter_tags_exclude = []
            filter_string_include = []
            filter_string_exclude = []
            current_image = None
            current_caption = None

        def load_src(src_path):
            # Enumerate all files recursively in src_path, and if they're an image, add them to the gallery
            global all_files
            global all_captions
            clear_globals()
            for root, dirs, files in os.walk(src_path):
                for file in files:
                    filename = os.path.join(root, file)
                    if is_image(filename):
                        existing_caption_txt_filename = os.path.splitext(filename)[0] + '.txt'
                        if os.path.exists(existing_caption_txt_filename):
                            with open(existing_caption_txt_filename, 'r', encoding="utf8") as f:
                                existing_caption_txt = f.read()
                        else:
                            file = os.path.splitext(file)[0]
                            existing_caption_txt = ''.join(c for c in file if c.isalpha() or c in [" ", ", "])
                        all_captions.append(existing_caption_txt)
                        all_files.append((filename, existing_caption_txt))
            process_tags(all_captions)
            tag_group = gr.update(choices=sort_tags_dict(), value="")

            print(f"Found {len(all_files)} images in {src_path}")
            outputs = [gr.update(value=all_files), tag_group]
            # Enumerate all elements in elements_to_clear except for the first two
            for _ in elements_to_clear[2:]:
                outputs.append(gr.update(value=None))
            return outputs

        def set_all_current(evt: gr.SelectData):
            global current_image
            if evt.selected:
                files = filter_gallery(True)
                current_image = files[evt.index]
                return gr.update(value=current_image[0]), gr.update(value=current_image[1])
            else:
                return gr.update(value=None), gr.update(value=None)

        def set_selected_current(evt: gr.SelectData):
            global current_image
            if evt.selected:
                files = filter_gallery(False)
                current_image = files[evt.index]
                return gr.update(value=current_image[0]), gr.update(value=current_image[1])
            else:
                return gr.update(value=None), gr.update(value=None)

        def add_selected():
            global selected_files
            global current_image
            if current_image and current_image not in selected_files:
                selected_files.append(current_image)
            return gr.update(value=selected_files)

        def remove_selected():
            global selected_files
            global current_image
            if current_image and current_image in selected_files:
                selected_files.remove(current_image)
            return gr.update(value=selected_files)

        def clear_selected():
            global selected_files
            global current_image
            current_image = None
            selected_files = []
            return gr.update(value=selected_files)

        def select_all():
            global selected_files
            files = filter_gallery(True)
            for file in files:
                if file not in selected_files:
                    selected_files.append(file)
            return gr.update(value=files)

        def params_to_dict(*args):
            global input_keys
            pp = list(args)
            params_dict = dict(zip(input_keys, pp))
            pp = ProcessParams().from_dict(params_dict)
            return pp

        import re

        def filter_gallery(use_all_files: bool):
            """
            Filters a collection of files based on specified inclusion and exclusion criteria for tags and filter strings.

            The function filters files based on tags extracted from file captions and matches these tags against specified
            inclusion and exclusion criteria. The criteria can be a set of plain tags, wildcard patterns, or regex expressions.

            Parameters:
            - use_all_files (bool): Determines whether to filter from all files or only selected files.
              If True, filters from all files; otherwise, filters from selected files.

            Globals:
            - filter_tags_include (list): Tags required to include a file.
            - filter_tags_exclude (list): Tags that lead to exclusion of a file.
            - filter_string_include (list): Filter strings for inclusion; supports wildcards and regex.
            - filter_string_exclude (list): Filter strings for exclusion; supports wildcards and regex.
            - all_files (list): List of all files, where each file is a tuple (filename, caption).
            - selected_files (list): List of selected files, where each file is a tuple (filename, caption).

            Returns:
            - filtered_files (list): List of files filtered based on the specified criteria.

            Examples:
            1. Plain Tag Matching:
               - To include files with a tag 'holiday', add 'holiday' to filter_tags_include.
               - To exclude files with a tag 'work', add 'work' to filter_tags_exclude.

            2. Wildcard Patterns:
               - To include files with tags starting with 'trip', add 'trip*' to filter_string_include.
               - To exclude files with tags ending with '2023', add '*2023' to filter_string_exclude.

            3. Regex Expressions:
               - To include files with tags that have any number followed by 'days', add '\\d+days' to filter_string_include.
               - To exclude files with tags formatted like dates (e.g., '2023-04-01'), add '\\d{4}-\\d{2}-\\d{2}' to filter_string_exclude.

            Note:
            - The function treats tags as case-sensitive.
            - Wildcard '*' matches any sequence of characters (including none).
            - Regex patterns should follow Python's 're' module syntax.
            """
            global filter_tags_include, filter_tags_exclude
            global filter_string_include, filter_string_exclude
            global all_files, selected_files, tags_dict

            def matches_pattern(pattern, string):
                # Convert wildcard to regex if necessary
                if '*' in pattern:
                    pattern = '^' + pattern.replace('*', '.*') + '$'
                    return re.match(pattern, string) is not None
                else:
                    if " " not in pattern:
                        parts = string.split(" ")
                        return pattern in parts
                    else:
                        return pattern in string

            def should_include(tag, filter_tags, filter_strings):
                if len(filter_tags) == 0 and len(filter_strings) == 0:
                    return True
                tag_match = False
                filter_match = False
                if tag in filter_tags and len(filter_tags) > 0:
                    tag_match = True
                if len(filter_strings) > 0:
                    for filter_string in filter_strings:
                        if matches_pattern(filter_string, tag):
                            filter_match = True
                            break
                return tag_match or filter_match

            def should_exclude(tag, filter_tags, filter_strings):
                if tag in filter_tags:
                    return True
                for filter_string in filter_strings:
                    if matches_pattern(filter_string, tag):
                        return True
                return False

            filtered = []
            files = all_files if use_all_files else selected_files

            for file in files:
                caption = file[1]
                tags = [tag.strip() for tag in caption.split(",")]
                out_tags = []
                for tag in tags:
                    include = True
                    if should_exclude(tag, filter_tags_exclude, filter_string_exclude):
                        include = False
                    elif not should_include(tag, filter_tags_include, filter_string_include):
                        include = False
                    if include:
                        out_tags.append(tag)
                if len(out_tags) > 0:
                    filtered.append(file)

            return filtered

        def include_tag(tag_dropdown_value):
            global filter_tags_include
            tag_dropdown_value = tag_dropdown_value.split(" (")[0]
            if tag_dropdown_value not in filter_tags_include:
                filter_tags_include.append(tag_dropdown_value)
            # Sort filter_tags_include by frequency, then alphabetically, ensuring we only add the values in filter_tags_include
            filter_tags_include = [x[0] for x in sorted(tags_dict.items(), key=lambda x: (-x[1], x[0])) if
                                   x[0] in filter_tags_include]

            tag_group = gr.update(choices=sort_tags_dict(), value="")
            all_gallery_items = filter_gallery(True)
            selected_gallery_items = filter_gallery(False)
            return tag_group, gr.update(value=filter_tags_include, choices=filter_tags_include), gr.update(
                value=all_gallery_items), gr.update(value=selected_gallery_items)

        def exclude_tag(tag_dropdown_value):
            global filter_tags_exclude
            tag_dropdown_value = tag_dropdown_value.split(" (")[0]
            if tag_dropdown_value not in filter_tags_exclude:
                filter_tags_exclude.append(tag_dropdown_value)
            # Sort filter_tags_exclude by frequency, then alphabetically
            filter_tags_exclude = [x[0] for x in sorted(tags_dict.items(), key=lambda x: (-x[1], x[0])) if
                                   x[0] in filter_tags_exclude]
            tag_group = gr.update(choices=sort_tags_dict(), value="")
            all_gallery_items = filter_gallery(True)
            selected_gallery_items = filter_gallery(False)
            return tag_group, gr.update(value=filter_tags_exclude, choices=filter_tags_exclude), gr.update(
                value=all_gallery_items), gr.update(value=selected_gallery_items)

        def remove_include_tag(select_data: gr.SelectData):
            global filter_tags_include
            if select_data.selected:
                filter_tags_include.append(select_data.value)
            else:
                filter_tags_include.remove(select_data.value)
            filter_tags_include = [x[0] for x in sorted(tags_dict.items(), key=lambda x: (-x[1], x[0])) if
                                   x[0] in filter_tags_include]
            all_gallery_items = filter_gallery(True)
            selected_gallery_items = filter_gallery(False)
            return gr.update(choices=sort_tags_dict(), value=""), gr.update(value=filter_tags_include,
                                                                            choices=filter_tags_include), gr.update(
                value=all_gallery_items), gr.update(value=selected_gallery_items)

        def remove_exclude_tag(select_data: gr.SelectData):
            global filter_tags_exclude
            if select_data.selected:
                filter_tags_exclude.append(select_data.value)
            else:
                filter_tags_exclude.remove(select_data.value)
            # Sort filter tags alphabetically
            filter_tags_exclude = [x[0] for x in sorted(tags_dict.items(), key=lambda x: (-x[1], x[0])) if
                                   x[0] in filter_tags_exclude]
            all_gallery_items = filter_gallery(True)
            selected_gallery_items = filter_gallery(False)
            return gr.update(choices=sort_tags_dict(), value=""), gr.update(value=filter_tags_exclude,
                                                                            choices=filter_tags_exclude), gr.update(
                value=all_gallery_items), gr.update(value=selected_gallery_items)

        def include_string(string_group_value):
            global filter_string_include
            if string_group_value not in filter_string_include:
                filter_string_include.append(string_group_value)
            # Sort filter_string_include alphabetically
            filter_string_include = sorted(filter_string_include)
            return gr.update(value=filter_string_include, choices=filter_string_include), gr.update(
                value=filter_gallery(True)), gr.update(value=filter_gallery(False)), gr.update(value="")

        def exclude_string(string_group_value):
            global filter_string_exclude
            if string_group_value not in filter_string_exclude:
                filter_string_exclude.append(string_group_value)
            # Sort filter_string_exclude alphabetically
            filter_string_exclude = sorted(filter_string_exclude)
            return gr.update(value=filter_string_exclude, choices=filter_string_exclude), gr.update(
                value=filter_gallery(True)), gr.update(value=filter_gallery(False)), gr.update(value="")

        def remove_include_string(select_data: gr.SelectData):
            global filter_string_include
            if select_data.selected:
                filter_string_include.append(select_data.value)
            else:
                filter_string_include.remove(select_data.value)
            filter_string_include = sorted(filter_string_include)
            return gr.update(value=filter_string_include, choices=filter_string_include), gr.update(
                value=filter_gallery(True)), gr.update(value=filter_gallery(False)), gr.update(value="")

        def remove_exclude_string(select_data: gr.SelectData):
            global filter_string_exclude
            if select_data.selected:
                filter_string_exclude.append(select_data.value)
            else:
                filter_string_exclude.remove(select_data.value)
            filter_string_exclude = sorted(filter_string_exclude)
            return gr.update(value=filter_string_exclude, choices=filter_string_exclude), gr.update(
                value=filter_gallery(True)), gr.update(value=filter_gallery(False)), gr.update(value="")

        def save_current_caption(caption, do_backup=True):
            temp_params = ProcessParams()
            temp_params.do_backup = do_backup
            global current_image
            if current_image:
                if isinstance(current_image, tuple):
                    image_path = current_image[0]
                else:
                    image_path = current_image
                caption_txt_filename = os.path.splitext(image_path)[0] + '.txt'
                caption_txt_filename = get_backup_path(caption_txt_filename, temp_params)
                with open(caption_txt_filename, 'w', encoding="utf8") as f:
                    f.write(caption)
                return gr.update(value="Caption saved to " + caption_txt_filename)
            else:
                return gr.update(value="No image selected")

        def save_current_image(do_backup, do_rename):
            temp_params = ProcessParams()
            temp_params.do_backup = do_backup
            temp_params.do_rename = do_rename
            temp_params.save_image = True
            global current_image
            if current_image:
                if isinstance(current_image, tuple):
                    image_path = current_image[0]
                else:
                    image_path = current_image
                save_path, backup_path = get_backup_path(image_path, temp_params)

        all_inputs = [
            sp_src,
            sp_clear_src,
            sp_do_rename,
            sp_pad,
            sp_crop,
            sp_crop_mode,
            sp_size,
            sp_txt_action,
            sp_caption,
            sp_captioners,
            sp_tags_to_ignore,
            sp_class,
            sp_subject,
            sp_replace_class,
            sp_restore_faces,
            sp_face_model,
            sp_upscale,
            sp_upscale_ratio,
            sp_upscaler_1,
            sp_upscaler_2,
            sp_do_backup,
            sp_auto_save
        ]

        for element in all_inputs:
            setattr(element, "do_not_save_to_config", True)
            var_name = [var_name for var_name, var in locals().items() if var is element]
            input_keys.append(var_name[0])

        sp_tag_include.click(
            fn=include_tag,
            inputs=[sp_tag_dropdown],
            outputs=[sp_tag_dropdown, sp_include_tag_group, sp_gallery_all, sp_gallery_selected]
        )

        sp_tag_exclude.click(
            fn=exclude_tag,
            inputs=[sp_tag_dropdown],
            outputs=[sp_tag_dropdown, sp_exclude_tag_group, sp_gallery_all, sp_gallery_selected]
        )

        sp_exclude_tag_group.select(
            fn=remove_exclude_tag,
            outputs=[sp_tag_dropdown, sp_exclude_tag_group, sp_gallery_all, sp_gallery_selected]
        )

        sp_include_tag_group.select(
            fn=remove_include_tag,
            outputs=[sp_tag_dropdown, sp_include_tag_group, sp_gallery_all, sp_gallery_selected]
        )

        sp_filter_string_include_group.select(
            fn=remove_include_string,
            outputs=[sp_filter_string_include_group, sp_gallery_all, sp_gallery_selected, sp_filter]
        )

        sp_filter_string_exclude_group.select(
            fn=remove_exclude_string,
            outputs=[sp_filter_string_exclude_group, sp_gallery_all, sp_gallery_selected, sp_filter]
        )

        sp_filter_include.click(
            fn=include_string,
            inputs=[sp_filter],
            outputs=[sp_filter_string_include_group, sp_gallery_all, sp_gallery_selected, sp_filter]
        )

        sp_filter_exclude.click(
            fn=exclude_string,
            inputs=[sp_filter],
            outputs=[sp_filter_string_exclude_group, sp_gallery_all, sp_gallery_selected, sp_filter]
        )

        sp_pre_current.click(
            fn=wrap_gradio_gpu_call(pre_process_current, extra_outputs=[gr.update()]),
            _js="start_smart_process",
            inputs=all_inputs,
            outputs=[sp_progress, sp_outcome, sp_preview]
        )

        sp_save_current_pre.click(
            fn=save_current_image,
            inputs=[sp_do_backup, sp_do_rename],
            outputs=[sp_outcome]
        )

        sp_caption_current.click(
            fn=wrap_gradio_call(caption_current, extra_outputs=None, add_stats=False),
            _js="start_smart_process",
            inputs=all_inputs,
            outputs=[sp_progress, sp_current_caption, sp_outcome]
        )

        sp_save_current_caption.click(
            fn=save_current_caption,
            inputs=[sp_current_caption, sp_do_backup],
            outputs=[sp_outcome]
        )

        sp_post_current.click(
            fn=wrap_gradio_gpu_call(post_process_current, extra_outputs=[gr.update()]),
            _js="start_smart_process",
            inputs=all_inputs,
            outputs=[sp_progress, sp_outcome, sp_preview]
        )

        sp_save_current_post.click(
            fn=save_current_image,
            inputs=[sp_do_backup, sp_do_rename],
            outputs=[sp_outcome]
        )

        sp_process_selected.click(
            fn=wrap_gradio_gpu_call(process_selected, extra_outputs=[gr.update()]),
            _js="start_smart_process",
            inputs=all_inputs,
            outputs=[sp_current_caption]
        )

        sp_process_all.click(
            fn=wrap_gradio_gpu_call(process_all, extra_outputs=[gr.update()]),
            _js="start_smart_process",
            inputs=all_inputs,
            outputs=[
                sp_progress,
                sp_outcome
            ],
        )

        sp_cancel.click(
            fn=lambda: shared.state.interrupt()
        )

        sp_gallery_all.select(fn=set_all_current, outputs=[sp_preview, sp_current_caption])
        sp_gallery_selected.select(fn=set_selected_current, outputs=[sp_preview, sp_current_caption])

        elements_to_clear = [sp_gallery_all,
                             sp_tag_dropdown,
                             sp_gallery_selected,
                             sp_filter,
                             sp_current_caption,
                             sp_preview,
                             sp_filter_string_include_group,
                             sp_filter_string_exclude_group,
                             sp_include_tag_group,
                             sp_exclude_tag_group,
                             ]

        elements_to_load = [sp_gallery_all, sp_tag_dropdown, *elements_to_clear[2:]]
        sp_load_src.click(fn=load_src, inputs=[sp_src],
                          outputs=elements_to_clear)

        def clear_elements():
            clear_globals()
            outputs = []
            for _ in elements_to_clear:
                # If the element has a choices attribute, set it to None
                if isinstance(_, gr.Dropdown) or isinstance(_, gr.CheckboxGroup):
                    print(f"Clearing {_}")
                    outputs.append(gr.update(choices=[], value=None))
                else:
                    print(f"Element type is {type(_)}")
                    outputs.append(gr.update(value=None))
            return outputs

        sp_clear_src.click(fn=clear_elements, outputs=elements_to_clear)

        sp_add_all.click(
            fn=select_all,
            outputs=[sp_gallery_selected]
        )

        sp_add_selected.click(
            fn=add_selected,
            inputs=[],
            outputs=[sp_gallery_selected]
        )

        sp_remove_selected.click(
            fn=remove_selected,
            inputs=[],
            outputs=[sp_gallery_selected]
        )

        sp_clear_selected.click(
            fn=clear_selected,
            outputs=[sp_gallery_selected]
        )

    return (sp_interface, "SmartProcess", "smartsp_interface"),


script_callbacks.on_ui_tabs(create_process_ui)
