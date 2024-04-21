import os
import shutil
import sys
from importlib import import_module
from typing import Tuple, Union

import gradio as gr

from extensions.sd_smartprocess import smartprocess
from extensions.sd_smartprocess.file_manager import FileManager, ImageData
from extensions.sd_smartprocess.interrogators.interrogator import InterrogatorRegistry
from extensions.sd_smartprocess.process_params import ProcessParams
from extensions.sd_smartprocess.smartprocess import get_backup_path
from modules import script_callbacks, shared
from modules.call_queue import wrap_gradio_gpu_call, wrap_gradio_call
from modules.ui import setup_progressbar
from modules.upscaler import Upscaler

refresh_symbol = "\U0001f4c2"  # 📂
delete_symbol = "\U0001F5D1"  # 🗑️

file_manager = FileManager()
# This holds all the inputs for the Smart Process tab
inputs_dict = {}

# This holds the accordions for the captioners
accordion_keys = []
captioner_accordions = []

# Sort all the tags by frequency
tags_dict = {}

current_image: ImageData = None
current_caption = None

# The list of processors to run
registry = InterrogatorRegistry()

int_dict = registry.list_interrogators()
print(f"Found {len(int_dict.keys())} interrogators: {int_dict}")
natural_captioner_names = ["BLIP", "LLAVA2", "MoonDream", "Idefics2"]
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

    for root, dirs, files in os.walk(scaler_dir):
        for file in files:
            if file.endswith("_model.py"):
                relative_path = os.path.relpath(os.path.join(root, file), scaler_dir)
                module_name = "extensions.sd_smartprocess.upscalers." + relative_path.replace(os.sep, '.').replace(
                    '.py', '')
                imported_module = import_module(module_name)

                module_dir = os.path.dirname(os.path.join(root, file))
                if module_dir not in sys.path:
                    sys.path.append(module_dir)

                for name, obj in vars(imported_module).items():
                    # Check if the object is a class and a subclass of Upscaler
                    if isinstance(obj, type) and (
                            issubclass(obj, Upscaler) or any(issubclass(base, Upscaler) for base in obj.__bases__)):
                        # Create an instance of the class and get the "scalers" attribute
                        instance = obj()
                        if hasattr(instance, 'scalers'):
                            for scaler_data in instance.scalers:
                                scalers.append(scaler_data)

    system_scalers = shared.sd_upscalers

    # Add unique scalers to the system_scalers list
    for scaler in scalers:
        if scaler.name not in [x.name for x in system_scalers]:
            system_scalers.append(scaler)


def generate_caption_section():
    global captioner_accordions
    global inputs_dict
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
                        max_value = 100
                        if "tokens" in param:
                            max_value = 300
                        temp_element = gr.Slider(label=label, value=param_value, step=1, minimum=0, maximum=max_value,
                                                 interactive=True)
                    elif isinstance(param_value, float):
                        temp_element = gr.Slider(label=label, value=param_value, step=0.01, minimum=0, maximum=1,
                                                 interactive=True)
                    elif isinstance(param_value, str):
                        if param == "group":
                            continue
                        temp_element = gr.Textbox(label=label, value=param_value, interactive=True)
                    if temp_element:
                        inputs_dict[param] = temp_element
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
                    setattr(temp_element, "do_not_save_to_config", True)
                    inputs_dict[param] = temp_element
    # Sort all_inputs and input_keys
    captioner_accordions.append(temp_accordion)
    accordion_keys.append(accordion_name)


def sort_tags_dict():
    global tags_dict
    global file_manager
    # Sort the tags into a list by frequency
    sorted_tags = sorted(tags_dict.items(), key=lambda x: x[1], reverse=True)
    tag_options = []
    for tag in sorted_tags:
        if tag[0] in file_manager.included_tags or tag[0] in file_manager.excluded_tags:
            continue
        tag_name = tag[0]
        tag_label = f"{tag_name} ({tag[1]})"
        tag_options.append(tag_label)
    return tag_options


def create_process_ui():
    """Create the UI for the Smart Process tab"""
    global tag_captioners
    global natural_captioners
    global inputs_dict
    global captioner_accordions
    global file_manager
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
                                                                                  choices=file_manager.included_strings,
                                                                                  value=file_manager.included_strings,
                                                                                  interactive=True)
                            with gr.Column():
                                sp_filter_string_exclude_group = gr.CheckboxGroup(label="Excluded Strings",
                                                                                  choices=file_manager.excluded_strings,
                                                                                  value=file_manager.excluded_strings,
                                                                                  interactive=True)

                        with gr.Row():
                            with gr.Column():
                                sp_include_tag_group = gr.CheckboxGroup(label="Included Tags",
                                                                        choices=file_manager.included_tags,
                                                                        value=file_manager.included_tags,
                                                                        interactive=True)
                            with gr.Column():
                                sp_exclude_tag_group = gr.CheckboxGroup(label="Excluded Tags",
                                                                        choices=file_manager.excluded_tags,
                                                                        value=file_manager.excluded_tags,
                                                                        interactive=True)

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
                                                    choices=["ignore", "include"])
                        generate_caption_section()
                        sp_tags_to_ignore = gr.Textbox(label="Tags To Ignore", value="")
                        sp_replace_class = gr.Checkbox(label='Replace Class with Subject in Caption', value=False)
                        sp_insert_subject = gr.Checkbox(label='Append subject to caption if not found', value=False)
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
                                                    value=shared.sd_upscalers[0].name)
                        sp_upscaler_2 = gr.Dropdown(label='Upscaler', elem_id="sp_scaler_2",
                                                    choices=[x.name for x in shared.sd_upscalers],
                                                    value=shared.sd_upscalers[0].name)

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

        def process_outputs(params: ProcessParams, current=False, all_files=False) -> Union[Tuple[gr.update, gr.update], Tuple[gr.update, gr.update, gr.update]]:
            """Process the images and update the UI
            :param params: The parameters to use for processing
            :param current: Whether to process the current image
            :param selected: Whether to process the selected images
            :return: The updated UI elements

            if current: Tuple[gr.update, gr.update, gr.update] - The updated current image, caption, and output message

            if selected/all: Tuple[gr.update, gr.update, gr.update] - The updated all images, selected images, and output message"""


            global file_manager
            global current_image
            image_data, output = smartprocess.do_process(params, all_files)
            file_manager.update_files(image_data)
            if current:
                if len(image_data) > 0:
                    current_image = image_data[0]
                    return gr.update(value=current_image.get_image()), gr.update(
                        value=current_image.caption), gr.update(value=output)
                else:
                    return gr.update(value=None), gr.update(value=None), gr.update(value=output)
            else:
                images = file_manager.filtered_files(True)
                selected_files = file_manager.filtered_and_selected_files(True)
                return gr.update(value=images), gr.update(value=selected_files), gr.update(value=output)

        def pre_process_current(*args):
            """Pre-process the current image
            :param args: The parameters to use for pre-processing
            :return: The updated current image, caption, and output message"""
            global current_image
            params = params_to_dict(*args)
            params.src_files = [current_image]
            params.pre_only()
            params.save_image = False
            return process_outputs(params, current=True)

        def caption_current(*args):
            """Caption the current image
            :param args: The parameters to use for captioning
            :return: The updated current image, caption, and output message"""
            global current_image
            cap_params = params_to_dict(*args)
            cap_params.src_files = [current_image]
            cap_params.save_caption = False
            cap_params.cap_only()
            return process_outputs(cap_params, current=True)

        def post_process_current(*args):
            """Post-process the current image
            :param args: The parameters to use for post-processing
            :return: The updated current image, caption, and output message"""
            global current_image
            post_params = params_to_dict(*args)
            post_params.src_files = [current_image]
            post_params.post_only()
            post_params.save_image = False
            return process_outputs(post_params, current=True)

        def process_selected(*args):
            """Process the selected images
            :param args: The parameters to use for processing
            :return: Updated all images, selected images, and output message"""
            global file_manager
            params = params_to_dict(*args)
            params.src_files = file_manager.filtered_and_selected_files()
            return process_outputs(params)

        def process_all(*args):
            """Process all the images
            :param args: The parameters to use for processing
            :return: Updated all images, selected images, and output message"""
            global file_manager
            params = params_to_dict(*args)
            params.src_files = file_manager.filtered_files()
            return process_outputs(params, all_files=True)

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
            global file_manager
            global current_image
            file_manager.clear()
            current_image = None

        def load_src(src_path):
            global file_manager
            file_manager.file_path = src_path
            file_manager.load_files()
            process_tags(file_manager.all_captions())
            tag_group = gr.update(choices=sort_tags_dict(), value="")
            all_files = file_manager.filtered_files(True)
            print(f"Found {len(all_files)} images in {src_path}")
            outputs = [gr.update(value=all_files), tag_group]
            for _ in elements_to_clear[2:]:
                outputs.append(gr.update(value=None))
            return outputs

        def set_all_current(evt: gr.SelectData):
            global current_image
            global file_manager
            if evt.selected:
                files = file_manager.filtered_files(False)
                current_image = files[evt.index]
                return gr.update(value=current_image.image_path), gr.update(value=current_image.caption)
            else:
                return gr.update(value=None), gr.update(value=None)

        def set_selected_current(evt: gr.SelectData):
            global current_image
            global file_manager
            if evt.selected:
                files = file_manager.filtered_files(False)
                current_image = files[evt.index]
                return gr.update(value=current_image.image_path), gr.update(value=current_image.caption)
            else:
                return gr.update(value=None), gr.update(value=None)

        def add_selected():
            global current_image
            global file_manager
            if current_image and not current_image.selected:
                current_image.selected = True
                file_manager.update_file(current_image)
            selected_files = file_manager.filtered_and_selected_files(True)
            return gr.update(value=selected_files)

        def remove_selected():
            global file_manager
            global current_image
            if current_image and current_image.selected:
                current_image.selected = False
                file_manager.update_file(current_image)
            selected_files = file_manager.filtered_and_selected_files(True)
            return gr.update(value=selected_files)

        def clear_selected():
            global file_manager
            global current_image
            current_image = None
            all_files = file_manager.all_files()
            for file in all_files:
                file.selected = False
            file_manager.update_files(all_files)
            selected_files = file_manager.filtered_and_selected_files(True)
            return gr.update(value=selected_files)

        def select_all():
            global file_manager
            all_files = file_manager.filtered_files()
            for file in all_files:
                file.selected = True
            file_manager.update_files(all_files)
            all_files = file_manager.filtered_files(True)
            return gr.update(value=all_files)

        def params_to_dict(*args):
            global inputs_dict
            input_keys = list(inputs_dict.keys())
            pp = list(args)
            params_dict = dict(zip(input_keys, pp))
            pp = ProcessParams().from_dict(params_dict)
            return pp

        def include_tag(tag_dropdown_value):
            global file_manager
            included_tags = file_manager.included_tags
            tag_dropdown_value = tag_dropdown_value.split(" (")[0]
            if tag_dropdown_value not in included_tags:
                included_tags.append(tag_dropdown_value)
            # Sort filter_tags_include by frequency, then alphabetically, ensuring we only add the values in filter_tags_include
            included_tags = [x[0] for x in sorted(tags_dict.items(), key=lambda x: (-x[1], x[0])) if
                             x[0] in included_tags]
            file_manager.included_tags = included_tags
            file_manager.update_filters()
            tag_group = gr.update(choices=sort_tags_dict(), value="")
            all_gallery_items = file_manager.filtered_files(True)
            selected_gallery_items = file_manager.filtered_files(True)
            return tag_group, gr.update(value=included_tags, choices=included_tags), gr.update(
                value=all_gallery_items), gr.update(value=selected_gallery_items)

        def exclude_tag(tag_dropdown_value):
            global file_manager
            excluded_tags = file_manager.excluded_tags
            tag_dropdown_value = tag_dropdown_value.split(" (")[0]
            if tag_dropdown_value not in excluded_tags:
                excluded_tags.append(tag_dropdown_value)
            # Sort filter_tags_exclude by frequency, then alphabetically
            excluded_tags = [x[0] for x in sorted(tags_dict.items(), key=lambda x: (-x[1], x[0])) if
                             x[0] in excluded_tags]
            file_manager.excluded_tags = excluded_tags
            file_manager.update_filters()
            tag_group = gr.update(choices=sort_tags_dict(), value="")
            all_gallery_items = file_manager.filtered_files(True)
            selected_gallery_items = file_manager.filtered_and_selected_files(True)
            return tag_group, gr.update(value=excluded_tags, choices=excluded_tags), gr.update(
                value=all_gallery_items), gr.update(value=selected_gallery_items)

        def remove_include_tag(select_data: gr.SelectData):
            global file_manager
            included_tags = file_manager.included_tags

            if not select_data.selected:
                included_tags.remove(select_data.value)
            included_tags = [x[0] for x in sorted(tags_dict.items(), key=lambda x: (-x[1], x[0])) if
                             x[0] in included_tags]
            file_manager.included_tags = included_tags
            file_manager.update_filters()
            all_gallery_items = file_manager.filtered_files(True)
            selected_gallery_items = file_manager.filtered_and_selected_files(True)
            return gr.update(choices=sort_tags_dict(), value=""), gr.update(value=included_tags,
                                                                            choices=included_tags), gr.update(
                value=all_gallery_items), gr.update(value=selected_gallery_items)

        def remove_exclude_tag(select_data: gr.SelectData):
            global file_manager
            filter_tags_exclude = file_manager.excluded_tags
            if select_data.selected:
                filter_tags_exclude.append(select_data.value)
            else:
                filter_tags_exclude.remove(select_data.value)
            # Sort filter tags alphabetically
            filter_tags_exclude = [x[0] for x in sorted(tags_dict.items(), key=lambda x: (-x[1], x[0])) if
                                   x[0] in filter_tags_exclude]
            file_manager.excluded_tags = filter_tags_exclude
            file_manager.update_filters()
            all_gallery_items = file_manager.filtered_files(True)
            selected_gallery_items = file_manager.filtered_and_selected_files(True)
            return gr.update(choices=sort_tags_dict(), value=""), gr.update(value=filter_tags_exclude,
                                                                            choices=filter_tags_exclude), gr.update(
                value=all_gallery_items), gr.update(value=selected_gallery_items)

        def include_string(string_group_value):
            global file_manager
            filter_string_include = file_manager.included_strings
            if string_group_value not in filter_string_include:
                filter_string_include.append(string_group_value)
            # Sort filter_string_include alphabetically
            filter_string_include = sorted(filter_string_include)
            file_manager.included_strings = filter_string_include
            file_manager.update_filters()
            all_gallery_items = file_manager.filtered_files(True)
            selected_gallery_items = file_manager.filtered_and_selected_files(True)
            return gr.update(value=filter_string_include, choices=filter_string_include), gr.update(
                value=all_gallery_items), gr.update(value=selected_gallery_items), gr.update(value="")

        def exclude_string(string_group_value):
            global file_manager
            filter_string_exclude = file_manager.excluded_strings
            if string_group_value not in filter_string_exclude:
                filter_string_exclude.append(string_group_value)
            # Sort filter_string_exclude alphabetically
            filter_string_exclude = sorted(filter_string_exclude)
            file_manager.excluded_strings = filter_string_exclude
            file_manager.update_filters()
            all_gallery_items = file_manager.filtered_files(True)
            selected_gallery_items = file_manager.filtered_and_selected_files(True)
            return gr.update(value=filter_string_exclude, choices=filter_string_exclude), gr.update(
                value=all_gallery_items), gr.update(value=selected_gallery_items), gr.update(value="")

        def remove_include_string(select_data: gr.SelectData):
            global file_manager
            filter_string_include = file_manager.included_strings
            if select_data.selected:
                filter_string_include.append(select_data.value)
            else:
                filter_string_include.remove(select_data.value)
            filter_string_include = sorted(filter_string_include)
            file_manager.included_strings = filter_string_include
            file_manager.update_filters()
            all_gallery_items = file_manager.filtered_files(True)
            selected_gallery_items = file_manager.filtered_and_selected_files(True)
            return gr.update(value=filter_string_include, choices=filter_string_include), gr.update(
                value=all_gallery_items), gr.update(value=selected_gallery_items), gr.update(value="")

        def remove_exclude_string(select_data: gr.SelectData):
            global file_manager
            filter_string_exclude = file_manager.excluded_strings
            if select_data.selected:
                filter_string_exclude.append(select_data.value)
            else:
                filter_string_exclude.remove(select_data.value)
            filter_string_exclude = sorted(filter_string_exclude)
            file_manager.excluded_strings = filter_string_exclude
            file_manager.update_filters()
            all_gallery_items = file_manager.filtered_files(True)
            selected_gallery_items = file_manager.filtered_and_selected_files(True)
            return gr.update(value=filter_string_exclude, choices=filter_string_exclude), gr.update(
                value=all_gallery_items), gr.update(value=selected_gallery_items), gr.update(value="")

        def save_current_caption(caption, do_backup=True):
            temp_params = ProcessParams()
            temp_params.do_backup = do_backup
            global current_image
            if current_image:
                image_path = current_image.image_path
                caption_txt_filename = os.path.splitext(image_path)[0] + '.txt'
                caption_txt_filename, caption_backup_path = get_backup_path(caption_txt_filename, temp_params)
                if caption_txt_filename != caption_backup_path:
                    shutil.copy(caption_txt_filename, caption_backup_path)
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
                image_path = current_image.image_path
                save_path, backup_path = get_backup_path(image_path, temp_params)
                if save_path != backup_path:
                    shutil.copy(image_path, backup_path)
                current_image_element = current_image.get_image()
                current_image_element.save(save_path)
                return gr.update(value=save_path), gr.update(value="Image saved to " + save_path)
            else:
                return gr.update(value=None), gr.update(value="No image selected")

        inputs_dict = {
            "auto_save": sp_auto_save,
            "caption": sp_caption,
            "captioners": sp_captioners,
            "class": sp_class,
            "clear_src": sp_clear_src,
            "crop": sp_crop,
            "crop_mode": sp_crop_mode,
            "do_backup": sp_do_backup,
            "do_rename": sp_do_rename,
            "face_model": sp_face_model,
            'insert_subject': sp_insert_subject,
            "nl_captioners": sp_nl_captioners,
            "pad": sp_pad,
            "replace_class": sp_replace_class,
            "restore_faces": sp_restore_faces,
            "size": sp_size,
            "src": sp_src,
            "subject": sp_subject,
            "tags_to_ignore": sp_tags_to_ignore,
            "txt_action": sp_txt_action,
            "upscale": sp_upscale,
            "upscale_ratio": sp_upscale_ratio,
            "upscaler_1": sp_upscaler_1,
            "upscaler_2": sp_upscaler_2
        }

        def all_inputs():
            return [x for x in inputs_dict.values()]

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
            fn=wrap_gradio_call(pre_process_current),
            _js="start_smart_process",
            inputs=all_inputs(),
            outputs=[sp_preview, sp_current_caption, sp_outcome]
        )

        sp_save_current_pre.click(
            fn=save_current_image,
            inputs=[sp_do_backup, sp_do_rename],
            outputs=[sp_preview, sp_outcome]
        )

        sp_caption_current.click(
            fn=wrap_gradio_call(caption_current),
            _js="start_smart_process",
            inputs=all_inputs(),
            outputs=[sp_preview, sp_current_caption, sp_outcome]
        )

        sp_save_current_caption.click(
            fn=save_current_caption,
            inputs=[sp_current_caption, sp_do_backup],
            outputs=[sp_outcome]
        )

        sp_post_current.click(
            fn=wrap_gradio_call(post_process_current),
            _js="start_smart_process",
            inputs=all_inputs(),
            outputs=[sp_preview, sp_current_caption, sp_outcome]
        )

        sp_save_current_post.click(
            fn=save_current_image,
            inputs=[sp_do_backup, sp_do_rename],
            outputs=[sp_preview, sp_outcome]
        )

        sp_process_selected.click(
            fn=wrap_gradio_call(process_selected),
            _js="start_smart_process",
            inputs=all_inputs(),
            outputs=[sp_gallery_all, sp_gallery_selected, sp_outcome]
        )

        sp_process_all.click(
            fn=wrap_gradio_call(process_all),
            _js="start_smart_process",
            inputs=all_inputs(),
            outputs=[sp_gallery_all, sp_gallery_selected, sp_outcome],
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
