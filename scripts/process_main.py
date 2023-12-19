import os

import gradio as gr

from extensions.sd_smartprocess import smartprocess
from extensions.sd_smartprocess.smartprocess import is_image
from modules import script_callbacks, shared
from modules.call_queue import wrap_gradio_gpu_call, wrap_gradio_call
from modules.ui import setup_progressbar

all_files = []
selected_files = []
all_captions = []
selected_captions = []
captioners = {
    "CLIP": False,
    "BLIP": False,
    "WD14": False,
    "DeepDanbooru": False,
    "LLAVA": True
}
all_current = None
selected_current = None
caption_current = None


def on_ui_tabs():
    with gr.Blocks() as sp_interface:
        with gr.Row():
            sp_interrogate = gr.Button(value="Caption Current", size="sm")
            sp_save_caption = gr.Button(value="Save Current", size="sm")
            sp_cancel = gr.Button(value="Cancel")
            sp_run = gr.Button(value="Preprocess All", variant='primary')
        with gr.Row(equal_height=True):
            with gr.Column(variant="panel"):
                sp_rename = gr.Checkbox(label="Rename images", value=False)
                with gr.Tab("Directories"):
                    sp_src = gr.Textbox(label='Source directory')
                    sp_dst = gr.Textbox(label='Destination directory')
                    sp_load_src = gr.Button(value="Load Source")

                with gr.Tab("Cropping"):
                    sp_size = gr.Slider(minimum=64, maximum=2048, step=64, label="Output Size", value=512)
                    sp_pad = gr.Checkbox(label="Pad Images")
                    sp_crop = gr.Checkbox(label='Crop Images')
                    sp_flip = gr.Checkbox(label='Create flipped copies')

                with gr.Tab("Captions"):
                    show_ll = captioners["LLAVA"]
                    show_clip = captioners["CLIP"]
                    show_blip = captioners["BLIP"]
                    show_wd = captioners["WD14"]
                    show_dd = captioners["DeepDanbooru"]
                    # show_mplug2 = captioners["MPLUG2"]
                    sp_caption = gr.Checkbox(label='Generate Captions')
                    sp_captioners = gr.CheckboxGroup(label='Captioners',
                                                     choices=["BLIP", "CLIP", "WD14", "DeepDanbooru", "LLAVA"], value=["LLAVA"],
                                                     interactive=True)
                    sp_caption_length = gr.Slider(label='Max Caption Length (0=unlimited)', value=75, step=75,
                                                  minimum=0, maximum=150)
                    sp_txt_action = gr.Dropdown(label='Existing Caption Action', value="ignore",
                                                choices=["ignore", "copy", "prepend", "append"])
                    with gr.Accordion(label="LLAVA", open=False, visible=show_ll) as sp_llava_section:
                        sp_llava_min_score = gr.Slider(label="Minimum Score for LLAVA Tags", value=0.85, minimum=0.01,
                                                       maximum=1,
                                                       step=0.01)
                    with gr.Accordion(label="BLIP", open=False, visible=show_blip) as sp_blip_section:
                        blip_initial_prompt = gr.Textbox(label="Initial Prompt", value="a caption for this image is: ")
                    with gr.Accordion(label="CLIP", open=False, visible=show_clip) as sp_clip_section:
                        sp_num_beams = gr.Slider(label="Number of CLIP beams", value=8, minimum=1, maximum=20)
                        sp_min_clip = gr.Slider(label="CLIP Minimum length", value=30, minimum=5, maximum=75, step=1)
                        sp_max_clip = gr.Slider(label="CLIP Maximum length", value=50, minimum=5, maximum=75, step=1)
                        sp_clip_use_v2 = gr.Checkbox(label="Use v2 CLIP Model", value=True)
                        sp_clip_append_flavor = gr.Checkbox(label="Append Flavor tags from CLIP")
                        sp_clip_max_flavors = gr.Number(label="Max flavors to append.", value=4, precision=0)
                        sp_clip_append_medium = gr.Checkbox(label="Append Medium tags from CLIP")
                        sp_clip_append_movement = gr.Checkbox(label="Append Movement tags from CLIP")
                        sp_clip_append_artist = gr.Checkbox(label="Append Artist tags from CLIP")
                        sp_clip_append_trending = gr.Checkbox(label="Append Trending tags from CLIP")
                    with gr.Accordion(label="WD14", open=False, visible=show_wd) as sp_wd14_section:
                        sp_wd14_min_score = gr.Slider(label="Minimum Score for WD14 Tags", value=0.85, minimum=0.01,
                                                      maximum=1,
                                                      step=0.01)
                    with gr.Accordion(label="DeepDanbooru", open=False, visible=show_dd) as sp_deepdanbooru_section:
                        sp_booru_min_score = gr.Slider(label="Minimum Score for DeepDanbooru Tags", value=0.85,
                                                       minimum=0.01, maximum=1, step=0.01)
                    sp_tags_to_ignore = gr.Textbox(label="Tags To Ignore", value="")
                    sp_replace_class = gr.Checkbox(label='Replace Class with Subject in Caption', value=False)
                    sp_class = gr.Textbox(label='Subject Class', placeholder='Subject class to crop (leave '
                                                                             'blank to auto-detect)')
                    sp_subject = gr.Textbox(label='Subject Name', placeholder='Subject Name to replace class '
                                                                              'with in captions')

                    def update_caption_groups(evt: gr.SelectData):
                        captioners[evt.value] = evt.selected
                        show_ll = captioners["LLAVA"]
                        show_clip = captioners["CLIP"]
                        show_blip = captioners["BLIP"]
                        show_wd = captioners["WD14"]
                        show_dd = captioners["DeepDanbooru"]
                        return gr.update(visible=show_ll), gr.update(visible=show_blip or show_clip), gr.update(visible=show_clip), gr.update(
                            visible=show_wd), gr.update(visible=show_dd)

                    sp_captioners.select(
                        fn=update_caption_groups,
                        inputs=[],
                        outputs=[sp_llava_section, sp_blip_section, sp_clip_section, sp_wd14_section, sp_deepdanbooru_section]
                    )

                with gr.Tab("Post-Processing"):
                    sp_restore_faces = gr.Checkbox(label='Restore Faces', value=False)
                    sp_face_model = gr.Dropdown(label="Face Restore Model", choices=["GFPGAN", "Codeformer"],
                                                value="GFPGAN")
                    sp_upscale = gr.Checkbox(label='Upscale and Resize', value=False)
                    sp_upscale_ratio = gr.Slider(label="Upscale Ratio", value=2, step=1, minimum=2, maximum=4)
                    sp_scaler = gr.Radio(label='Upscaler', elem_id="sp_scaler",
                                         choices=[x.name for x in shared.sd_upscalers],
                                         value=shared.sd_upscalers[0].name, type="index")

            # Preview/progress
            with gr.Column(variant="panel"):
                with gr.Row() as sp_cap_row:
                    sp_current_caption = gr.Textbox(label="Current Caption", value="", lines=2)
                    sp_caption_current = gr.Button(value="Caption Current", size="sm")
                sp_progress = gr.HTML(elem_id="sp_progress", value="")
                sp_outcome = gr.HTML(elem_id="sp_error", value="")
                sp_progressbar = gr.HTML(elem_id="sp_progressbar")
                sp_gallery = gr.Gallery(label='All', show_label=False, elem_id='sp_gallery', rows=2, columns=4,
                                        allow_preview=False)
                with gr.Row():
                    sp_add_all = gr.Button(value="Add All", size="sm")
                    sp_add_selected = gr.Button(value="Add Selected", size="sm")
                    sp_remove_selected = gr.Button(value="Remove Selected", size="sm")
                    sp_clear_selected = gr.Button(value="Remove All", size="sm")
                sp_selected = gr.Gallery(label='Selected', show_label=False, elem_id='sp_selected', rows=2, columns=4,
                                         allow_preview=False)
                sp_preview = gr.Image(label='Preview', elem_id='sp_preview', visible=False)

                setup_progressbar(sp_progressbar, sp_preview, 'sp', textinfo=sp_progress)

                def caption_single():
                    global selected_current
                    global all_current
                    files = []
                    if selected_current:
                        files.append(selected_current)
                    elif all_current:
                        files.append(all_current)
                    outputs, caption_dict, msg = start_caption(files)
                    caption = ""
                    # Get the first key from the caption dict
                    if len(caption_dict) > 0:
                        captions = list(caption_dict.values())
                        caption = captions[0]
                    return gr.update(visible=False), gr.update(value=caption), gr.update(value=msg)

                sp_caption_current.click(
                    fn=wrap_gradio_call(caption_single, extra_outputs=None, add_stats=False),
                    _js="start_smart_process",
                    inputs=[],
                    outputs=[sp_progress, sp_current_caption, sp_outcome]
                )

        def caption_selected():
            global selected_files
            return start_caption(selected_files)

        def start_caption(files, do_save=False):
            if len(files) > 0:
                outputs, caption_dict, msg = smartprocess.caption_images(files,
                                                                         captioners=captioners,
                                                                         blip_initial_prompt=blip_initial_prompt.value,
                                                                         caption_length=sp_caption_length.value,
                                                                         txt_action=sp_txt_action.value,
                                                                         num_beams=sp_num_beams.value,
                                                                         min_clip=sp_min_clip.value,
                                                                         max_clip=sp_max_clip.value,
                                                                         clip_use_v2=sp_clip_use_v2.value,
                                                                         clip_append_flavor=sp_clip_append_flavor.value,
                                                                         clip_max_flavors=sp_clip_max_flavors.value,
                                                                         clip_append_medium=sp_clip_append_medium.value,
                                                                         clip_append_movement=sp_clip_append_movement.value,
                                                                         clip_append_artist=sp_clip_append_artist.value,
                                                                         clip_append_trending=sp_clip_append_trending.value,
                                                                         wd14_min_score=sp_wd14_min_score.value,
                                                                         booru_min_score=sp_booru_min_score.value,
                                                                         tags_to_ignore=sp_tags_to_ignore.value,
                                                                         subject_class=sp_class.value,
                                                                         subject=sp_subject.value,
                                                                         replace_class=sp_replace_class.value,
                                                                         save_output=False
                                                                         )
                return outputs, caption_dict, msg

        sp_interrogate.click(
            fn=wrap_gradio_gpu_call(caption_selected, extra_outputs=[gr.update()]),
            _js="start_smart_process",
            inputs=[],
            outputs=[sp_current_caption]
        )

        sp_cancel.click(
            fn=lambda: shared.state.interrupt()
        )

        def set_all_current(evt: gr.SelectData):
            global all_current
            global caption_current
            if evt.selected:
                all_current = all_files[evt.index]
                caption_current = all_captions[evt.index]
            else:
                all_current = None
                caption_current = None
            return gr.update(value=caption_current)

        def set_selected_current(evt: gr.SelectData):
            global selected_current
            global caption_current
            if evt.selected:
                selected_current = selected_files[evt.index]
                caption_current = selected_captions[evt.index]
            else:
                selected_current = None
                caption_current = None
            return gr.update(value=caption_current)

        sp_gallery.select(fn=set_all_current, outputs=[sp_current_caption])
        sp_selected.select(fn=set_selected_current, outputs=[sp_current_caption])

        def load_src(src_path):
            # Enumerate all files recursively in src_path, and if they're an image, add them to the gallery
            global all_files
            global all_captions
            global selected_files
            global selected_captions
            all_files = []
            selected_files = []
            all_captions = []
            selected_captions = []
            for root, dirs, files in os.walk(src_path):
                for file in files:
                    filename = os.path.join(root, file)
                    if is_image(filename):
                        existing_caption_txt = ""
                        existing_caption_txt_filename = os.path.splitext(filename)[0] + '.txt'
                        if os.path.exists(existing_caption_txt_filename):
                            with open(existing_caption_txt_filename, 'r', encoding="utf8") as f:
                                existing_caption_txt = f.read()
                        else:
                            existing_caption_txt = ''.join(c for c in file if c.isalpha() or c in [" ", ", "])
                        all_captions.append(existing_caption_txt)
                        all_files.append(filename)
            print(f"Found {len(all_files)} images in {src_path}")
            return gr.update(value=all_files), gr.update(value=selected_files)

        def add_selected():
            global selected_files
            global all_current
            global selected_captions
            if all_current and all_current not in selected_files:
                selected_files.append(all_current)
                selected_captions.append(caption_current)
            return gr.update(value=selected_files)

        def remove_selected():
            global selected_files
            global selected_current
            global selected_captions
            if selected_current and selected_current in selected_files:
                caption_index = selected_files.index(selected_current)
                selected_files.remove(selected_current)
                selected_captions.pop(caption_index)
            return gr.update(value=selected_files)

        def clear_selected():
            global selected_files
            global selected_current
            global selected_captions
            selected_current = None
            selected_files = []
            selected_captions = []
            return gr.update(value=selected_files)

        def select_all():
            global selected_files
            global all_files
            global selected_captions
            global all_captions
            selected_files = all_files
            selected_captions = all_captions
            return gr.update(value=selected_files)

        sp_load_src.click(fn=load_src, inputs=[sp_src], outputs=[sp_gallery, sp_selected])

        sp_add_all.click(
            fn=select_all,
            outputs=[sp_selected]
        )

        sp_add_selected.click(
            fn=add_selected,
            inputs=[],
            outputs=[sp_selected]
        )

        sp_remove_selected.click(
            fn=remove_selected,
            inputs=[],
            outputs=[sp_selected]
        )

        sp_clear_selected.click(
            fn=clear_selected,
            outputs=[sp_selected]
        )

        sp_run.click(
            fn=wrap_gradio_gpu_call(smartprocess.do_process, extra_outputs=[gr.update()]),
            _js="start_smart_process",
            inputs=[
                sp_rename,
                sp_src,
                sp_dst,
                sp_pad,
                sp_crop,
                sp_size,
                sp_txt_action,
                sp_flip,
                sp_caption,
                sp_captioners,
                sp_caption_length,
                blip_initial_prompt,
                sp_num_beams,
                sp_min_clip,
                sp_max_clip,
                sp_clip_use_v2,
                sp_clip_append_flavor,
                sp_clip_max_flavors,
                sp_clip_append_medium,
                sp_clip_append_movement,
                sp_clip_append_artist,
                sp_clip_append_trending,
                sp_wd14_min_score,
                sp_booru_min_score,
                sp_tags_to_ignore,
                sp_class,
                sp_subject,
                sp_replace_class,
                sp_restore_faces,
                sp_face_model,
                sp_upscale,
                sp_upscale_ratio,
                sp_scaler
            ],
            outputs=[
                sp_progress,
                sp_outcome
            ],
        )

    return (sp_interface, "Smart Preprocess", "smartsp_interface"),


script_callbacks.on_ui_tabs(on_ui_tabs)
