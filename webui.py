# coding: utf-8

"""
The entrance of the gradio
"""
import gradio as gr
import os.path as osp
import shlex
import threading
import gradio.flagging as gradio_flagging
from omegaconf import OmegaConf

from src.pipelines.gradio_live_portrait_pipeline import GradioLivePortraitPipeline
from src.pipelines.mlx_audio_tts import MLX_AUDIO_KOKORO_VOICES
from src.runtime_assets import ensure_runtime_assets

# Gradio's CSVLogger uses multiprocessing.Lock for example/flagging logs. This
# local single-process UI only needs thread safety, and using threading.Lock
# avoids Python 3.13 resource_tracker semaphore leak warnings at shutdown.
gradio_flagging.Lock = threading.Lock


def load_description(fp):
    with open(fp, 'r', encoding='utf-8') as f:
        content = f.read()
    return content


import argparse

parser = argparse.ArgumentParser(description="FasterLivePortrait-MLX: Bring Portraits to Life in Real Time")
parser.add_argument(
    "--host_ip", type=str, default="127.0.0.1", help="host ip"
)
parser.add_argument("--port", type=int, default=9870, help="server port")
args, unknown = parser.parse_known_args()

cfg_path = "configs/mlx_infer.yaml"
infer_cfg = OmegaConf.load(cfg_path)
ensure_runtime_assets(infer_cfg)
gradio_pipeline = GradioLivePortraitPipeline(infer_cfg)
demo_theme = gr.themes.Soft(font=[gr.themes.GoogleFont("Plus Jakarta Sans")])


def gpu_wrapped_execute_video(*args, **kwargs):
    return gradio_pipeline.execute_video(*args, **kwargs)


def gpu_wrapped_execute_image(*args, **kwargs):
    return gradio_pipeline.execute_image(*args, **kwargs)


def change_animal_model(is_animal):
    ensure_runtime_assets(gradio_pipeline.cfg)
    gradio_pipeline.clean_models()
    gradio_pipeline.init_models(is_animal=is_animal)
    return gr.update(value=not is_animal)


def update_source_mode(mode):
    return gr.update(visible=mode == "Image"), gr.update(visible=mode == "Video")


def update_driving_mode(mode):
    return (
        gr.update(visible=mode == "Video"),
        gr.update(visible=mode == "Image"),
        gr.update(visible=mode == "Pickle"),
        gr.update(visible=mode == "Audio"),
        gr.update(visible=mode == "Text"),
    )


def build_cli_command(
        input_source_image_path=None,
        input_source_video_path=None,
        input_driving_video_path=None,
        input_driving_image_path=None,
        input_driving_pickle_path=None,
        input_driving_audio_path=None,
        input_driving_text=None,
        source_mode=None,
        driving_mode=None,
        flag_relative_input=True,
        flag_do_crop_input=True,
        flag_remap_input=True,
        driving_multiplier=1.0,
        flag_stitching=True,
        flag_crop_driving_video_input=True,
        flag_video_editing_head_rotation=False,
        flag_is_animal=False,
        animation_region="all",
        scale=2.3,
        vx_ratio=0.0,
        vy_ratio=-0.125,
        scale_crop_driving_video=2.2,
        vx_ratio_crop_driving_video=0.0,
        vy_ratio_crop_driving_video=-0.1,
        driving_smooth_observation_variance=1e-7,
        cfg_scale=4.0,
        voice_name=None,
):
    def has_value(value):
        return value is not None and str(value) not in ("", "None")

    selected_source_mode = source_mode if source_mode in ("Image", "Video") else None
    if selected_source_mode is None:
        selected_source_mode = "Video" if has_value(input_source_video_path) else "Image"
    source_path = input_source_video_path if selected_source_mode == "Video" else input_source_image_path

    driving_values = {
        "Video": input_driving_video_path,
        "Image": input_driving_image_path,
        "Pickle": input_driving_pickle_path,
        "Audio": input_driving_audio_path,
        "Text": input_driving_text,
    }
    selected_driving_mode = driving_mode if driving_mode in driving_values else None
    if selected_driving_mode is None:
        for mode, value in driving_values.items():
            if has_value(value):
                selected_driving_mode = mode
                break
    driving_path = driving_values[selected_driving_mode] if selected_driving_mode else None

    if selected_driving_mode not in ("Video", "Pickle"):
        return (
            "# CLI retry currently supports Video and Pickle driving through run.py.\n"
            "# Select a driving video or pickle to generate a runnable command."
        )

    parts = ["uv", "run", "python", "run.py", "--cfg", cfg_path, "--mlx-profile", "quality"]
    parts.extend(["--src_image", str(source_path) if has_value(source_path) else "<source_path>"])
    parts.extend(["--dri_video", str(driving_path) if has_value(driving_path) else "<driving_path>"])
    if flag_is_animal:
        parts.append("--animal")
    if flag_remap_input:
        parts.append("--paste_back")

    bool_flags = [
        ("--relative-motion", flag_relative_input),
        ("--do-crop", flag_do_crop_input),
        ("--stitching", flag_stitching),
        ("--crop-driving-video", flag_crop_driving_video_input),
        ("--video-editing-head-rotation", flag_video_editing_head_rotation),
    ]
    for flag, enabled in bool_flags:
        parts.append(flag if enabled else "--no-" + flag[2:])

    value_flags = [
        ("--driving-multiplier", driving_multiplier),
        ("--animation-region", animation_region),
        ("--src-scale", scale),
        ("--src-vx-ratio", vx_ratio),
        ("--src-vy-ratio", vy_ratio),
        ("--dri-scale", scale_crop_driving_video),
        ("--dri-vx-ratio", vx_ratio_crop_driving_video),
        ("--dri-vy-ratio", vy_ratio_crop_driving_video),
        ("--driving-smooth-observation-variance", driving_smooth_observation_variance),
        ("--cfg-scale", cfg_scale),
    ]
    for flag, value in value_flags:
        if value is not None:
            parts.extend([flag, str(value)])

    return " ".join(shlex.quote(part) for part in parts)


# assets
title_md = "assets/gradio/gradio_title.md"
example_portrait_dir = "assets/examples/source"
example_video_dir = "assets/examples/driving"
#################### interface logic ####################

# Define components first
eye_retargeting_slider = gr.Slider(minimum=0, maximum=0.8, step=0.01, label="target eyes-open ratio")
lip_retargeting_slider = gr.Slider(minimum=0, maximum=0.8, step=0.01, label="target lip-open ratio")
retargeting_input_image = gr.Image(type="filepath")
output_image = gr.Image(format="png", type="numpy")
output_image_paste_back = gr.Image(format="png", type="numpy")

js_func = """
    function refresh() {
        const url = new URL(window.location);

        if (url.searchParams.get('__theme') !== 'dark') {
            url.searchParams.set('__theme', 'dark');
            window.location.href = url.href;
        }
    }
    """

with gr.Blocks() as demo:
    gr.HTML(load_description(title_md))

    gr.Markdown(load_description("assets/gradio/gradio_description_upload.md"))
    with gr.Row():
        with gr.Column():
            source_mode = gr.Radio(["Image", "Video"], value="Image", label="Source Input")
            with gr.Group(visible=True) as source_image_group:
                with gr.Accordion(open=True, label="Source Image"):
                    source_image_input = gr.Image(type="filepath")
            with gr.Group(visible=False) as source_video_group:
                with gr.Accordion(open=True, label="Source Video"):
                    source_video_input = gr.Video()

            with gr.Accordion(open=True, label="Cropping Options for Source Image or Video"):
                with gr.Row():
                    flag_do_crop_input = gr.Checkbox(value=True, label="do crop (source)")
                    scale = gr.Number(value=2.3, label="source crop scale", minimum=1.8, maximum=3.2, step=0.05)
                    vx_ratio = gr.Number(value=0.0, label="source crop x", minimum=-0.5, maximum=0.5, step=0.01)
                    vy_ratio = gr.Number(value=-0.125, label="source crop y", minimum=-0.5, maximum=0.5, step=0.01)

        with gr.Column():
            driving_mode = gr.Radio(["Video", "Image", "Pickle", "Audio", "Text"], value="Video",
                                    label="Driving Input")
            with gr.Group(visible=True) as driving_video_group:
                with gr.Accordion(open=True, label="Driving Video"):
                    driving_video_input = gr.Video()
            with gr.Group(visible=False) as driving_image_group:
                with gr.Accordion(open=True, label="Driving Image"):
                    driving_image_input = gr.Image(type="filepath")

            with gr.Group(visible=False) as driving_pickle_group:
                with gr.Accordion(open=True, label="Driving Pickle"):
                    driving_pickle_input = gr.File(type="filepath", file_types=[".pkl"])

            with gr.Group(visible=False) as driving_audio_group:
                with gr.Accordion(open=True, label="Driving Audio"):
                    driving_audio_input = gr.Audio(
                        value=None,
                        type="filepath",
                        interactive=True,
                        show_label=False,
                        waveform_options=gr.WaveformOptions(
                            sample_rate=24000,
                        ),
                    )

            with gr.Group(visible=False) as driving_text_group:
                with gr.Accordion(open=True, label="Driving Text"):
                    driving_text_input = gr.Textbox(value="Hi, I am created by Faster LivePortrait!",
                                                    label="Driving Text")
                    voice_name = gr.Dropdown(
                        choices=list(MLX_AUDIO_KOKORO_VOICES),
                        value='af_heart',
                        label="Voice Name")

            # with gr.Accordion(open=False, label="Animation Instructions"):
            # gr.Markdown(load_description("assets/gradio/gradio_description_animation.md"))
            with gr.Accordion(open=True, label="Cropping Options for Driving Video"):
                with gr.Row():
                    flag_crop_driving_video_input = gr.Checkbox(value=False, label="do crop (driving)")
                    scale_crop_driving_video = gr.Number(value=2.2, label="driving crop scale", minimum=1.8,
                                                         maximum=3.2, step=0.05)
                    vx_ratio_crop_driving_video = gr.Number(value=0.0, label="driving crop x", minimum=-0.5,
                                                            maximum=0.5, step=0.01)
                    vy_ratio_crop_driving_video = gr.Number(value=-0.1, label="driving crop y", minimum=-0.5,
                                                            maximum=0.5, step=0.01)

    with gr.Row():
        with gr.Accordion(open=True, label="Animation Options"):
            with gr.Row():
                flag_relative_input = gr.Checkbox(value=False, label="relative motion")
                flag_stitching = gr.Checkbox(value=True, label="stitching")
                driving_multiplier = gr.Number(value=1.0, label="driving multiplier", minimum=0.0, maximum=2.0,
                                               step=0.02)
                cfg_scale = gr.Number(value=4.0, label="cfg_scale", minimum=0.0, maximum=10.0, step=0.5)
                flag_remap_input = gr.Checkbox(value=True, label="paste-back")
                animation_region = gr.Radio(["exp", "pose", "lip", "eyes", "all"], value="all",
                                            label="animation region")
                flag_video_editing_head_rotation = gr.Checkbox(value=False, label="relative head rotation (v2v)")
                driving_smooth_observation_variance = gr.Number(value=1e-7, label="motion smooth strength (v2v)",
                                                                minimum=1e-11, maximum=1e-2, step=1e-8)
                flag_is_animal = gr.Checkbox(value=False, label="is_animal")

    cli_command = gr.Code(
        label="CLI retry command",
        language="shell",
        lines=5,
        max_lines=8,
        interactive=False,
        wrap_lines=True,
        show_line_numbers=False,
        buttons=["copy"],
    )

    gr.Markdown(load_description("assets/gradio/gradio_description_animate_clear.md"))
    with gr.Row():
        process_button_animation = gr.Button("🚀 Animate", variant="primary")

    with gr.Column():
        with gr.Row():
            with gr.Column():
                output_video_i2v = gr.Video(autoplay=False, label="The animated video in the original image space")
            with gr.Column():
                output_video_concat_i2v = gr.Video(autoplay=False, label="The animated video")
        with gr.Row():
            with gr.Column():
                output_image_i2i = gr.Image(format="png", type="numpy",
                                            label="The animated image in the original image space",
                                            visible=False)
            with gr.Column():
                output_image_concat_i2i = gr.Image(format="png", type="numpy", label="The animated image",
                                                   visible=False)
    with gr.Row():
        process_button_reset = gr.ClearButton(
            [source_image_input, source_video_input, driving_pickle_input, driving_video_input,
             driving_image_input, driving_audio_input, driving_text_input, output_video_i2v, output_video_concat_i2v,
             output_image_i2i, output_image_concat_i2i],
            value="🧹 Clear")

    # Retargeting
    gr.Markdown(load_description("assets/gradio/gradio_description_retargeting.md"), visible=True)
    with gr.Row(visible=True):
        eye_retargeting_slider.render()
        lip_retargeting_slider.render()
    with gr.Row(visible=True):
        process_button_retargeting = gr.Button("🚗 Retargeting", variant="primary")
        process_button_reset_retargeting = gr.ClearButton(
            [
                eye_retargeting_slider,
                lip_retargeting_slider,
                retargeting_input_image,
                output_image,
                output_image_paste_back
            ],
            value="🧹 Clear"
        )
    with gr.Row(visible=True):
        with gr.Column():
            with gr.Accordion(open=True, label="Retargeting Input"):
                retargeting_input_image.render()
                gr.Examples(
                    examples=[
                        [osp.join(example_portrait_dir, "s9.jpg")],
                        [osp.join(example_portrait_dir, "s6.jpg")],
                        [osp.join(example_portrait_dir, "s10.jpg")],
                        [osp.join(example_portrait_dir, "s5.jpg")],
                        [osp.join(example_portrait_dir, "s7.jpg")],
                        [osp.join(example_portrait_dir, "s12.jpg")],
                    ],
                    inputs=[retargeting_input_image],
                    cache_examples=False,
                )
        with gr.Column():
            with gr.Accordion(open=True, label="Retargeting Result"):
                output_image.render()
        with gr.Column():
            with gr.Accordion(open=True, label="Paste-back Result"):
                output_image_paste_back.render()

    cli_command_inputs = [
        source_image_input,
        source_video_input,
        driving_video_input,
        driving_image_input,
        driving_pickle_input,
        driving_audio_input,
        driving_text_input,
        source_mode,
        driving_mode,
        flag_relative_input,
        flag_do_crop_input,
        flag_remap_input,
        driving_multiplier,
        flag_stitching,
        flag_crop_driving_video_input,
        flag_video_editing_head_rotation,
        flag_is_animal,
        animation_region,
        scale,
        vx_ratio,
        vy_ratio,
        scale_crop_driving_video,
        vx_ratio_crop_driving_video,
        vy_ratio_crop_driving_video,
        driving_smooth_observation_variance,
        cfg_scale,
        voice_name,
    ]
    demo.load(
        build_cli_command,
        inputs=cli_command_inputs,
        outputs=[cli_command],
        api_name=False,
        queue=False,
        show_progress="hidden",
    )
    for cli_input in cli_command_inputs:
        cli_input.change(
            build_cli_command,
            inputs=cli_command_inputs,
            outputs=[cli_command],
            api_name=False,
            queue=False,
            show_progress="hidden",
        )

    flag_is_animal.change(
        change_animal_model,
        inputs=[flag_is_animal],
        outputs=[flag_stitching],
        concurrency_limit=1,
        concurrency_id="flp_pipeline",
    )
    source_mode.change(
        update_source_mode,
        inputs=[source_mode],
        outputs=[source_image_group, source_video_group],
        queue=False,
        show_progress="hidden",
    )
    driving_mode.change(
        update_driving_mode,
        inputs=[driving_mode],
        outputs=[
            driving_video_group,
            driving_image_group,
            driving_pickle_group,
            driving_audio_group,
            driving_text_group,
        ],
        queue=False,
        show_progress="hidden",
    )
    # binding functions for buttons
    process_button_retargeting.click(
        # fn=gradio_pipeline.execute_image,
        fn=gpu_wrapped_execute_image,
        inputs=[eye_retargeting_slider, lip_retargeting_slider, retargeting_input_image, flag_do_crop_input],
        outputs=[output_image, output_image_paste_back],
        show_progress="full",
        trigger_mode="once",
        concurrency_limit=1,
        concurrency_id="flp_pipeline",
    )
    process_button_animation.click(
        fn=gradio_pipeline.execute_video,
        inputs=[
            source_image_input,
            source_video_input,
            driving_video_input,
            driving_image_input,
            driving_pickle_input,
            driving_audio_input,
            driving_text_input,
            source_mode,
            driving_mode,
            flag_relative_input,
            flag_do_crop_input,
            flag_remap_input,
            driving_multiplier,
            flag_stitching,
            flag_crop_driving_video_input,
            flag_video_editing_head_rotation,
            flag_is_animal,
            animation_region,
            scale,
            vx_ratio,
            vy_ratio,
            scale_crop_driving_video,
            vx_ratio_crop_driving_video,
            vy_ratio_crop_driving_video,
            driving_smooth_observation_variance,
            cfg_scale,
            voice_name
        ],
        outputs=[output_video_i2v, output_video_concat_i2v, output_image_i2i, output_image_concat_i2i],
        show_progress="full",
        trigger_mode="once",
        concurrency_limit=1,
        concurrency_id="flp_pipeline",
    )

if __name__ == '__main__':
    demo.queue(default_concurrency_limit=1, max_size=8).launch(
        server_port=args.port,
        share=False,
        server_name=args.host_ip,
        show_error=True,
        max_threads=8,
        theme=demo_theme,
        js=js_func
    )
