import sys
from pathlib import Path

import pytest

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipelines.gradio_live_portrait_pipeline import GradioLivePortraitPipeline


def _pipeline_with_source_state(
    src_imgs,
    src_infos,
    prepare_result=True,
    fill_source=True,
    prepare_source_error=None,
):
    pipe = GradioLivePortraitPipeline.__new__(GradioLivePortraitPipeline)
    pipe.source_path = "source.jpg"
    pipe.src_imgs = src_imgs
    pipe.src_infos = src_infos
    pipe.prepare_source_error = None
    pipe.calls = []

    def init_vars(**kwargs):
        pipe.calls.append(("init_vars", kwargs))
        pipe.source_path = None
        pipe.src_imgs = []
        pipe.src_infos = []

    def prepare_source(source_path):
        pipe.calls.append(("prepare_source", source_path))
        pipe.source_path = source_path
        pipe.prepare_source_error = prepare_source_error
        if fill_source:
            pipe.src_imgs = [object()]
            pipe.src_infos = [[object()]]
        return prepare_result

    pipe.init_vars = init_vars
    pipe.prepare_source = prepare_source
    return pipe


def test_ensure_source_prepared_retries_when_cached_frames_are_empty():
    pipe = _pipeline_with_source_state(src_imgs=[], src_infos=[])

    pipe._ensure_source_prepared("source.jpg")

    assert pipe.calls == [("init_vars", {}), ("prepare_source", "source.jpg")]
    assert pipe.src_imgs
    assert pipe.src_infos


def test_ensure_source_prepared_keeps_valid_cached_source():
    pipe = _pipeline_with_source_state(src_imgs=[object()], src_infos=[[object()]])

    pipe._ensure_source_prepared("source.jpg")

    assert pipe.calls == []


@pytest.mark.parametrize("prepare_result", [False, True])
def test_ensure_source_prepared_errors_when_source_remains_unusable(prepare_result):
    pipe = _pipeline_with_source_state(
        src_imgs=[],
        src_infos=[],
        prepare_result=prepare_result,
        fill_source=False,
        prepare_source_error="No human face detected in the source.",
    )

    with pytest.raises(gr.Error, match="No human face detected"):
        pipe._ensure_source_prepared("source.jpg")
