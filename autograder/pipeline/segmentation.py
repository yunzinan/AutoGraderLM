"""Segmentation pipeline – split student PDFs into per-question images via LLM."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Any, TypedDict

from jinja2 import Template
from langgraph.graph import END, StateGraph

from autograder.config import AppConfig
from autograder.llm import build_llm, build_vision_message, extract_json
from autograder.models import QuestionConfig, SegmentationResult
from autograder.pdf_utils import render_pdf_pages, save_answer_images

logger = logging.getLogger(__name__)


class SegState(TypedDict):
    pdf_path: str
    page_image_paths: list[str]
    questions: list[dict]
    result: dict | None
    retry_count: int
    max_retry: int
    error: str


def _render_and_cache_pages(pdf_path: str) -> list[str]:
    """Render PDF pages to temporary PNGs, return file paths."""
    from PIL import Image

    pages = render_pdf_pages(pdf_path)
    stem = Path(pdf_path).stem
    cache_dir = Path("./answers") / stem / "_pages"
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for i, img in enumerate(pages):
        p = cache_dir / f"page_{i + 1}.png"
        img.save(str(p))
        paths.append(str(p))
    return paths


def build_segmentation_graph(cfg: AppConfig) -> StateGraph:
    llm_cfg = cfg.assignment_segmentation.llm
    llm = build_llm(llm_cfg)
    prompt_tpl = Template(Path(llm_cfg.prompt_template).read_text(encoding="utf-8"))

    def segment_node(state: SegState) -> dict:
        questions = state["questions"]
        page_paths = state["page_image_paths"]

        prompt_text = prompt_tpl.render(
            num_questions=len(questions),
            questions=questions,
        )
        msg = build_vision_message(prompt_text, page_paths)
        try:
            resp = llm.invoke([msg])
            raw = extract_json(resp.content)
            if raw is None:
                return {
                    "result": None,
                    "retry_count": state["retry_count"] + 1,
                    "error": "JSON parse failed",
                }
            parsed = SegmentationResult(**raw)
            return {"result": parsed.model_dump(), "retry_count": state["retry_count"] + 1, "error": ""}
        except Exception as e:
            logger.exception("Segmentation LLM call failed for %s", state["pdf_path"])
            return {
                "result": None,
                "retry_count": state["retry_count"] + 1,
                "error": str(e),
            }

    def should_retry(state: SegState) -> str:
        if state["result"] is not None:
            return "done"
        if state["retry_count"] < state["max_retry"]:
            return "retry"
        return "done"

    graph = StateGraph(SegState)
    graph.add_node("segment", segment_node)
    graph.add_conditional_edges("segment", should_retry, {"retry": "segment", "done": END})
    graph.set_entry_point("segment")
    return graph


async def run_segmentation(
    cfg: AppConfig,
    questions: list[QuestionConfig],
    pdf_paths: list[str],
    on_progress=None,
) -> dict[str, dict]:
    """Run segmentation for a list of PDFs.

    Returns mapping pdf_stem -> {qid: [image_path, ...]}
    """
    graph = build_segmentation_graph(cfg)
    app = graph.compile()
    max_retry = cfg.assignment_segmentation.max_retry
    q_dicts = [q.model_dump() for q in questions]

    all_results: dict[str, dict] = {}

    for idx, pdf_path in enumerate(pdf_paths):
        stem = Path(pdf_path).stem
        if on_progress:
            await on_progress(f"切分中: {stem} ({idx + 1}/{len(pdf_paths)})", idx + 1, len(pdf_paths))

        page_paths = _render_and_cache_pages(pdf_path)

        init_state: SegState = {
            "pdf_path": pdf_path,
            "page_image_paths": page_paths,
            "questions": q_dicts,
            "result": None,
            "retry_count": 0,
            "max_retry": max_retry,
            "error": "",
        }
        final = app.invoke(init_state)

        if final["result"] is None:
            logger.error("Segmentation failed for %s after %d retries: %s", pdf_path, max_retry, final.get("error"))
            continue

        seg_result = SegmentationResult(**final["result"])
        page_images = render_pdf_pages(pdf_path)
        answer_map: dict[str, list[str]] = {}

        for qr in seg_result.questions:
            flat_regions = []
            for region in qr.regions:
                flat_regions.append({"qid": qr.qid, "page": region.page, "bbox": region.bbox})
            paths = save_answer_images(pdf_path, page_images, flat_regions)
            answer_map[qr.qid] = paths

        all_results[stem] = answer_map

    return all_results
