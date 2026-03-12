"""Segmentation pipeline – split student PDFs into per-question images via LLM."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, TypedDict

from jinja2 import Template
from langgraph.graph import END, StateGraph

from autograder.config import AppConfig, get_answers_dir, resolve_assignment_path
from autograder.llm import build_llm, build_vision_message_segmented, extract_json, invoke_with_log
from autograder.models import QuestionConfig, SegmentationResult
from autograder.pdf_utils import render_pdf_pages, save_answer_images, student_canonical_stem

logger = logging.getLogger(__name__)

# 发送给 LLM 的页面图最大宽度。API 端可能再次缩放，故我们主动缩放到已知尺寸，
# 并在 prompt 中声明，使模型返回的 bbox 与我们的尺寸一致，再按比例还原到原图裁剪。
SEGMENTATION_MAX_WIDTH = 1024
class SegState(TypedDict):
    pdf_path: str
    page_image_paths: list[str]
    page_dimensions: list[tuple[int, int]]  # 每页 (width, height)，与 page_image_paths 一一对应
    questions: list[dict]
    result: dict | None
    retry_count: int
    max_retry: int
    error: str


def _resize_to_max_width(img, max_width: int):
    """按 max_width 等比例缩放，返回 (resized_pil_image, (w, h))。"""
    from PIL import Image

    w, h = img.size
    if w <= max_width:
        return img, (w, h)
    scale = max_width / w
    new_w = max_width
    new_h = int(round(h * scale))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return resized, (new_w, new_h)


def _render_and_cache_pages(pdf_path: str) -> tuple[list[str], list[tuple[int, int]]]:
    """渲染 PDF，生成供 LLM 使用的固定尺度页面图，避免 API 端缩放导致 bbox 坐标系错乱。

    返回 (resized_image_paths, resized_dimensions_per_page)。
    原图仍保存到 _pages/ 供后续按比例还原 bbox 时使用（或再次 render_pdf_pages）。
    """
    from PIL import Image

    pages = render_pdf_pages(pdf_path)
    canonical = student_canonical_stem(pdf_path)
    cache_dir = get_answers_dir() / canonical / "_pages"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # 原图保存到 _pages（便于人工查看）
    for i, img in enumerate(pages):
        p = cache_dir / f"page_{i + 1}.png"
        img.save(str(p))
    # 供 LLM 的缩放图保存到 _pages/for_llm/
    llm_dir = cache_dir / "for_llm"
    llm_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    dims: list[tuple[int, int]] = []
    for i, img in enumerate(pages):
        resized, (w, h) = _resize_to_max_width(img, SEGMENTATION_MAX_WIDTH)
        p = llm_dir / f"page_{i + 1}.png"
        resized.save(str(p))
        paths.append(str(p))
        dims.append((w, h))
    return paths, dims


def _format_question_block(q: dict) -> str:
    """单题的说明文字，用于「每题文字后紧接该题题目图」的段落。"""
    idx = q.get("question_index")
    if idx is not None:
        parts = [f"- **{q.get('qid', '')}**（习题集序号 **{idx}**，学生答卷上可能写「{idx}.」；满分 {q.get('score', 0)} 分）"]
    else:
        parts = [f"- **{q.get('qid', '')}**（满分 {q.get('score', 0)} 分）"]
    if q.get("question_text"):
        parts.append(f"  题干：{q['question_text']}")
    if q.get("rubric"):
        parts.append(f"  评分要点：{q['rubric']}")
    return "\n".join(parts) + "\n\n"


def build_segmentation_graph(cfg: AppConfig) -> StateGraph:
    llm_cfg = cfg.assignment_segmentation.llm
    llm = build_llm(llm_cfg)
    prompt_dir = Path(llm_cfg.prompt_template).parent
    intro_tpl = Template((prompt_dir / "segmentation_intro.jinja").read_text(encoding="utf-8"))
    tail_tpl = Template((prompt_dir / "segmentation_tail.jinja").read_text(encoding="utf-8"))
    pages_intro_tpl = Template((prompt_dir / "segmentation_pages_intro.jinja").read_text(encoding="utf-8"))

    def segment_node(state: SegState) -> dict:
        questions = state["questions"]
        page_paths = state["page_image_paths"]
        page_dimensions = state["page_dimensions"]

        # 输入结构：intro → [每题文字 + 该题题目图] → tail（任务/输出/注意事项）→ 图片尺寸+「以下是页面图」→ 页面图
        intro_text = intro_tpl.render(num_questions=len(questions))
        segments: list[tuple[str, list[str]]] = [(intro_text, [])]

        for q in questions:
            q_text = _format_question_block(q)
            q_images = [
                str(resolve_assignment_path(p)) for p in (q.get("question_images") or [])
                if p and resolve_assignment_path(p).exists()
            ]
            segments.append((q_text, q_images))

        tail_text = tail_tpl.render()
        segments.append((tail_text, []))
        pages_intro_text = pages_intro_tpl.render(page_dimensions=page_dimensions)
        segments.append((pages_intro_text, list(page_paths)))

        msg = build_vision_message_segmented(segments)
        try:
            ctx = {"stage": "segmentation", "pdf": Path(state["pdf_path"]).stem}
            resp = invoke_with_log(llm, [msg], ctx)
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


def _process_one_pdf_sync(
    cfg: AppConfig,
    q_dicts: list[dict],
    pdf_path: str,
) -> tuple[str, dict[str, list[str]] | None]:
    """Process a single PDF (render, LLM segment, save images). Runs in thread.

    Returns (canonical_stem, answer_map) on success, (canonical_stem, None) on failure.
    使用规范 stem（学号_姓名）作为 answers 目录名，便于同一学生更新作业时覆盖。
    """
    canonical = student_canonical_stem(pdf_path)
    graph = build_segmentation_graph(cfg)
    app = graph.compile()
    max_retry = cfg.assignment_segmentation.max_retry

    page_paths, page_dims = _render_and_cache_pages(pdf_path)
    init_state: SegState = {
        "pdf_path": pdf_path,
        "page_image_paths": page_paths,
        "page_dimensions": page_dims,
        "questions": q_dicts,
        "result": None,
        "retry_count": 0,
        "max_retry": max_retry,
        "error": "",
    }
    final = app.invoke(init_state)

    if final["result"] is None:
        logger.error(
            "Segmentation failed for %s after %d retries: %s",
            pdf_path,
            max_retry,
            final.get("error"),
        )
        return (canonical, None)

    seg_result = SegmentationResult(**final["result"])
    # 持久化 for_llm 坐标系下的 segment，供人工重新切分界面加载与保存
    cache_dir = get_answers_dir() / canonical / "_pages"
    segments_path = cache_dir / "segments.json"
    segments_data = {
        "dimensions": page_dims,
        "questions": [
            {"qid": qr.qid, "regions": [{"page": r.page, "bbox": r.bbox} for r in qr.regions]}
            for qr in seg_result.questions
        ],
    }
    try:
        segments_path.write_text(json.dumps(segments_data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to write %s: %s", segments_path, e)

    full_res_pages = render_pdf_pages(pdf_path)
    answer_map: dict[str, list[str]] = {}

    for qr in seg_result.questions:
        flat_regions = []
        for region in qr.regions:
            page_1based = region.page
            if page_1based < 1 or page_1based > len(full_res_pages):
                continue
            if page_1based > len(page_dims):
                continue
            # 将 LLM 返回的 bbox（基于缩放图坐标）还原为原图坐标，不额外外扩
            resized_w, resized_h = page_dims[page_1based - 1]
            full_img = full_res_pages[page_1based - 1]
            full_w, full_h = full_img.size
            x1, y1, x2, y2 = region.bbox
            scale_x = full_w / resized_w
            scale_y = full_h / resized_h
            x1_full = max(0, min(int(round(x1 * scale_x)), full_w))
            y1_full = max(0, min(int(round(y1 * scale_y)), full_h))
            x2_full = max(0, min(int(round(x2 * scale_x)), full_w))
            y2_full = max(0, min(int(round(y2 * scale_y)), full_h))
            if x2_full <= x1_full or y2_full <= y1_full:
                continue
            bbox_full = [x1_full, y1_full, x2_full, y2_full]
            flat_regions.append({"qid": qr.qid, "page": page_1based, "bbox": bbox_full})
        if flat_regions:
            paths = save_answer_images(pdf_path, full_res_pages, flat_regions, output_dir=get_answers_dir())
            answer_map[qr.qid] = paths

    return (canonical, answer_map)


async def run_segmentation(
    cfg: AppConfig,
    questions: list[QuestionConfig],
    pdf_paths: list[str],
    on_progress=None,
) -> dict[str, dict]:
    """Run segmentation for a list of PDFs.

    使用 num_workers 并发处理多个 PDF，每个 PDF 在独立线程中执行。
    Returns mapping pdf_stem -> {qid: [image_path, ...]}.
    """
    num_workers = cfg.assignment_segmentation.num_workers
    q_dicts = [q.model_dump() for q in questions]
    total = len(pdf_paths)
    all_results: dict[str, dict] = {}
    sem = asyncio.Semaphore(num_workers)
    completed = [0]  # 用 list 以便闭包内可修改

    async def process_one(pdf_path: str) -> tuple[str, dict[str, list[str]] | None]:
        async with sem:
            stem_result, answer_map = await asyncio.to_thread(
                _process_one_pdf_sync,
                cfg,
                q_dicts,
                pdf_path,
            )
        completed[0] += 1
        if on_progress:
            await on_progress(
                f"切分中: ({completed[0]}/{total})",
                completed[0],
                total,
            )
        return (stem_result, answer_map)

    tasks = [process_one(pdf_path) for pdf_path in pdf_paths]
    for stem_result, answer_map in await asyncio.gather(*tasks):
        if answer_map is not None:
            all_results[stem_result] = answer_map

    return all_results


def load_segments_for_stem(stem: str) -> dict | None:
    """加载某份作业的 segments.json（for_llm 坐标系）。不存在则返回 None。"""
    segments_path = get_answers_dir() / stem / "_pages" / "segments.json"
    if not segments_path.exists():
        return None
    try:
        return json.loads(segments_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to load %s: %s", segments_path, e)
        return None


def save_segments_for_stem(stem: str, dimensions: list[list[int]], questions: list[dict]) -> None:
    """将 segments 写入 answers/{stem}/_pages/segments.json（for_llm 坐标系）。"""
    cache_dir = get_answers_dir() / stem / "_pages"
    cache_dir.mkdir(parents=True, exist_ok=True)
    segments_data = {
        "dimensions": dimensions,
        "questions": [
            {"qid": q["qid"], "regions": [{"page": r["page"], "bbox": r["bbox"]} for r in q.get("regions", [])]}
            for q in questions
        ],
    }
    (cache_dir / "segments.json").write_text(
        json.dumps(segments_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def apply_manual_segments(
    pdf_path: str,
    dimensions_for_llm: list[tuple[int, int]],
    questions: list[dict],
) -> dict[str, list[str]]:
    """根据 for_llm 坐标系下的 segments 重新生成答案图并保存。

    questions: [{"qid": str, "regions": [{"page": int, "bbox": [x1,y1,x2,y2]}]}]
    同一题的多个 region 按 (page, y1) 排序后依次作为该题的作答部分。
    使用规范 stem（学号_姓名）作为 answers 目录名。
    """
    stem = student_canonical_stem(pdf_path)
    full_res_pages = render_pdf_pages(pdf_path)
    answer_map: dict[str, list[str]] = {}

    for q in questions:
        qid = q.get("qid", "")
        regions = list(q.get("regions", []))
        if not regions:
            continue
        # 按 page 优先、再按 bbox 左上角 y 排序（PDF 中靠前的在前）
        regions = sorted(regions, key=lambda r: (r["page"], r["bbox"][1] if len(r["bbox"]) >= 2 else 0))
        flat_regions = []
        for r in regions:
            page_1based = r["page"]
            if page_1based < 1 or page_1based > len(full_res_pages):
                continue
            if page_1based > len(dimensions_for_llm):
                continue
            resized_w, resized_h = dimensions_for_llm[page_1based - 1]
            full_img = full_res_pages[page_1based - 1]
            full_w, full_h = full_img.size
            x1, y1, x2, y2 = r["bbox"]
            scale_x = full_w / resized_w
            scale_y = full_h / resized_h
            x1_full = int(round(x1 * scale_x))
            y1_full = int(round(y1 * scale_y))
            x2_full = int(round(x2 * scale_x))
            y2_full = int(round(y2 * scale_y))
            bbox_full = [
                max(0, min(x1_full, full_w)),
                max(0, min(y1_full, full_h)),
                max(0, min(x2_full, full_w)),
                max(0, min(y2_full, full_h)),
            ]
            if bbox_full[2] <= bbox_full[0] or bbox_full[3] <= bbox_full[1]:
                continue
            flat_regions.append({"qid": qid, "page": page_1based, "bbox": bbox_full})
        if flat_regions:
            paths = save_answer_images(pdf_path, full_res_pages, flat_regions, output_dir=get_answers_dir())
            answer_map[qid] = paths

    # 持久化 segments（for_llm 坐标）供下次编辑
    dims_list = [list(d) for d in dimensions_for_llm]
    save_segments_for_stem(stem, dims_list, questions)
    return answer_map
