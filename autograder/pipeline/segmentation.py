"""Segmentation pipeline – split student PDFs into per-question images via LLM."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, TypedDict

from jinja2 import Template
from langgraph.graph import END, StateGraph
from PIL import Image, ImageDraw, ImageFont

from autograder.config import AppConfig, SegmentationPostprocessConfig, get_answers_dir, resolve_assignment_path
from autograder.llm import build_llm, build_vision_message_segmented, extract_json, invoke_with_log
from autograder.models import BBox, QuestionConfig, QuestionRegion, SegmentationResult
from autograder.pdf_utils import render_pdf_pages, save_answer_images, student_canonical_stem

logger = logging.getLogger(__name__)

# 发送给 LLM 的页面图最大宽度。API 端可能再次缩放，故我们主动缩放到已知尺寸，
# 并在 prompt 中声明，使模型返回的 bbox 与我们的尺寸一致，再按比例还原到原图裁剪。
SEGMENTATION_MAX_WIDTH = 1024
class SegState(TypedDict):
    pdf_path: str
    page_image_paths: list[str]
    page_dimensions: list[tuple[int, int]]  # 每页 (width, height)，与 page_image_paths 一一对应
    page_numbers: list[int]
    questions: list[dict]
    result: dict | None
    retry_count: int
    max_retry: int
    error: str


class CandidateBand(TypedDict):
    id: str
    page: int
    bbox: list[int]


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


def _clamp_bbox(bbox: list[int] | tuple[int, int, int, int], width: int, height: int) -> list[int] | None:
    """Clamp and normalize a bbox in page-image coordinates."""
    if len(bbox) != 4:
        return None
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    left = max(0, min(x1, x2, width))
    right = max(0, min(max(x1, x2), width))
    top = max(0, min(y1, y2, height))
    bottom = max(0, min(max(y1, y2), height))
    if right <= left or bottom <= top:
        return None
    return [left, top, right, bottom]


def _expand_bbox(bbox: list[int], width: int, height: int, x_margin: int, y_margin: int) -> list[int] | None:
    return _clamp_bbox(
        [
            bbox[0] - x_margin,
            bbox[1] - y_margin,
            bbox[2] + x_margin,
            bbox[3] + y_margin,
        ],
        width,
        height,
    )


def _ensure_min_height(bbox: list[int], width: int, height: int, min_height: int) -> list[int]:
    if bbox[3] - bbox[1] >= min_height:
        return bbox
    center = (bbox[1] + bbox[3]) // 2
    half = max(1, min_height // 2)
    expanded = _clamp_bbox([bbox[0], center - half, bbox[2], center + half], width, height)
    return expanded or bbox


def _ink_bbox_in_region(
    page_img: Image.Image,
    search_bbox: list[int],
    *,
    threshold: int,
    min_ink_pixels: int,
) -> list[int] | None:
    """Return the dark-pixel bbox inside search_bbox, in page coordinates."""
    crop = page_img.crop(tuple(search_bbox)).convert("L")
    # Threshold to a white-on-black mask in L mode so getbbox and histogram are cheap.
    mask = crop.point(lambda p: 255 if p < threshold else 0, "L")
    hist = mask.histogram()
    ink_pixels = hist[255] if len(hist) > 255 else 0
    if ink_pixels < min_ink_pixels:
        return None
    local = mask.getbbox()
    if local is None:
        return None
    return [
        search_bbox[0] + local[0],
        search_bbox[1] + local[1],
        search_bbox[0] + local[2],
        search_bbox[1] + local[3],
    ]


def _detect_ink_bands(
    page_img: Image.Image,
    *,
    threshold: int,
    min_ink_per_row: int,
    gap_tolerance: int,
) -> list[tuple[int, int]]:
    """Detect vertical ink bands as (y1, y2) ranges in page coordinates."""
    gray = page_img.convert("L")
    width, height = gray.size
    mask = gray.point(lambda p: 255 if p < threshold else 0, "L")
    mask_bytes = mask.tobytes()
    bands: list[tuple[int, int]] = []
    active_start: int | None = None
    last_active_y: int | None = None

    for y in range(height):
        row_start = y * width
        ink = mask_bytes[row_start:row_start + width].count(255)
        if ink >= min_ink_per_row:
            if active_start is None:
                active_start = y
            last_active_y = y
            continue
        if active_start is not None and last_active_y is not None and y - last_active_y > gap_tolerance:
            bands.append((active_start, last_active_y + 1))
            active_start = None
            last_active_y = None

    if active_start is not None and last_active_y is not None:
        bands.append((active_start, last_active_y + 1))
    return bands


def _merge_band_ranges(
    ink_bands: list[tuple[int, int]],
    bridge_gap: int,
) -> list[tuple[int, int]]:
    """Merge nearby vertical ink bands into candidate answer blocks."""
    if not ink_bands:
        return []
    merged: list[tuple[int, int]] = []
    cur_top, cur_bottom = ink_bands[0]
    for top, bottom in ink_bands[1:]:
        if top - cur_bottom <= bridge_gap:
            cur_bottom = max(cur_bottom, bottom)
            continue
        merged.append((cur_top, cur_bottom))
        cur_top, cur_bottom = top, bottom
    merged.append((cur_top, cur_bottom))
    return merged


def _limit_band_ranges(
    ranges: list[tuple[int, int]],
    max_count: int,
) -> list[tuple[int, int]]:
    """Keep prompts bounded by merging the closest adjacent ranges first."""
    ranges = list(ranges)
    while len(ranges) > max_count and len(ranges) > 1:
        best_idx = 0
        best_gap = ranges[1][0] - ranges[0][1]
        for idx in range(1, len(ranges) - 1):
            gap = ranges[idx + 1][0] - ranges[idx][1]
            if gap < best_gap:
                best_idx = idx
                best_gap = gap
        merged = (
            ranges[best_idx][0],
            max(ranges[best_idx][1], ranges[best_idx + 1][1]),
        )
        ranges[best_idx:best_idx + 2] = [merged]
    return ranges


def _build_candidate_bands(
    page_img: Image.Image,
    *,
    page_no: int,
    pp_cfg: SegmentationPostprocessConfig,
) -> list[CandidateBand]:
    """Build deterministic answer-band candidates for LLM assignment."""
    width, height = page_img.size
    ink_bands = _detect_ink_bands(
        page_img,
        threshold=pp_cfg.ink_threshold,
        min_ink_per_row=pp_cfg.band_min_ink_per_row,
        gap_tolerance=pp_cfg.band_gap_tolerance,
    )
    ranges = _merge_band_ranges(ink_bands, pp_cfg.candidate_bridge_gap)
    ranges = _limit_band_ranges(ranges, pp_cfg.candidate_max_bands)

    candidates: list[CandidateBand] = []
    for idx, (top, bottom) in enumerate(ranges, start=1):
        y1 = max(0, top - pp_cfg.band_padding)
        y2 = min(height, bottom + pp_cfg.band_padding)
        if pp_cfg.full_width:
            x1 = pp_cfg.horizontal_margin
            x2 = max(pp_cfg.horizontal_margin + 1, width - pp_cfg.horizontal_margin)
        else:
            x1 = 0
            x2 = width
        bbox = _clamp_bbox([x1, y1, x2, y2], width, height)
        if bbox is None:
            continue
        bbox = _ensure_min_height(bbox, width, height, pp_cfg.min_box_height)
        candidates.append({"id": f"B{idx}", "page": page_no, "bbox": bbox})
    return candidates


def _save_candidate_overlay(
    page_img: Image.Image,
    candidates: list[CandidateBand],
    path: Path,
) -> None:
    """Save an annotated page image with candidate IDs visible to the model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    annotated = page_img.convert("RGBA")
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    colors = [
        (220, 38, 38, 255),
        (37, 99, 235, 255),
        (5, 150, 105, 255),
        (217, 119, 6, 255),
        (124, 58, 237, 255),
    ]

    for idx, candidate in enumerate(candidates):
        x1, y1, x2, y2 = candidate["bbox"]
        color = colors[idx % len(colors)]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        label = candidate["id"]
        label_x = max(2, x1 + 4)
        label_y = max(2, y1 + 4)
        try:
            text_box = draw.textbbox((label_x, label_y), label, font=font)
        except AttributeError:
            text_w, text_h = draw.textsize(label, font=font)
            text_box = (label_x, label_y, label_x + text_w, label_y + text_h)
        draw.rectangle(
            (
                text_box[0] - 2,
                text_box[1] - 1,
                text_box[2] + 2,
                text_box[3] + 1,
            ),
            fill=(255, 255, 255, 230),
        )
        draw.text((label_x, label_y), label, fill=color, font=font)

    annotated.convert("RGB").save(path)


def _candidate_ids_from_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, str):
        raw_items = value.replace(",", " ").replace(";", " ").split()
    else:
        raw_items = [value]

    out: list[str] = []
    for item in raw_items:
        cid = str(item).strip().strip("\"'[](){}")
        if not cid:
            continue
        if cid.isdigit():
            cid = f"B{cid}"
        cid = cid.upper()
        if cid not in out:
            out.append(cid)
    return out


def _regions_from_candidate_bands(
    selected: list[CandidateBand],
    pp_cfg: SegmentationPostprocessConfig,
) -> list[BBox]:
    if not selected:
        return []
    unique: list[CandidateBand] = []
    seen: set[tuple[int, str]] = set()
    for candidate in selected:
        key = (candidate["page"], candidate["id"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    ordered = sorted(
        unique,
        key=lambda c: (c["page"], c["bbox"][1], c["bbox"][0]),
    )
    regions: list[BBox] = []
    group: list[CandidateBand] = []

    def flush_group() -> None:
        if not group:
            return
        page = group[0]["page"]
        left = min(c["bbox"][0] for c in group)
        top = min(c["bbox"][1] for c in group)
        right = max(c["bbox"][2] for c in group)
        bottom = max(c["bbox"][3] for c in group)
        regions.append(BBox(page=page, bbox=[left, top, right, bottom]))

    for candidate in ordered:
        if not group:
            group = [candidate]
            continue
        prev = group[-1]
        same_page = candidate["page"] == prev["page"]
        gap = candidate["bbox"][1] - prev["bbox"][3]
        if same_page and gap <= pp_cfg.band_bridge_gap:
            group.append(candidate)
            continue
        flush_group()
        group = [candidate]

    flush_group()
    return regions


def _candidate_assignment_to_segmentation_result(
    raw: dict | list,
    candidates: list[CandidateBand],
    pp_cfg: SegmentationPostprocessConfig,
) -> SegmentationResult:
    candidate_by_id = {c["id"].upper(): c for c in candidates}
    if isinstance(raw, dict):
        entries = raw.get("questions") or raw.get("assignments") or raw.get("answers") or []
    else:
        entries = raw
    if not isinstance(entries, list):
        entries = []

    out: list[QuestionRegion] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        qid = str(item.get("qid") or item.get("question_id") or item.get("question") or "").strip()
        if not qid:
            continue
        ids_value = (
            item.get("candidate_ids")
            if "candidate_ids" in item
            else item.get("candidates", item.get("bands", item.get("candidate_id")))
        )
        candidate_ids = _candidate_ids_from_value(ids_value)
        selected = [candidate_by_id[cid] for cid in candidate_ids if cid in candidate_by_id]
        out.append(QuestionRegion(qid=qid, regions=_regions_from_candidate_bands(selected, pp_cfg)))
    return SegmentationResult(questions=out)


def _snap_bbox_to_ink_bands(
    bbox: list[int],
    ink_bands: list[tuple[int, int]],
    width: int,
    height: int,
    pp_cfg: SegmentationPostprocessConfig,
) -> list[int]:
    """Expand bbox to nearby vertical ink bands that likely form one answer block."""
    if not ink_bands:
        return bbox

    search_top = max(0, bbox[1] - pp_cfg.snap_padding)
    search_bottom = min(height, bbox[3] + pp_cfg.snap_padding)
    selected = [
        idx for idx, (top, bottom) in enumerate(ink_bands)
        if bottom >= search_top and top <= search_bottom
    ]
    if not selected:
        return bbox

    first = min(selected)
    last = max(selected)
    bridge_top = max(0, search_top - pp_cfg.band_bridge_gap)
    bridge_bottom = min(height, search_bottom + pp_cfg.band_bridge_gap)

    while first > 0:
        prev_top, prev_bottom = ink_bands[first - 1]
        cur_top, _cur_bottom = ink_bands[first]
        if prev_bottom < bridge_top or cur_top - prev_bottom > pp_cfg.band_bridge_gap:
            break
        first -= 1

    while last < len(ink_bands) - 1:
        _cur_top, cur_bottom = ink_bands[last]
        next_top, _next_bottom = ink_bands[last + 1]
        if next_top > bridge_bottom or next_top - cur_bottom > pp_cfg.band_bridge_gap:
            break
        last += 1

    band_top = max(0, ink_bands[first][0] - pp_cfg.band_padding)
    band_bottom = min(height, ink_bands[last][1] + pp_cfg.band_padding)
    snapped = _clamp_bbox(
        [bbox[0], min(bbox[1], band_top), bbox[2], max(bbox[3], band_bottom)],
        width,
        height,
    )
    return snapped or bbox


def _repair_region_bbox(
    bbox: list[int],
    page_img: Image.Image,
    page_dim: tuple[int, int],
    pp_cfg: SegmentationPostprocessConfig,
    ink_bands: list[tuple[int, int]] | None = None,
) -> list[int] | None:
    """Make a coarse LLM bbox safer for answer cropping."""
    width, height = page_dim
    fixed = _clamp_bbox(bbox, width, height)
    if fixed is None:
        return None

    fixed = _ensure_min_height(fixed, width, height, pp_cfg.min_box_height)

    if pp_cfg.snap_to_content:
        search = _expand_bbox(
            fixed,
            width,
            height,
            pp_cfg.snap_padding,
            pp_cfg.snap_padding,
        )
        if search:
            ink = _ink_bbox_in_region(
                page_img,
                search,
                threshold=pp_cfg.ink_threshold,
                min_ink_pixels=pp_cfg.min_ink_pixels,
            )
            if ink:
                fixed = [
                    min(fixed[0], ink[0]),
                    min(fixed[1], ink[1]),
                    max(fixed[2], ink[2]),
                    max(fixed[3], ink[3]),
                ]

    if pp_cfg.snap_to_ink_bands and ink_bands:
        fixed = _snap_bbox_to_ink_bands(fixed, ink_bands, width, height, pp_cfg)

    if pp_cfg.full_width:
        fixed[0] = pp_cfg.horizontal_margin
        fixed[2] = max(pp_cfg.horizontal_margin + 1, width - pp_cfg.horizontal_margin)
        x_margin = 0
    else:
        x_margin = pp_cfg.horizontal_margin

    fixed = _expand_bbox(
        fixed,
        width,
        height,
        x_margin,
        pp_cfg.vertical_margin,
    )
    if fixed is None:
        return None
    return _ensure_min_height(fixed, width, height, pp_cfg.min_box_height)


def _postprocess_segmentation_result(
    seg_result: SegmentationResult,
    page_dims: list[tuple[int, int]],
    page_image_paths: list[str],
    pp_cfg: SegmentationPostprocessConfig,
    expected_qids: list[str],
) -> SegmentationResult:
    """Repair LLM regions and preserve all expected qids for the manual editor."""
    if not pp_cfg.enabled:
        seen = {q.qid for q in seg_result.questions}
        missing = [
            QuestionRegion(qid=qid, regions=[])
            for qid in expected_qids
            if qid not in seen
        ]
        return SegmentationResult(questions=list(seg_result.questions) + missing)

    page_images: list[Image.Image] = []
    try:
        page_images = [Image.open(path).convert("RGB") for path in page_image_paths]
        page_ink_bands = [
            _detect_ink_bands(
                img,
                threshold=pp_cfg.ink_threshold,
                min_ink_per_row=pp_cfg.band_min_ink_per_row,
                gap_tolerance=pp_cfg.band_gap_tolerance,
            )
            if pp_cfg.snap_to_ink_bands else []
            for img in page_images
        ]
        by_qid: dict[str, list[BBox]] = {qid: [] for qid in expected_qids}
        extras: dict[str, list[BBox]] = {}

        for qr in seg_result.questions:
            target = by_qid.setdefault(qr.qid, []) if qr.qid in by_qid else extras.setdefault(qr.qid, [])
            for region in qr.regions:
                page_idx = region.page - 1
                if page_idx < 0 or page_idx >= len(page_dims) or page_idx >= len(page_images):
                    continue
                repaired = _repair_region_bbox(
                    region.bbox,
                    page_images[page_idx],
                    page_dims[page_idx],
                    pp_cfg,
                    page_ink_bands[page_idx],
                )
                if repaired is None:
                    continue
                target.append(BBox(page=region.page, bbox=repaired))

        out: list[QuestionRegion] = []
        for qid in expected_qids:
            regions = sorted(by_qid.get(qid, []), key=lambda r: (r.page, r.bbox[1], r.bbox[0]))
            out.append(QuestionRegion(qid=qid, regions=regions))
        for qid, regions in extras.items():
            out.append(QuestionRegion(
                qid=qid,
                regions=sorted(regions, key=lambda r: (r.page, r.bbox[1], r.bbox[0])),
            ))
        return SegmentationResult(questions=out)
    finally:
        for img in page_images:
            try:
                img.close()
            except Exception:
                pass


def _normalize_result_page_numbers(seg_result: SegmentationResult, page_numbers: list[int]) -> SegmentationResult:
    """Repair page numbers when a single-page call returns page=1 for a later PDF page."""
    if len(page_numbers) != 1:
        return seg_result
    actual_page = page_numbers[0]
    for qr in seg_result.questions:
        for region in qr.regions:
            if region.page != actual_page:
                region.page = actual_page
    return seg_result


def _merge_segmentation_results(
    results: list[SegmentationResult],
    expected_qids: list[str],
) -> SegmentationResult:
    """Merge page-level segmentation results while preserving configured qid order."""
    by_qid: dict[str, list[BBox]] = {qid: [] for qid in expected_qids}
    extras: dict[str, list[BBox]] = {}
    for result in results:
        for qr in result.questions:
            target = by_qid.setdefault(qr.qid, []) if qr.qid in by_qid else extras.setdefault(qr.qid, [])
            target.extend(qr.regions)

    out: list[QuestionRegion] = []
    for qid in expected_qids:
        out.append(QuestionRegion(
            qid=qid,
            regions=sorted(by_qid.get(qid, []), key=lambda r: (r.page, r.bbox[1], r.bbox[0])),
        ))
    for qid, regions in extras.items():
        out.append(QuestionRegion(
            qid=qid,
            regions=sorted(regions, key=lambda r: (r.page, r.bbox[1], r.bbox[0])),
        ))
    return SegmentationResult(questions=out)


def _qid_regions_map(seg_result: SegmentationResult) -> dict[str, list[BBox]]:
    by_qid: dict[str, list[BBox]] = {}
    for qr in seg_result.questions:
        by_qid.setdefault(qr.qid, []).extend(qr.regions)
    return by_qid


def _segmentation_diagnostics(
    seg_result: SegmentationResult,
    expected_qids: list[str],
) -> dict[str, Any]:
    by_qid = _qid_regions_map(seg_result)
    missing_qids = [qid for qid in expected_qids if not by_qid.get(qid)]
    nonempty_qids = [qid for qid in expected_qids if by_qid.get(qid)]
    total_regions = sum(len(by_qid.get(qid, [])) for qid in expected_qids)
    assigned_ratio = len(nonempty_qids) / len(expected_qids) if expected_qids else 1.0
    return {
        "expected_qids": len(expected_qids),
        "nonempty_qids": len(nonempty_qids),
        "missing_qids": missing_qids,
        "assigned_ratio": assigned_ratio,
        "total_regions": total_regions,
    }


def _fill_missing_qids_from_fallback(
    primary: SegmentationResult,
    fallback: SegmentationResult,
    expected_qids: list[str],
) -> SegmentationResult:
    primary_by_qid = _qid_regions_map(primary)
    fallback_by_qid = _qid_regions_map(fallback)
    extras: dict[str, list[BBox]] = {}
    expected_set = set(expected_qids)
    for qr in primary.questions:
        if qr.qid not in expected_set:
            extras.setdefault(qr.qid, []).extend(qr.regions)
    for qr in fallback.questions:
        if qr.qid not in expected_set and qr.qid not in extras:
            extras.setdefault(qr.qid, []).extend(qr.regions)

    out: list[QuestionRegion] = []
    for qid in expected_qids:
        regions = primary_by_qid.get(qid) or fallback_by_qid.get(qid, [])
        out.append(QuestionRegion(
            qid=qid,
            regions=sorted(regions, key=lambda r: (r.page, r.bbox[1], r.bbox[0])),
        ))
    for qid, regions in extras.items():
        out.append(QuestionRegion(
            qid=qid,
            regions=sorted(regions, key=lambda r: (r.page, r.bbox[1], r.bbox[0])),
        ))
    return SegmentationResult(questions=out)


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
        page_numbers = state["page_numbers"]

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
        page_infos = [
            {"page": page_no, "width": dim[0], "height": dim[1]}
            for page_no, dim in zip(page_numbers, page_dimensions)
        ]
        pages_intro_text = pages_intro_tpl.render(page_dimensions=page_dimensions, page_infos=page_infos)
        segments.append((pages_intro_text, list(page_paths)))

        msg = build_vision_message_segmented(segments)
        try:
            ctx = {
                "stage": "segmentation",
                "pdf": Path(state["pdf_path"]).stem,
                "pages": ",".join(str(n) for n in page_numbers),
            }
            resp = invoke_with_log(llm, [msg], ctx)
            raw = extract_json(resp.content)
            if raw is None:
                return {
                    "result": None,
                    "retry_count": state["retry_count"] + 1,
                    "error": "JSON parse failed",
                }
            parsed = _normalize_result_page_numbers(SegmentationResult(**raw), page_numbers)
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


def _run_page_by_page_segmentation(
    app: Any,
    *,
    pdf_path: str,
    q_dicts: list[dict],
    page_paths: list[str],
    page_dims: list[tuple[int, int]],
    max_retry: int,
    expected_qids: list[str],
) -> SegmentationResult | None:
    page_results: list[SegmentationResult] = []
    page_errors: list[str] = []
    for page_idx, (page_path, page_dim) in enumerate(zip(page_paths, page_dims)):
        page_no = page_idx + 1
        init_state: SegState = {
            "pdf_path": pdf_path,
            "page_image_paths": [page_path],
            "page_dimensions": [page_dim],
            "page_numbers": [page_no],
            "questions": q_dicts,
            "result": None,
            "retry_count": 0,
            "max_retry": max_retry,
            "error": "",
        }
        final = app.invoke(init_state)
        if final["result"] is None:
            err = f"page {page_no}: {final.get('error', 'unknown')}"
            page_errors.append(err)
            logger.warning("Segmentation failed for %s %s", pdf_path, err)
            continue
        page_results.append(SegmentationResult(**final["result"]))

    if not page_results:
        logger.error(
            "Page-by-page segmentation failed for %s after %d pages: %s",
            pdf_path,
            len(page_paths),
            "; ".join(page_errors) or "unknown",
        )
        return None
    return _merge_segmentation_results(page_results, expected_qids)


def _run_band_assignment_segmentation(
    cfg: AppConfig,
    q_dicts: list[dict],
    pdf_path: str,
    page_paths: list[str],
    page_dims: list[tuple[int, int]],
    expected_qids: list[str],
) -> SegmentationResult | None:
    """Ask the model to map deterministic candidate bands to qids."""
    llm_cfg = cfg.assignment_segmentation.llm
    pp_cfg = cfg.assignment_segmentation.postprocess
    prompt_dir = Path(llm_cfg.prompt_template).parent
    band_tpl_path = prompt_dir / "segmentation_band_assign.jinja"
    if not band_tpl_path.exists():
        logger.warning("Band assignment prompt not found: %s", band_tpl_path)
        return None
    band_tpl = Template(band_tpl_path.read_text(encoding="utf-8"))
    llm = build_llm(llm_cfg)

    canonical = student_canonical_stem(pdf_path)
    candidate_dir = get_answers_dir() / canonical / "_pages" / "candidates"
    page_results: list[SegmentationResult] = []
    page_errors: list[str] = []

    for page_idx, (page_path, page_dim) in enumerate(zip(page_paths, page_dims)):
        page_no = page_idx + 1
        with Image.open(page_path) as raw_img:
            page_img = raw_img.convert("RGB")
        try:
            candidates = _build_candidate_bands(page_img, page_no=page_no, pp_cfg=pp_cfg)
            if not candidates:
                page_results.append(SegmentationResult(questions=[]))
                continue
            overlay_path = candidate_dir / f"page_{page_no}_candidates.png"
            _save_candidate_overlay(page_img, candidates, overlay_path)
        finally:
            page_img.close()

        intro_text = (
            f"本次作业共有 {len(q_dicts)} 道题。以下每题先给出文字说明，"
            "再给出该题的题干图（若有）。"
        )
        segments: list[tuple[str, list[str]]] = [(intro_text, [])]
        for q in q_dicts:
            q_text = _format_question_block(q)
            q_images = [
                str(resolve_assignment_path(p)) for p in (q.get("question_images") or [])
                if p and resolve_assignment_path(p).exists()
            ]
            segments.append((q_text, q_images))

        candidate_text = band_tpl.render(
            page_no=page_no,
            width=page_dim[0],
            height=page_dim[1],
            candidates=candidates,
            expected_qids=expected_qids,
        )
        segments.append((candidate_text, [str(overlay_path)]))
        msg = build_vision_message_segmented(segments)

        parsed_result: SegmentationResult | None = None
        last_error = ""
        for attempt in range(max(1, cfg.assignment_segmentation.max_retry)):
            try:
                ctx = {
                    "stage": "segmentation-band-assign",
                    "pdf": Path(pdf_path).stem,
                    "page": page_no,
                    "attempt": attempt + 1,
                }
                resp = invoke_with_log(llm, [msg], ctx)
                raw = extract_json(resp.content)
                if raw is None:
                    last_error = "JSON parse failed"
                    continue
                parsed_result = _candidate_assignment_to_segmentation_result(raw, candidates, pp_cfg)
                break
            except Exception as e:
                last_error = str(e)
                logger.exception("Band assignment LLM call failed for %s page %s", pdf_path, page_no)
        if parsed_result is None:
            page_errors.append(f"page {page_no}: {last_error or 'unknown'}")
            continue
        page_results.append(parsed_result)

    if not page_results:
        logger.warning(
            "Band assignment segmentation failed for %s after %d pages: %s",
            pdf_path,
            len(page_paths),
            "; ".join(page_errors) or "unknown",
        )
        return None
    if page_errors:
        logger.warning(
            "Band assignment segmentation skipped failed pages for %s: %s",
            pdf_path,
            "; ".join(page_errors),
        )
    return _merge_segmentation_results(page_results, expected_qids)


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
    max_retry = cfg.assignment_segmentation.max_retry

    page_paths, page_dims = _render_and_cache_pages(pdf_path)
    expected_qids = [q.get("qid", "") for q in q_dicts if q.get("qid")]
    strategy = getattr(cfg.assignment_segmentation, "strategy", "global")
    seg_result: SegmentationResult | None = None
    segmentation_meta: dict[str, Any] = {
        "requested_strategy": strategy,
        "fallback_attempted": False,
        "fallback_used": False,
    }

    if strategy == "page_by_page":
        graph = build_segmentation_graph(cfg)
        seg_result = _run_page_by_page_segmentation(
            graph.compile(),
            pdf_path=pdf_path,
            q_dicts=q_dicts,
            page_paths=page_paths,
            page_dims=page_dims,
            max_retry=max_retry,
            expected_qids=expected_qids,
        )
        if seg_result is None:
            return (canonical, None)
    elif strategy == "band_assign":
        band_result = _run_band_assignment_segmentation(
            cfg,
            q_dicts,
            pdf_path,
            page_paths,
            page_dims,
            expected_qids,
        )
        if band_result is None:
            logger.warning("Falling back to page-by-page bbox segmentation for %s", pdf_path)
            segmentation_meta["fallback_attempted"] = True
            graph = build_segmentation_graph(cfg)
            seg_result = _run_page_by_page_segmentation(
                graph.compile(),
                pdf_path=pdf_path,
                q_dicts=q_dicts,
                page_paths=page_paths,
                page_dims=page_dims,
                max_retry=max_retry,
                expected_qids=expected_qids,
            )
            if seg_result is None:
                return (canonical, None)
            segmentation_meta["fallback_used"] = True
            segmentation_meta["fallback_mode"] = "replace_after_band_assign_failure"
            segmentation_meta["page_by_page_fallback"] = _segmentation_diagnostics(seg_result, expected_qids)
        else:
            band_diag = _segmentation_diagnostics(band_result, expected_qids)
            segmentation_meta["band_assign"] = band_diag
            seg_result = band_result
            should_try_fallback = (
                cfg.assignment_segmentation.band_assign_fill_missing_with_page_by_page
                and bool(band_diag["missing_qids"])
            )
            if should_try_fallback:
                logger.info(
                    "Band assignment left missing qids for %s: %s; trying page-by-page fallback",
                    pdf_path,
                    ", ".join(band_diag["missing_qids"]),
                )
                graph = build_segmentation_graph(cfg)
                segmentation_meta["fallback_attempted"] = True
                fallback_result = _run_page_by_page_segmentation(
                    graph.compile(),
                    pdf_path=pdf_path,
                    q_dicts=q_dicts,
                    page_paths=page_paths,
                    page_dims=page_dims,
                    max_retry=max_retry,
                    expected_qids=expected_qids,
                )
                if fallback_result is not None:
                    fallback_diag = _segmentation_diagnostics(fallback_result, expected_qids)
                    segmentation_meta["page_by_page_fallback"] = fallback_diag
                    min_ratio = cfg.assignment_segmentation.band_assign_min_assigned_ratio
                    if (
                        band_diag["assigned_ratio"] < min_ratio
                        and fallback_diag["assigned_ratio"] > band_diag["assigned_ratio"]
                    ):
                        seg_result = fallback_result
                        segmentation_meta["fallback_used"] = True
                        segmentation_meta["fallback_mode"] = "replace_sparse_band_assign"
                    else:
                        filled_result = _fill_missing_qids_from_fallback(
                            band_result,
                            fallback_result,
                            expected_qids,
                        )
                        filled_diag = _segmentation_diagnostics(filled_result, expected_qids)
                        if filled_diag["nonempty_qids"] > band_diag["nonempty_qids"]:
                            seg_result = filled_result
                            segmentation_meta["fallback_used"] = True
                            segmentation_meta["fallback_mode"] = "fill_missing_qids"
                            segmentation_meta["filled"] = filled_diag
                        else:
                            segmentation_meta["fallback_mode"] = "no_improvement"
                else:
                    segmentation_meta["fallback_mode"] = "page_by_page_failed"
    else:
        graph = build_segmentation_graph(cfg)
        app = graph.compile()
        init_state: SegState = {
            "pdf_path": pdf_path,
            "page_image_paths": page_paths,
            "page_dimensions": page_dims,
            "page_numbers": list(range(1, len(page_paths) + 1)),
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

    if seg_result is None:
        return (canonical, None)

    segmentation_meta["before_postprocess"] = _segmentation_diagnostics(seg_result, expected_qids)
    seg_result = _postprocess_segmentation_result(
        seg_result,
        page_dims,
        page_paths,
        cfg.assignment_segmentation.postprocess,
        expected_qids,
    )
    segmentation_meta["after_postprocess"] = _segmentation_diagnostics(seg_result, expected_qids)
    # 持久化 for_llm 坐标系下的 segment，供人工重新切分界面加载与保存
    cache_dir = get_answers_dir() / canonical / "_pages"
    segments_path = cache_dir / "segments.json"
    segments_data = {
        "dimensions": page_dims,
        "strategy": strategy,
        "diagnostics": segmentation_meta,
        "postprocess": cfg.assignment_segmentation.postprocess.model_dump(),
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
