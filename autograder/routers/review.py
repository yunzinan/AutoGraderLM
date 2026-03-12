"""Router: manual review for low-confidence grading results."""

from __future__ import annotations

import unicodedata
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from autograder.config import get_config, get_answers_dir, get_results_dir
from autograder.models import StudentResult, _now_iso
from autograder.pipeline.grading import load_all_results, load_student_result
from autograder.routers.questions import load_all_questions

router = APIRouter(prefix="/api/review", tags=["review"])


def _normalize_stem(s: str) -> str:
    """去掉 .pdf 后缀，与 results 文件名一致。"""
    return s.removesuffix(".pdf") if s.endswith(".pdf") else s


def _resolve_stem_dir(stem: str) -> Path | None:
    """解析 stem 对应的 answers 下实际目录（与 pipeline 一致，处理 Unicode 等）。"""
    stem = _normalize_stem(stem)
    answers_dir = get_answers_dir()
    if not answers_dir.exists():
        return None
    stem_nfc = unicodedata.normalize("NFC", stem)
    for d in answers_dir.iterdir():
        if d.is_dir() and not d.name.startswith("_") and unicodedata.normalize("NFC", d.name) == stem_nfc:
            return d
    return answers_dir / stem if (answers_dir / stem).exists() else None


def _get_answer_paths_from_disk(stem: str, qid: str) -> list[str]:
    """从磁盘 answers/{stem}/ 读取该题当前作答图片路径（持久化文件），不依赖 results JSON。
    人工重新切分后或服务重启后仍能显示最新切分结果。"""
    stem_dir = _resolve_stem_dir(stem)
    if not stem_dir or not stem_dir.is_dir():
        return []
    files = sorted(stem_dir.glob(f"{qid}-*.png"), key=lambda f: f.name)
    return [f"answers/{stem_dir.name}/{f.name}" for f in files]


@router.get("/item/{filename_stem}/{qid}")
def get_review_item(filename_stem: str, qid: str) -> dict:
    """获取指定学生指定题目的评阅记录，用于人工复核（不论置信度）。
    作答图片路径始终从磁盘 answers 目录读取，保证人工重新切分后、服务重启后仍为最新。"""
    filename_stem = _normalize_stem(filename_stem)
    sr = load_student_result(filename_stem)
    if not sr:
        return {"error": "未找到该学生的评阅结果"}
    for r in sr.records:
        if r.qid == qid:
            answer_paths = _get_answer_paths_from_disk(filename_stem, qid)
            return {
                "filename": sr.filename,
                "student_id": sr.student_id,
                "student_name": sr.student_name,
                "qid": r.qid,
                "answer": answer_paths,
                "grader": r.grader,
                "graded_at": r.graded_at,
                "score": r.score,
                "confidence": r.confidence,
                "summary": r.summary,
                "comments": r.comments,
            }
    return {"error": f"未找到 {qid} 的评阅记录"}


def _should_include_in_review(record_score: int, qid: str, confidence: int, question_scores: dict[str, int], ratio: float) -> bool:
    """是否纳入人工复核：置信度 <= 2，或得分 <= ratio * 题目总分。"""
    if confidence <= 2:
        return True
    full_score = question_scores.get(qid)
    if full_score is None or full_score <= 0:
        return False
    threshold = ratio * full_score
    return record_score <= threshold


@router.get("")
def get_review_items() -> list[dict]:
    """返回需人工复核的评阅记录：置信度 <= 2，或得分 <= (add_to_regrade_when_below * 题目总分)。
    全量评分完成后，低分作答会自动出现在本列表中。作答图片路径从磁盘 answers 目录读取。"""
    results = load_all_results()
    questions = load_all_questions()
    question_scores = {q.qid: q.score for q in questions}
    cfg = get_config()
    ratio = cfg.assignment_regrade.add_to_regrade_when_below
    items = []
    stem_norm = _normalize_stem
    for sr in results:
        file_stem = stem_norm(Path(sr.filename).stem)
        for r in sr.records:
            if _should_include_in_review(r.score, r.qid, r.confidence, question_scores, ratio):
                answer_paths = _get_answer_paths_from_disk(file_stem, r.qid)
                items.append({
                    "filename": sr.filename,
                    "student_id": sr.student_id,
                    "student_name": sr.student_name,
                    "qid": r.qid,
                    "answer": answer_paths,
                    "grader": r.grader,
                    "graded_at": r.graded_at,
                    "score": r.score,
                    "confidence": r.confidence,
                    "summary": r.summary,
                    "comments": r.comments,
                })
    return items


class ReviewUpdate(BaseModel):
    score: int
    confidence: int
    summary: str = ""
    comments: str = ""


@router.put("/{filename_stem}/{qid}")
def update_review(filename_stem: str, qid: str, body: ReviewUpdate) -> dict:
    """Teacher manually updates a grading record."""
    filename_stem = _normalize_stem(filename_stem)
    path = get_results_dir() / f"{filename_stem}.json"
    if not path.exists():
        return {"error": "结果文件不存在"}

    sr = StudentResult.model_validate_json(path.read_text(encoding="utf-8"))
    updated = False
    for r in sr.records:
        if r.qid == qid:
            r.score = body.score
            r.confidence = body.confidence
            r.summary = body.summary
            r.comments = body.comments
            r.grader = "human"
            r.graded_at = _now_iso()
            updated = True
            break

    if not updated:
        return {"error": f"未找到 {qid} 的记录"}

    sr.total_score = sum(r.score for r in sr.records)
    path.write_text(sr.model_dump_json(indent=2), encoding="utf-8")
    return {"ok": True}
