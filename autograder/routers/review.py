"""Router: manual review for low-confidence grading results."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from autograder.models import GradingRecord, StudentResult
from autograder.pipeline.grading import RESULTS_DIR, load_all_results

router = APIRouter(prefix="/api/review", tags=["review"])


@router.get("")
def get_review_items() -> list[dict]:
    """Return all grading records with confidence <= 2, grouped for review."""
    results = load_all_results()
    items = []
    for sr in results:
        for r in sr.records:
            if r.confidence <= 2:
                items.append({
                    "filename": sr.filename,
                    "student_id": sr.student_id,
                    "student_name": sr.student_name,
                    "qid": r.qid,
                    "answer": r.answer,
                    "grader": r.grader,
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
    path = RESULTS_DIR / f"{filename_stem}.json"
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
            updated = True
            break

    if not updated:
        return {"error": f"未找到 {qid} 的记录"}

    sr.total_score = sum(r.score for r in sr.records)
    path.write_text(sr.model_dump_json(indent=2), encoding="utf-8")
    return {"ok": True}
