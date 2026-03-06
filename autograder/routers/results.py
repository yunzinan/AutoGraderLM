"""Router: view grading results per student."""

from __future__ import annotations

from fastapi import APIRouter

from autograder.config import to_relative_url_path
from autograder.models import StudentResult
from autograder.pdf_utils import parse_student_info
from autograder.pipeline.grading import load_all_results, load_student_result

router = APIRouter(prefix="/api/results", tags=["results"])


@router.get("/students")
def list_students() -> list[dict]:
    """Return list of students with basic info."""
    results = load_all_results()
    return [
        {
            "filename": sr.filename,
            "filename_stem": sr.filename.rsplit(".", 1)[0],
            "student_id": sr.student_id,
            "student_name": sr.student_name,
            "total_score": sr.total_score,
        }
        for sr in results
    ]


@router.get("/{filename_stem}")
def get_student_result(filename_stem: str) -> StudentResult | dict:
    sr = load_student_result(filename_stem)
    if sr is None:
        return {"error": "未找到结果"}
    data = sr.model_dump()
    for r in data.get("records", []):
        r["answer"] = [to_relative_url_path(p) for p in (r.get("answer") or [])]
    return data
