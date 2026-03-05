"""Router: score statistics, roster table, and question reports."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

from autograder.config import get_config
from autograder.excel_utils import read_student_roster
from autograder.pipeline.grading import load_all_results
from autograder.pipeline.report import compute_score_distribution

router = APIRouter(prefix="/api/stats", tags=["stats"])


@router.get("/roster")
def get_roster() -> list[dict]:
    """Return the student roster (with scores if exported)."""
    cfg = get_config()
    out_path = Path(cfg.assignment_report.xlsx_out_path)
    if out_path.exists():
        return read_student_roster(out_path)
    in_path = cfg.assignment_configuration.xlsx_in_path
    if Path(in_path).exists():
        return read_student_roster(in_path)
    return []


@router.get("/distribution/{qid}")
def get_score_distribution(qid: str) -> dict:
    results = load_all_results()
    dist = compute_score_distribution(results, qid)
    return {"qid": qid, "distribution": dist}


@router.get("/export", response_model=None)
def download_export():
    cfg = get_config()
    out_path = Path(cfg.assignment_report.xlsx_out_path)
    if not out_path.exists():
        return {"error": "导出文件不存在，请先生成报告"}
    return FileResponse(
        path=str(out_path),
        filename=out_path.name,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@router.get("/summary")
def get_summary() -> dict:
    """Overall grading summary."""
    results = load_all_results()
    if not results:
        return {"total_students": 0}
    scores = [sr.total_score for sr in results]
    return {
        "total_students": len(results),
        "avg_score": round(sum(scores) / len(scores), 1) if scores else 0,
        "max_score": max(scores) if scores else 0,
        "min_score": min(scores) if scores else 0,
    }
