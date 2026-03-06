"""Router: score statistics, roster table, and question reports."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

from autograder.config import get_config
from autograder.excel_utils import (
    HEADER_COMMENT,
    HEADER_SCORE,
    HEADER_STUDENT_ID,
    get_roster_headers,
    read_student_roster,
    write_roster_to_xls,
)
from autograder.pipeline.grading import load_all_results
from autograder.pipeline.report import build_student_comment, compute_score_distribution

router = APIRouter(prefix="/api/stats", tags=["stats"])


def _roster_with_live_scores() -> list[dict]:
    """以持久化评阅结果（results 目录）为数据源构建 roster，不论是否有人工复核。
    每个有结果文件的学生一行；若配置了 xlsx，则合并 xlsx 中的其他列（如姓名等）。
    """
    cfg = get_config()
    in_path = Path(cfg.assignment_configuration.excel_in_path)
    results = load_all_results()

    # 以 results 为数据源：每个学生结果对应一行
    xlsx_by_id: dict[str, dict] = {}
    if in_path.exists():
        for row in read_student_roster(in_path):
            sid = str(row.get(HEADER_STUDENT_ID, "") or row.get("学号", "")).strip()
            if sid:
                xlsx_by_id[sid] = dict(row)

    roster = []
    for sr in results:
        sid = str(sr.student_id).strip()
        row = {
            HEADER_STUDENT_ID: sr.student_id,
            "姓名": sr.student_name,
            HEADER_SCORE: sr.total_score,
            HEADER_COMMENT: build_student_comment(sr.records),
        }
        if sid in xlsx_by_id:
            row.update(xlsx_by_id[sid])
            row[HEADER_STUDENT_ID] = sr.student_id
            row[HEADER_SCORE] = sr.total_score
            row[HEADER_COMMENT] = build_student_comment(sr.records)
        roster.append(row)
    return roster


def _is_roster_key_hidden(k: str) -> bool:
    k = (k or "").strip()
    k_lower = k.lower()
    return "作业id" in k_lower or "提交作业状态" in k


def _filter_roster_for_display(roster: list[dict]) -> list[dict]:
    """移除网页表格不展示的列（学生作业ID、作业ID、提交作业状态），导出仍用完整数据。"""
    hidden = {k for row in roster for k in row if _is_roster_key_hidden(k)}
    if not hidden:
        return roster
    return [{k: v for k, v in row.items() if k not in hidden} for row in roster]


@router.get("/roster")
def get_roster() -> list[dict]:
    """Return the student roster with live scores/comments from current results."""
    roster = _roster_with_live_scores()
    return _filter_roster_for_display(roster)


@router.get("/distribution/{qid}")
def get_score_distribution(qid: str) -> dict:
    results = load_all_results()
    dist = compute_score_distribution(results, qid)
    return {"qid": qid, "distribution": dist}


@router.get("/export", response_model=None)
def download_export():
    """以 in.xls 为模板：列与行顺序与 in.xls 完全一致，仅在有评阅结果时填入成绩、评语，写出为 .xls。"""
    cfg = get_config()
    in_path = Path(cfg.assignment_configuration.excel_in_path)
    out_path = Path(cfg.assignment_report.excel_out_path).with_suffix(".xls")
    if not in_path.exists():
        roster = _roster_with_live_scores()
        headers = get_roster_headers(in_path)
        write_roster_to_xls(roster, headers, out_path)
    else:
        roster = read_student_roster(in_path)
        headers = list(roster[0].keys()) if roster else [HEADER_STUDENT_ID, "姓名", HEADER_SCORE, HEADER_COMMENT]
        results = load_all_results()
        score_map = {
            str(sr.student_id).strip(): (sr.total_score, build_student_comment(sr.records))
            for sr in results
        }
        for row in roster:
            sid = str(row.get(HEADER_STUDENT_ID, "") or row.get("学号", "")).strip()
            if sid in score_map:
                row[HEADER_SCORE], row[HEADER_COMMENT] = score_map[sid]
        write_roster_to_xls(roster, headers, out_path)
    return FileResponse(
        path=str(out_path),
        filename=out_path.name,
        media_type="application/vnd.ms-excel",
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
