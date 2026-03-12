"""Router: score statistics, roster table, question reports, and assignment overview."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

from autograder.config import get_config, get_answers_dir, get_results_dir
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


def _count_pdfs() -> int:
    """pdf_folder_path 下 PDF 文件数量，表示已提交的作业份数。"""
    cfg = get_config()
    folder = Path(cfg.assignment_configuration.pdf_folder_path)
    if not folder.exists():
        return 0
    count = 0
    for ext in ("*.pdf", "*.PDF"):
        count += len(list(folder.glob(ext)))
    return count


def _count_expected_from_excel() -> int:
    """excel_in_path 中数据行数，表示应交作业份数。"""
    cfg = get_config()
    in_path = Path(cfg.assignment_configuration.excel_in_path)
    if not in_path.exists():
        return 0
    try:
        rows = read_student_roster(in_path)
        return len(rows)
    except Exception:
        return 0


def _count_segmented() -> int:
    """answers/ 下有效作业目录数（含 _pages/for_llm 或 segments），表示已完成切分的份数。"""
    answers_dir = get_answers_dir()
    if not answers_dir.exists():
        return 0
    count = 0
    for sub in answers_dir.iterdir():
        if not sub.is_dir() or sub.name.startswith("_"):
            continue
        if (sub / "_pages" / "for_llm").exists() or (sub / "_pages" / "segments.json").exists():
            count += 1
    return count


def _count_graded() -> int:
    """results/ 下学生结果 JSON 数量（排除 question_reports.json），表示已完成评阅的份数。"""
    results_dir = get_results_dir()
    if not results_dir.exists():
        return 0
    return sum(
        1 for p in results_dir.glob("*.json")
        if p.name != "question_reports.json"
    )


def _list_assignments_with_status() -> list[dict]:
    """列出所有已提交作业（以 PDF 为准）及其状态：未切分、已切分未评审、已评审。"""
    cfg = get_config()
    pdf_folder = Path(cfg.assignment_configuration.pdf_folder_path)
    answers_dir = get_answers_dir()
    results_dir = get_results_dir()

    stems: set[str] = set()
    for ext in ("*.pdf", "*.PDF"):
        for p in pdf_folder.glob(ext):
            stems.add(p.stem)

    out = []
    for stem in sorted(stems):
        has_answer = (
            answers_dir.exists()
            and (answers_dir / stem).is_dir()
            and (
                (answers_dir / stem / "_pages" / "for_llm").exists()
                or (answers_dir / stem / "_pages" / "segments.json").exists()
            )
        )
        has_result = (
            results_dir.exists()
            and (results_dir / f"{stem}.json").exists()
        )
        if has_result:
            status = "已评审"
        elif has_answer:
            status = "已切分未评审"
        else:
            status = "未切分"
        out.append({"stem": stem, "label": stem, "status": status})
    return out


@router.get("/overview")
def get_overview() -> dict:
    """作业概览：应交份数、已提交份数、已切分份数、已评阅份数。"""
    return {
        "expected_count": _count_expected_from_excel(),
        "submitted_count": _count_pdfs(),
        "segmented_count": _count_segmented(),
        "graded_count": _count_graded(),
    }


@router.get("/overview/assignments")
def get_overview_assignments() -> list[dict]:
    """每份作业的状态列表，用于增量切分/评阅与重新执行。"""
    return _list_assignments_with_status()


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
