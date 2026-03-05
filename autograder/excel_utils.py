"""Excel utilities – read student roster (.xls/.xlsx) and export graded results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import openpyxl
import xlrd


def read_student_roster(path: str | Path) -> list[dict[str, Any]]:
    """Read the student roster spreadsheet.

    Supports both .xls (xlrd) and .xlsx (openpyxl).
    Returns a list of dicts with keys from the header row.
    """
    path = Path(path)
    if path.suffix == ".xls":
        return _read_xls(path)
    return _read_xlsx(path)


def _read_xls(path: Path) -> list[dict[str, Any]]:
    wb = xlrd.open_workbook(str(path))
    ws = wb.sheet_by_index(0)
    headers = [str(ws.cell_value(0, c)).strip() for c in range(ws.ncols)]
    rows: list[dict[str, Any]] = []
    for r in range(1, ws.nrows):
        row = {}
        for c, h in enumerate(headers):
            val = ws.cell_value(r, c)
            if isinstance(val, float) and val == int(val):
                val = int(val)
            row[h] = val
        rows.append(row)
    return rows


def _read_xlsx(path: Path) -> list[dict[str, Any]]:
    wb = openpyxl.load_workbook(str(path), read_only=True)
    ws = wb.active
    data = list(ws.iter_rows(values_only=True))
    if not data:
        return []
    headers = [str(h).strip() for h in data[0]]
    rows: list[dict[str, Any]] = []
    for row_vals in data[1:]:
        row = {}
        for h, v in zip(headers, row_vals):
            row[h] = v
        rows.append(row)
    wb.close()
    return rows


HEADER_SCORE = "成绩（录入项）"
HEADER_COMMENT = "评语（录入项）"
HEADER_STUDENT_ID = "学号"


def export_graded_xlsx(
    roster_path: str | Path,
    results: dict[str, tuple[int, str]],
    out_path: str | Path,
) -> None:
    """Copy the roster and fill in scores/comments.

    *results* maps student_id -> (total_score, comment_text).
    """
    roster_path = Path(roster_path)
    out_path = Path(out_path)

    if roster_path.suffix == ".xls":
        _export_from_xls(roster_path, results, out_path)
    else:
        _export_from_xlsx(roster_path, results, out_path)


def _export_from_xls(src: Path, results: dict, dst: Path) -> None:
    """Read .xls with xlrd, write .xlsx with openpyxl."""
    rows = _read_xls(src)
    wb = openpyxl.Workbook()
    ws = wb.active
    if not rows:
        wb.save(str(dst))
        return

    headers = list(rows[0].keys())
    for c, h in enumerate(headers, 1):
        ws.cell(1, c, h)

    for r_idx, row in enumerate(rows, 2):
        for c, h in enumerate(headers, 1):
            ws.cell(r_idx, c, row.get(h, ""))
        sid = str(row.get(HEADER_STUDENT_ID, ""))
        if sid in results:
            score, comment = results[sid]
            score_col = headers.index(HEADER_SCORE) + 1 if HEADER_SCORE in headers else None
            comment_col = headers.index(HEADER_COMMENT) + 1 if HEADER_COMMENT in headers else None
            if score_col:
                ws.cell(r_idx, score_col, score)
            if comment_col:
                ws.cell(r_idx, comment_col, comment)
    wb.save(str(dst))


def _export_from_xlsx(src: Path, results: dict, dst: Path) -> None:
    wb = openpyxl.load_workbook(str(src))
    ws = wb.active
    headers = [str(c.value).strip() for c in ws[1]]

    sid_col = headers.index(HEADER_STUDENT_ID) + 1 if HEADER_STUDENT_ID in headers else None
    score_col = headers.index(HEADER_SCORE) + 1 if HEADER_SCORE in headers else None
    comment_col = headers.index(HEADER_COMMENT) + 1 if HEADER_COMMENT in headers else None

    if sid_col is None:
        wb.save(str(dst))
        return

    for row in ws.iter_rows(min_row=2):
        sid = str(row[sid_col - 1].value or "")
        if sid in results:
            score, comment = results[sid]
            if score_col:
                ws.cell(row[0].row, score_col, score)
            if comment_col:
                ws.cell(row[0].row, comment_col, comment)
    wb.save(str(dst))
