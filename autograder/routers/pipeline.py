"""Router: pipeline execution (segmentation & grading)."""

from __future__ import annotations

import asyncio
import glob
import json
import logging
import re
import shutil
import subprocess
import tempfile
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, Body, Response

from autograder.config import get_config, get_answers_dir, get_results_dir, resolve_assignment_path
from autograder.path_utils import is_safe_path_segment
from autograder.pdf_utils import student_canonical_stem
from autograder.models import PipelineStatus, QuestionReport
from autograder.pipeline.grading import (
    _grade_one_sync,
    _save_student_result,
    build_grading_graph,
    load_all_results,
    load_student_result,
    run_grading,
)
from autograder.pipeline.report import generate_all_reports, generate_question_report_sync, run_report_sync
from autograder.pipeline.segmentation import (
    apply_manual_segments,
    load_segments_for_stem,
    run_segmentation,
)
from autograder.routers.questions import load_all_questions

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

_PAGE_FILE_RE = re.compile(r"^page_(\d+)\.png$", re.IGNORECASE)
_ANSWER_FILE_RE = re.compile(r"^(?P<qid>.+)-(?P<idx>\d+)\.png$", re.IGNORECASE)


def _page_file_sort_key(path: Path) -> tuple[int, str]:
    """按 page_数字.png 的数字序号排序，避免 page_10 排到 page_2 前。"""
    match = _PAGE_FILE_RE.match(path.name)
    if match:
        return (int(match.group(1)), path.name)
    return (10**9, path.name)


def _answer_file_sort_key(path: Path) -> tuple[str, int, str]:
    """按 qid + 序号排序，避免 Q1-10.png 在 Q1-2.png 前。"""
    match = _ANSWER_FILE_RE.match(path.name)
    if match:
        return (match.group("qid"), int(match.group("idx")), path.name)
    return (path.stem, 10**9, path.name)


def _reports_dir() -> Path:
    """报告目录：results/reports/，下含 Q1/report.md、Q1/reference.md 等"""
    return get_results_dir() / "reports"


def _reports_meta_path() -> Path:
    return _reports_dir() / "reports_meta.json"


def _legacy_reports_json_path() -> Path:
    """旧格式：question_reports.json（用于迁移）"""
    return get_results_dir() / "question_reports.json"


def _build_answer_map_from_disk() -> dict[str, dict[str, list[str]]]:
    """从 answers 目录恢复 answer_map（服务重启后或未先点「开始切分」时可用）。"""
    result: dict[str, dict[str, list[str]]] = {}
    answers_dir = get_answers_dir()
    if not answers_dir.exists():
        return result
    for stem_dir in sorted(answers_dir.iterdir()):
        if not stem_dir.is_dir() or stem_dir.name.startswith("_"):
            continue
        stem = stem_dir.name
        by_qid: dict[str, list[str]] = {}
        for f in sorted(stem_dir.glob("*.png"), key=_answer_file_sort_key):
            if not f.is_file():
                continue
            # 文件名格式 Q1-0.png, Q2-1.png -> qid 为 Q1, Q2
            parts = f.stem.split("-", 1)
            qid = parts[0] if parts else f.stem
            path_str = str(resolve_assignment_path(f"answers/{stem}/{f.name}"))
            by_qid.setdefault(qid, []).append(path_str)
        if by_qid:
            result[stem] = by_qid
    return result


_status = PipelineStatus()
_status_lock = threading.Lock()
_answer_map: dict[str, dict[str, list[str]]] = {}
_reports: list[QuestionReport] = []
_running_task: asyncio.Task | None = None
_report_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="report")


def _anonymize_text(text: str, names: list[str]) -> str:
    """将文本中出现的学生姓名统一替换为 ***。"""
    if not text:
        return ""
    cleaned = [n for n in names if n]
    if not cleaned:
        return text
    escaped_names = sorted((re.escape(n) for n in cleaned), key=len, reverse=True)
    pattern = re.compile("|".join(escaped_names))
    return pattern.sub("***", text)


def _anonymize_text_with_token(text: str, names: list[str], token: str) -> str:
    """将文本中出现的学生姓名替换为指定 token（用于后续安全渲染）。"""
    if not text:
        return ""
    cleaned = [n for n in names if n]
    if not cleaned:
        return text
    escaped_names = sorted((re.escape(n) for n in cleaned), key=len, reverse=True)
    pattern = re.compile("|".join(escaped_names))
    return pattern.sub(token, text)


def _strip_outer_markdown_fence(text: str) -> str:
    """去掉最外层 markdown 代码围栏。"""
    if not text:
        return ""
    out = text.strip().replace("\r\n", "\n")
    fence = re.match(r"^```(?:markdown|md)?\s*\n([\s\S]*?)\n```\s*$", out)
    return fence.group(1).strip() if fence else out


def _normalize_latex_artifacts(text: str) -> str:
    """修正常见的 LaTeX 转义污染（如 \\{}begin / \\{}[ / \\{}Theta）。"""
    if not text:
        return ""
    out = text
    # 原始 `\\` 常被污染为 `\{}\{}`
    out = out.replace(r"\{}\{}", r"\\")
    # 常见污染：`\{}` 作为反斜杠占位符，恢复为 `\`
    out = out.replace(r"\{}", "\\")
    # 清理无效单反斜杠（如 `\ `、`\,` 等被污染残留）
    out = re.sub(r"(?<!\\)\\(?=\s)", "", out)
    out = re.sub(r"(?<!\\)\\(?=[,，。；：!！?？])", "", out)
    # 归一化行尾空白
    out = out.replace("\r\n", "\n")
    return out


def _protect_math_segments(text: str) -> tuple[str, dict[str, str]]:
    """保护 LaTeX 数学公式片段，避免被文本转义破坏。"""
    tokens: dict[str, str] = {}
    pattern = re.compile(
        r"\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|(?<!\$)\$(?!\$)(?:\\.|[^$\\\n])+(?<!\\)\$(?!\$)",
        re.MULTILINE,
    )

    def _repl(match: re.Match) -> str:
        key = f"AUTOMATHTOKEN{len(tokens)}"
        tokens[key] = match.group(0)
        return key

    return pattern.sub(_repl, text), tokens


def _escape_latex_text(text: str) -> str:
    """转义普通文本中的 LaTeX 特殊字符。"""
    out = text
    out = out.replace("\\", r"\textbackslash{}")
    out = out.replace("&", r"\&")
    out = out.replace("%", r"\%")
    out = out.replace("#", r"\#")
    out = out.replace("_", r"\_")
    out = out.replace("{", r"\{")
    out = out.replace("}", r"\}")
    out = out.replace("~", r"\textasciitilde{}")
    out = out.replace("^", r"\textasciicircum{}")
    return out


def _restore_tokens(text: str, tokens: dict[str, str], key_escaped: bool = False) -> str:
    """安全恢复占位符，按 key 长度倒序避免 TOKEN1 误替换 TOKEN10。"""
    out = text
    for key in sorted(tokens.keys(), key=len, reverse=True):
        lookup = _escape_latex_text(key) if key_escaped else key
        out = out.replace(lookup, tokens[key])
    return out


def _inline_markdown_to_latex(text: str) -> str:
    """将行内 markdown 片段转为 latex。"""
    if not text:
        return ""
    normalized = _normalize_latex_artifacts(text)
    protected, math_tokens = _protect_math_segments(normalized)

    code_tokens: dict[str, str] = {}

    def _code_repl(match: re.Match) -> str:
        key = f"@@CODE_{len(code_tokens)}@@"
        code_tokens[key] = r"\texttt{" + _escape_latex_text(match.group(1)) + "}"
        return key

    protected = re.sub(r"`([^`]+)`", _code_repl, protected)
    out = _escape_latex_text(protected)
    out = _restore_tokens(out, code_tokens, key_escaped=True)
    out = _restore_tokens(out, math_tokens, key_escaped=True)
    out = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", out)
    out = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\\emph{\1}", out)
    return out


def _markdown_to_latex(text: str) -> str:
    """将常见 markdown 结构转为 latex（保留数学公式）。"""
    src = _normalize_latex_artifacts(_strip_outer_markdown_fence(text))
    if not src:
        return ""
    # 先全段保护数学公式，避免跨行公式在逐行处理中被转义破坏
    src, global_math_tokens = _protect_math_segments(src)

    lines = src.split("\n")
    out_lines: list[str] = []
    list_mode: str | None = None
    in_code = False
    code_lines: list[str] = []

    def _close_list():
        nonlocal list_mode
        if list_mode == "itemize":
            out_lines.append(r"\end{itemize}")
        elif list_mode == "enumerate":
            out_lines.append(r"\end{enumerate}")
        list_mode = None

    def _open_list(mode: str):
        nonlocal list_mode
        if list_mode == mode:
            return
        _close_list()
        list_mode = mode
        if mode == "itemize":
            out_lines.append(r"\begin{itemize}[leftmargin=1.8em,itemsep=2pt]")
        else:
            out_lines.append(r"\begin{enumerate}[leftmargin=2.0em,itemsep=2pt]")

    for line in lines:
        if re.match(r"^\s*```", line):
            if not in_code:
                _close_list()
                in_code = True
                code_lines = []
            else:
                in_code = False
                out_lines.append(r"\begin{verbatim}")
                out_lines.extend(code_lines)
                out_lines.append(r"\end{verbatim}")
            continue

        if in_code:
            code_lines.append(line)
            continue

        if not line.strip():
            _close_list()
            out_lines.append("")
            continue

        m_head = re.match(r"^\s*(#{1,6})\s+(.+?)\s*$", line)
        if m_head:
            _close_list()
            level = len(m_head.group(1))
            title = _inline_markdown_to_latex(m_head.group(2))
            if level <= 1:
                out_lines.append(rf"\subsection*{{{title}}}")
            elif level == 2:
                out_lines.append(rf"\subsubsection*{{{title}}}")
            else:
                out_lines.append(rf"\paragraph{{{title}}}")
            continue

        m_ul = re.match(r"^\s*[-*+]\s+(.+?)\s*$", line)
        if m_ul:
            _open_list("itemize")
            out_lines.append(r"\item " + _inline_markdown_to_latex(m_ul.group(1)))
            continue

        m_ol = re.match(r"^\s*\d+\.\s+(.+?)\s*$", line)
        if m_ol:
            _open_list("enumerate")
            out_lines.append(r"\item " + _inline_markdown_to_latex(m_ol.group(1)))
            continue

        _close_list()
        out_lines.append(_inline_markdown_to_latex(line))

    if in_code:
        out_lines.append(r"\begin{verbatim}")
        out_lines.extend(code_lines)
        out_lines.append(r"\end{verbatim}")
    _close_list()
    out = "\n".join(out_lines).strip()
    out = _restore_tokens(out, global_math_tokens, key_escaped=False)
    return out


def _latex_include_graphics(path: Path, width: str = r"0.72\linewidth") -> str:
    """生成单张图片的 latex 插图语句。"""
    p = str(path.resolve()).replace("\\", "/")
    return rf"\includegraphics[width={width}]{{\detokenize{{{p}}}}}"


def _render_report_markdown_block(text: str, names: list[str], anonymize: bool) -> str:
    """报告 markdown 段落渲染（支持匿名 token 防止 *** 被 markdown 吃掉）。"""
    anon_token = "AUTOGRADERANONMARKER"
    src = text or ""
    if anonymize:
        src = _anonymize_text_with_token(src, names, anon_token)
    out = _markdown_to_latex(src)
    if anonymize:
        out = out.replace(anon_token, "***")
    return out


def _build_reports_tex(
    reports: list[QuestionReport],
    questions: list,
    assignment_name: str,
    anonymize: bool,
) -> str:
    """组装答题报告 latex 源码。"""
    q_map = {q.qid: q for q in questions}
    date_text = f"{datetime.now().year}年{datetime.now().month}月{datetime.now().day}日"
    title_text = assignment_name or "Assignment"

    parts: list[str] = [
        r"\documentclass[12pt]{article}",
        r"\usepackage[a4paper,margin=2.2cm]{geometry}",
        r"\usepackage{fontspec}",
        r"\usepackage[UTF8,fontset=fandol]{ctex}",
        r"\usepackage{amsmath,amssymb}",
        r"\usepackage{graphicx}",
        r"\usepackage{hyperref}",
        r"\usepackage{enumitem}",
        r"\usepackage{titlesec}",
        r"\usepackage{fancyhdr}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{6pt}",
        r"\titleformat{\section}{\Large\bfseries}{}{0pt}{}",
        r"\titleformat{\subsection}{\large\bfseries}{}{0pt}{}",
        r"\titleformat{\subsubsection}{\normalsize\bfseries}{}{0pt}{}",
        r"\pagestyle{fancy}",
        r"\fancyhf{}",
        r"\chead{}",
        rf"\rhead{{{_inline_markdown_to_latex(date_text)}}}",
        r"\lhead{AutoGraderLM}",
        r"\cfoot{\thepage}",
        r"\begin{document}",
        r"\begin{center}",
        rf"{{\LARGE\bfseries {_inline_markdown_to_latex(title_text)}}}",
        r"\end{center}",
        r"\vspace{0.6em}",
    ]

    if not reports:
        parts.append("暂无可导出的报告内容。")
    else:
        for idx, rpt in enumerate(reports):
            question = q_map.get(rpt.qid)
            qidx = rpt.question_index
            qtitle = f"{rpt.qid}"
            if qidx is not None:
                qtitle = f"{rpt.qid}（习题集序号 {qidx}）"

            parts.append(rf"\section*{{{_inline_markdown_to_latex(qtitle)}}}")

            if question and (question.question_text or question.question_images):
                parts.append(r"\subsection*{题目}")
                if question.question_text:
                    parts.append(_markdown_to_latex(question.question_text))
                if question.question_images:
                    parts.append(r"\begin{center}")
                    for img in question.question_images:
                        abs_img = resolve_assignment_path(img)
                        if not abs_img.exists():
                            continue
                        parts.append(_latex_include_graphics(abs_img))
                        parts.append(r"\\[0.8em]")
                    parts.append(r"\end{center}")

            if question and question.rubric:
                parts.append(r"\subsection*{评分标准}")
                parts.append(_markdown_to_latex(question.rubric))

            dist = rpt.score_distribution or {}
            dist_text = "无"
            if dist:
                items = sorted(dist.items(), key=lambda x: x[0])
                dist_text = "；".join(f"{score}分：{count}人" for score, count in items)
            parts.append(r"\subsection*{得分分布}")
            parts.append(_inline_markdown_to_latex(dist_text))

            names = list((rpt.student_answers or {}).keys())

            parts.append(r"\subsection*{答题报告}")
            parts.append(_render_report_markdown_block(rpt.report_text or "", names, anonymize))

            ref_text = rpt.reference_answer or ""
            if ref_text.strip():
                parts.append(r"\subsection*{参考作答}")
                parts.append(_render_report_markdown_block(ref_text, names, anonymize))

            if idx != len(reports) - 1:
                parts.append(r"\newpage")

    parts.append(r"\end{document}")
    return "\n".join(parts)


def _compile_latex_to_pdf_bytes(tex_content: str) -> bytes:
    """使用 xelatex 编译 latex 并返回 PDF 字节。"""
    if not shutil.which("xelatex"):
        raise RuntimeError("系统未安装 xelatex，无法导出高质量 PDF")

    with tempfile.TemporaryDirectory(prefix="autograder_report_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        tex_path = tmp_path / "report.tex"
        pdf_path = tmp_path / "report.pdf"
        tex_path.write_text(tex_content, encoding="utf-8")

        proc = subprocess.run(
            ["xelatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0 or not pdf_path.exists():
            err = (proc.stderr or proc.stdout or "").strip()
            if len(err) > 2000:
                err = err[-2000:]
            raise RuntimeError(f"xelatex 编译失败: {err or 'unknown error'}")
        return pdf_path.read_bytes()


def _build_reports_pdf_bytes(
    reports: list[QuestionReport],
    questions: list,
    anonymize: bool,
) -> bytes:
    """构建仅包含答题报告内容的 PDF 二进制（XeLaTeX 渲染）。"""
    cfg = get_config()
    tex = _build_reports_tex(reports, questions, cfg.assignment_name, anonymize)
    return _compile_latex_to_pdf_bytes(tex)


def _load_reports_from_disk() -> list[QuestionReport]:
    """从磁盘加载报告：优先新格式（markdown + meta），否则从旧 question_reports.json 迁移。"""
    meta_path = _reports_meta_path()
    legacy_path = _legacy_reports_json_path()

    # 1. 新格式：reports/reports_meta.json + reports/{qid}/report.md, reference.md
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            reports_dir = _reports_dir()
            out: list[QuestionReport] = []
            for d in meta:
                qid = d.get("qid", "")
                qdir = reports_dir / qid
                report_text = (qdir / "report.md").read_text(encoding="utf-8") if (qdir / "report.md").exists() else ""
                ref_path = qdir / "reference.md"
                reference_answer = ref_path.read_text(encoding="utf-8") if ref_path.exists() else ""
                dist = d.get("score_distribution") or {}
                out.append(QuestionReport(
                    qid=qid,
                    question_index=d.get("question_index"),
                    score_distribution={int(k): v for k, v in dist.items()},
                    report_text=report_text,
                    reference_answer=reference_answer,
                    student_answers=d.get("student_answers") or {},
                ))
            return out
        except Exception as e:
            logger.warning("Failed to load reports from %s: %s", meta_path, e)
            return []

    # 2. 旧格式：迁移并保存为新格式
    if legacy_path.exists():
        try:
            raw = json.loads(legacy_path.read_text(encoding="utf-8"))
            out: list[QuestionReport] = []
            for d in raw:
                dist = d.get("score_distribution") or {}
                d = dict(d)
                d["score_distribution"] = {int(k): v for k, v in dist.items()}
                out.append(QuestionReport.model_validate(d))
            if out:
                _save_reports_to_disk(out)
                logger.info("Migrated reports from legacy JSON to markdown format")
            return out
        except Exception as e:
            logger.warning("Failed to load/migrate from %s: %s", legacy_path, e)
            return []

    return []


def _save_reports_to_disk(reports: list[QuestionReport]) -> None:
    """持久化报告为 markdown 文件 + 元数据 JSON。"""
    try:
        reports_dir = _reports_dir()
        reports_dir.mkdir(parents=True, exist_ok=True)
        meta: list[dict] = []
        for r in reports:
            qdir = reports_dir / r.qid
            qdir.mkdir(parents=True, exist_ok=True)
            (qdir / "report.md").write_text(r.report_text or "", encoding="utf-8")
            (qdir / "reference.md").write_text(r.reference_answer or "", encoding="utf-8")
            meta.append({
                "qid": r.qid,
                "question_index": r.question_index,
                "score_distribution": r.score_distribution,
                "student_answers": r.student_answers,
            })
        _reports_meta_path().write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning("Failed to save reports to %s: %s", _reports_dir(), e)


def _update_status_sync(msg: str, current: int, total: int) -> None:
    """线程安全地更新 _status（供报告线程调用）。"""
    with _status_lock:
        _status.message = msg
        _status.current = current
        _status.total = total


@router.get("/status")
def get_status() -> PipelineStatus:
    """返回当前 pipeline 状态（线程安全读）。"""
    with _status_lock:
        return PipelineStatus(
            stage=_status.stage,
            message=_status.message,
            current=_status.current,
            total=_status.total,
            errors=list(_status.errors),
        )


def _list_pdfs() -> list[str]:
    cfg = get_config()
    folder = cfg.assignment_configuration.pdf_folder_path
    paths: list[str] = []
    for ext in ("*.pdf", "*.PDF"):
        paths.extend(glob.glob(str(Path(folder) / ext)))
    return sorted(paths)


def _list_unsegmented_pdfs() -> list[str]:
    """仅返回尚未切分的 PDF（answers 下无该学生规范目录 学号_姓名 的 for_llm 或 segments）。"""
    all_pdfs = _list_pdfs()
    answers_dir = get_answers_dir()
    if not answers_dir.exists():
        return all_pdfs
    out = []
    for p in all_pdfs:
        canonical = student_canonical_stem(p)
        sub = answers_dir / canonical
        if not sub.is_dir():
            out.append(p)
            continue
        if (sub / "_pages" / "for_llm").exists() or (sub / "_pages" / "segments.json").exists():
            continue
        out.append(p)
    return out


def _filter_answer_map_ungraded(answer_map: dict[str, dict[str, list[str]]]) -> dict[str, dict[str, list[str]]]:
    """仅保留尚未评阅的 stem（results 下无该 stem 的 json）。"""
    results_dir = get_results_dir()
    if not results_dir.exists():
        return dict(answer_map)
    return {
        stem: by_q for stem, by_q in answer_map.items()
        if not (results_dir / f"{stem}.json").exists()
    }


async def _update_status(msg: str, current: int, total: int):
    _status.message = msg
    _status.current = current
    _status.total = total


@router.post("/segment")
async def start_segmentation(body: dict | None = Body(default=None)) -> dict:
    """全量或增量切分。body 可含 "incremental": true，仅处理尚未切分的 PDF。"""
    global _running_task, _answer_map
    if _status.stage not in ("idle", "done", "error"):
        return {"error": f"Pipeline already running: {_status.stage}"}

    incremental = bool((body or {}).get("incremental"))
    questions = load_all_questions()
    pdfs = _list_unsegmented_pdfs() if incremental else _list_pdfs()
    num_pdfs = len(pdfs)
    if not num_pdfs:
        return {
            "error": "增量切分时未找到待切分作业" if incremental else "未找到 PDF 文件",
            "num_pdfs": 0,
        }

    _status.stage = "segmentation"
    _status.errors = []
    _status.total = num_pdfs
    _status.current = 0
    _status.message = f"即将{'增量' if incremental else ''}切分 {num_pdfs} 份作业…"

    async def _run():
        global _answer_map
        try:
            cfg = get_config()
            new_map = await run_segmentation(cfg, questions, pdfs, on_progress=_update_status)
            _answer_map = _build_answer_map_from_disk()
            _status.stage = "done"
            _status.message = f"切分完成，本次处理 {len(new_map)} 份作业"
        except Exception as e:
            logger.exception("Segmentation pipeline failed")
            _status.stage = "error"
            _status.errors.append(str(e))

    _running_task = asyncio.create_task(_run())
    return {"ok": True, "num_pdfs": num_pdfs, "incremental": incremental}


@router.post("/segment/one")
async def start_segmentation_one(body: dict = Body(...)) -> dict:
    """对指定同学的作业重新执行 AI 切分（如重新提交了 PDF）。stem 可为规范名 学号_姓名 或完整 PDF 文件名 stem。"""
    global _running_task, _answer_map
    stem = (body.get("stem") or "").strip().removesuffix(".pdf")
    if not stem:
        return {"error": "请提供 stem（作业目录名，如学号_姓名 或 学号_姓名_随机码）"}

    if _status.stage not in ("idle", "done", "error"):
        return {"error": f"Pipeline already running: {_status.stage}"}

    pdfs = _list_pdfs()
    pdf_path = None
    for p in pdfs:
        if Path(p).stem == stem or student_canonical_stem(p) == stem:
            pdf_path = p
            break
    if not pdf_path:
        return {"error": f"未找到该作业的 PDF：{stem}"}

    _status.stage = "segmentation"
    _status.errors = []
    _status.total = 1
    _status.current = 0
    _status.message = f"正在重新切分：{stem}"

    async def _run():
        global _answer_map
        try:
            cfg = get_config()
            questions = load_all_questions()
            new_map = await run_segmentation(cfg, questions, [pdf_path], on_progress=_update_status)
            _answer_map = _build_answer_map_from_disk()
            _status.stage = "done"
            _status.message = f"已重新切分：{stem}"
        except Exception as e:
            logger.exception("Segmentation one failed for %s", stem)
            _status.stage = "error"
            _status.errors.append(str(e))

    _running_task = asyncio.create_task(_run())
    return {"ok": True, "stem": stem}


@router.post("/grade")
async def start_grading(body: dict | None = Body(default=None)) -> dict:
    """全量或增量评分。body 可含 "incremental": true，仅评尚未有 results 的作业。"""
    global _running_task, _answer_map
    if _status.stage not in ("idle", "done", "error"):
        return {"error": f"Pipeline already running: {_status.stage}"}

    _answer_map = _build_answer_map_from_disk()
    if not _answer_map:
        return {"error": "请先运行作业切分，或确认 answers 目录下已有切分结果"}

    incremental = bool((body or {}).get("incremental"))
    to_grade = _filter_answer_map_ungraded(_answer_map) if incremental else _answer_map
    if not to_grade:
        return {
            "error": "增量评分时没有待评阅作业" if incremental else "没有可评阅的切分结果",
        }

    _status.stage = "grading"
    _status.errors = []
    questions = load_all_questions()
    total = sum(len(by_q) for by_q in to_grade.values())
    _status.total = total
    _status.current = 0
    _status.message = f"即将{'增量' if incremental else ''}评分，共 {len(to_grade)} 份作业"

    async def _run():
        try:
            cfg = get_config()
            await run_grading(cfg, questions, to_grade, on_progress=_update_status)
            _status.stage = "done"
            _status.message = "评分完成"
        except Exception as e:
            logger.exception("Grading pipeline failed")
            _status.stage = "error"
            _status.errors.append(str(e))

    _running_task = asyncio.create_task(_run())
    return {"ok": True, "incremental": incremental, "num_students": len(to_grade)}


@router.post("/grade/one")
async def start_grading_one(body: dict = Body(...)) -> dict:
    """对指定同学的作业重新执行 AI 评阅（如重新提交后已重新切分）。"""
    global _running_task, _answer_map
    stem = (body.get("stem") or "").strip().removesuffix(".pdf")
    if not stem:
        return {"error": "请提供 stem（作业目录名）"}

    if _status.stage not in ("idle", "done", "error"):
        return {"error": f"Pipeline already running: {_status.stage}"}

    _answer_map = _build_answer_map_from_disk()
    if stem not in _answer_map:
        return {"error": f"未找到该作业的切分结果：{stem}，请先执行切分"}

    one_map = {stem: _answer_map[stem]}
    _status.stage = "grading"
    _status.errors = []
    questions = load_all_questions()
    _status.total = sum(len(by_q) for by_q in one_map.values())
    _status.current = 0
    _status.message = f"正在重新评阅：{stem}"

    async def _run():
        try:
            cfg = get_config()
            await run_grading(cfg, questions, one_map, on_progress=_update_status)
            _status.stage = "done"
            _status.message = f"已重新评阅：{stem}"
        except Exception as e:
            logger.exception("Grading one failed for %s", stem)
            _status.stage = "error"
            _status.errors.append(str(e))

    _running_task = asyncio.create_task(_run())
    return {"ok": True, "stem": stem}


@router.post("/report")
async def start_report_generation() -> dict:
    """在后台线程中生成报告，不阻塞事件循环，评阅/下载等接口可照常响应。"""
    global _running_task, _reports
    with _status_lock:
        if _status.stage not in ("idle", "done", "error"):
            return {"error": f"Pipeline already running: {_status.stage}"}
        _status.stage = "report"
        _status.message = "正在生成报告…"
        _status.current = 0
        _status.total = 0
        _status.errors = []

    questions = load_all_questions()
    cfg = get_config()
    results = load_all_results()
    if not results:
        with _status_lock:
            _status.stage = "error"
            _status.errors.append("没有找到评分结果，请先运行评分")
        return {"error": "没有找到评分结果，请先运行评分"}

    def _run_in_thread() -> tuple[list[QuestionReport] | None, str | None]:
        """在线程中执行，返回 (reports, error_message)。"""
        try:
            reports = run_report_sync(
                cfg, questions, results,
                on_progress=_update_status_sync,
            )
            return (reports, None)
        except Exception as e:
            logger.exception("Report pipeline failed")
            return (None, str(e))

    loop = asyncio.get_event_loop()

    async def _wait_and_finish():
        global _reports
        future = loop.run_in_executor(_report_executor, _run_in_thread)
        reports, err = await future
        with _status_lock:
            if err:
                _status.stage = "error"
                _status.errors.append(err)
            else:
                _reports = reports or []
                _save_reports_to_disk(_reports)
                _status.stage = "done"
                _status.message = "报告生成完成"

    _running_task = asyncio.create_task(_wait_and_finish())
    return {"ok": True}


@router.post("/report/one")
async def start_report_generation_one(body: dict = Body(...)) -> dict:
    """仅重新生成指定题目的报告。"""
    global _running_task, _reports
    qid = (body.get("qid") or "").strip()
    if not qid:
        return {"error": "请提供 qid"}

    with _status_lock:
        if _status.stage not in ("idle", "done", "error"):
            return {"error": f"Pipeline already running: {_status.stage}"}
        _status.stage = "report"
        _status.message = f"正在重新生成报告：{qid}"
        _status.current = 0
        _status.total = 1
        _status.errors = []

    questions = load_all_questions()
    q = next((x for x in questions if x.qid == qid), None)
    if q is None:
        with _status_lock:
            _status.stage = "error"
            _status.errors.append(f"未找到题目：{qid}")
        return {"error": f"未找到题目：{qid}"}

    cfg = get_config()
    results = load_all_results()
    if not results:
        with _status_lock:
            _status.stage = "error"
            _status.errors.append("没有找到评分结果，请先运行评分")
        return {"error": "没有找到评分结果，请先运行评分"}

    def _run_one_in_thread() -> tuple[QuestionReport | None, str | None]:
        try:
            report = generate_question_report_sync(cfg, q, results)
            return (report, None)
        except Exception as e:
            logger.exception("Single report generation failed for %s", qid)
            return (None, str(e))

    loop = asyncio.get_event_loop()

    async def _wait_and_finish():
        global _reports
        future = loop.run_in_executor(_report_executor, _run_one_in_thread)
        report, err = await future
        with _status_lock:
            if err:
                _status.stage = "error"
                _status.errors.append(err)
                return
            if not report:
                _status.stage = "error"
                _status.errors.append(f"{qid} 报告生成失败")
                return

            if not _reports:
                _reports = _load_reports_from_disk()
            replaced = False
            for i, rpt in enumerate(_reports):
                if rpt.qid == qid:
                    _reports[i] = report
                    replaced = True
                    break
            if not replaced:
                _reports.append(report)

            # 按题目配置顺序排序，保证前端展示稳定
            order = {qq.qid: idx for idx, qq in enumerate(questions)}
            _reports.sort(key=lambda r: order.get(r.qid, 10**9))
            _save_reports_to_disk(_reports)
            _status.current = 1
            _status.stage = "done"
            _status.message = f"{qid} 报告重新生成完成"

    _running_task = asyncio.create_task(_wait_and_finish())
    return {"ok": True, "qid": qid}


@router.get("/answer_map")
def get_answer_map() -> dict:
    return _answer_map


# ---------- 人工重新切分（segment 编辑） ----------


def _list_segment_assignments() -> list[dict]:
    """列出已有 _pages/for_llm 的作业（stem + 显示名）。"""
    answers_dir = get_answers_dir()
    if not answers_dir.exists():
        return []
    out = []
    for sub in sorted(answers_dir.iterdir()):
        if not sub.is_dir():
            continue
        stem = sub.name
        llm_dir = sub / "_pages" / "for_llm"
        if not llm_dir.is_dir():
            continue
        pages = sorted(llm_dir.glob("page_*.png"), key=_page_file_sort_key)
        if not pages:
            continue
        out.append({"stem": stem, "label": stem})
    return out


@router.get("/segment-editor/assignments")
def segment_editor_list_assignments() -> list:
    """获取可编辑 segment 的作业列表。"""
    return _list_segment_assignments()


def _resolve_stem_for_answers(stem: str) -> str | None:
    """解析 stem 对应的 answers 下实际目录名（处理 Unicode 规范化差异、.pdf 后缀）。"""
    import unicodedata
    stem = stem.removesuffix(".pdf") if stem.endswith(".pdf") else stem
    if not is_safe_path_segment(stem):
        return None
    answers_dir = get_answers_dir()
    if not answers_dir.exists():
        return None
    stem_nfc = unicodedata.normalize("NFC", stem)
    for d in answers_dir.iterdir():
        if d.is_dir() and unicodedata.normalize("NFC", d.name) == stem_nfc:
            return d.name
    return stem if (answers_dir / stem).exists() else None


@router.get("/segment-editor/assignment/{stem}")
def segment_editor_get_assignment(stem: str) -> dict:
    """获取某份作业的 for_llm 页面与当前 segment 数据。"""
    from PIL import Image

    actual_stem = _resolve_stem_for_answers(stem)
    if actual_stem is None:
        return {"error": "未找到该作业的 for_llm 页面"}
    base = get_answers_dir() / actual_stem / "_pages" / "for_llm"
    if not base.exists() or not base.is_dir():
        return {"error": "未找到该作业的 for_llm 页面"}
    page_files = sorted(base.glob("page_*.png"), key=_page_file_sort_key)
    pages = []
    dimensions = []
    for f in page_files:
        try:
            img = Image.open(str(f))
            w, h = img.size
            img.close()
        except Exception:
            w, h = 0, 0
        dimensions.append([w, h])
        # 前端通过 /files/answers/{stem}/_pages/for_llm/page_N.png 访问（用实际目录名）
        rel = f"/files/answers/{actual_stem}/_pages/for_llm/{f.name}"
        pages.append({"url": rel, "width": w, "height": h, "name": f.name})
    loaded = load_segments_for_stem(actual_stem)
    questions = []
    if loaded:
        raw = loaded.get("questions") or loaded.get("question")
        if isinstance(raw, list):
            questions = raw
        if loaded.get("dimensions"):
            dimensions = loaded["dimensions"]
    qids = [q.qid for q in load_all_questions()]
    return {
        "stem": actual_stem,
        "pages": pages,
        "dimensions": dimensions,
        "questions": questions,
        "qids": qids,
    }


@router.post("/segment-editor/assignment/{stem}/save")
def segment_editor_save_assignment(stem: str, body: dict = Body(...)) -> dict:
    """保存人工修改的 segment，并重新生成答案图。"""
    global _answer_map
    if not is_safe_path_segment(stem):
        return {"error": "无效的作业目录名"}
    dimensions = body.get("dimensions") or []
    questions = body.get("questions") or []
    if not dimensions and not questions:
        return {"error": "缺少 dimensions 或 questions"}
    pdfs = _list_pdfs()
    pdf_path = None
    for p in pdfs:
        if student_canonical_stem(p) == stem:
            pdf_path = p
            break
    if not pdf_path:
        return {"error": f"未找到对应 PDF：{stem}"}
    # dimensions 转为 list of tuple 供 apply_manual_segments
    dims_tuples = [tuple(d) if isinstance(d, list) else d for d in dimensions]
    try:
        answer_map = apply_manual_segments(pdf_path, dims_tuples, questions)
        _answer_map[stem] = answer_map
        return {"ok": True, "message": "已更新该作业的 segment 并重新生成答案图"}
    except Exception as e:
        logger.exception("segment-editor save failed for %s", stem)
        return {"error": str(e)}


@router.get("/reports")
def get_reports() -> list:
    global _reports
    if not _reports:
        _reports = _load_reports_from_disk()
    return [r.model_dump() for r in _reports]


@router.get("/reports/export-pdf")
def export_reports_pdf(anonymize: bool = False) -> Response:
    """导出答题报告 PDF（仅报告内容，学生姓名统一匿名为 ***）。"""
    global _reports
    if not _reports:
        _reports = _load_reports_from_disk()
    if not _reports:
        return Response(
            content=json.dumps({"error": "暂无报告可导出，请先生成报告"}, ensure_ascii=False),
            status_code=400,
            media_type="application/json",
        )

    try:
        questions = load_all_questions()
        pdf_bytes = _build_reports_pdf_bytes(_reports, questions, anonymize=anonymize)
    except Exception as e:
        logger.exception("Export reports PDF failed")
        return Response(
            content=json.dumps({"error": f"导出 PDF 失败：{e}"}, ensure_ascii=False),
            status_code=500,
            media_type="application/json",
        )

    headers = {"Content-Disposition": 'attachment; filename="report.pdf"'}
    return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)


@router.post("/cancel")
async def cancel_pipeline() -> dict:
    """取消当前正在运行的切分/评分/报告任务。"""
    global _running_task
    if _running_task is None:
        return {"ok": True, "message": "没有正在运行的任务"}
    if _running_task.done():
        _running_task = None
        with _status_lock:
            _status.stage = "idle"
            _status.message = ""
        return {"ok": True, "message": "任务已结束"}
    _running_task.cancel()
    try:
        await _running_task
    except asyncio.CancelledError:
        pass
    finally:
        _running_task = None
        with _status_lock:
            _status.stage = "idle"
            _status.message = "已取消"
            _status.current = 0
            _status.total = 0
            _status.errors = []
    return {"ok": True, "message": "已取消"}


@router.post("/regrade/{stem}/{qid}")
async def regrade_single(stem: str, qid: str) -> dict:
    """对指定学生的指定题目进行 AI 重新评阅。"""
    stem = stem.removesuffix(".pdf") if stem.endswith(".pdf") else stem
    if not is_safe_path_segment(stem) or not is_safe_path_segment(qid):
        return {"error": "无效的作业目录名或题目编号"}
    global _answer_map
    if not _answer_map:
        _answer_map = _build_answer_map_from_disk()
    if stem not in _answer_map or qid not in _answer_map[stem]:
        return {"error": f"未找到该学生的 {qid} 作答"}

    from autograder.config import get_config

    questions = load_all_questions()
    q_map = {q.qid: q for q in questions}
    q = q_map.get(qid)
    if not q:
        return {"error": f"未找到题目配置 {qid}"}

    cfg = get_config()
    graph = build_grading_graph(cfg)
    app = graph.compile()
    grading_cfg = cfg.assignment_grading
    answer_paths = _answer_map[stem][qid]

    try:
        _, record = await asyncio.to_thread(
            _grade_one_sync,
            app,
            grading_cfg,
            stem,
            qid,
            q,
            answer_paths,
        )
    except Exception as e:
        logger.exception("Regrade failed for %s/%s", stem, qid)
        return {"error": str(e)}

    sr = load_student_result(stem)
    if not sr:
        return {"error": "未找到该学生的评阅结果"}

    updated = False
    for r in sr.records:
        if r.qid == qid:
            r.score = record.score
            r.confidence = record.confidence
            r.summary = record.summary
            r.comments = record.comments
            r.grader = record.grader
            r.graded_at = record.graded_at
            updated = True
            break
    if not updated:
        sr.records.append(record)
        sr.records.sort(key=lambda x: x.qid)

    sr.total_score = sum(r.score for r in sr.records)
    _save_student_result(sr)
    return {"ok": True, "score": record.score, "confidence": record.confidence}


@router.post("/reset")
def reset_status() -> dict:
    with _status_lock:
        _status.stage = "idle"
        _status.message = ""
        _status.current = 0
        _status.total = 0
        _status.errors = []
    return {"ok": True}
