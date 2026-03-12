"""Report generation – per-student summary & per-question analysis (two-step)."""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from jinja2 import Template

from autograder.config import AppConfig, resolve_assignment_path, to_relative_url_path
from autograder.excel_utils import export_graded_xlsx, read_student_roster
from autograder.llm import build_llm, build_vision_message_segmented, invoke_with_log
from autograder.models import (
    GradingRecord,
    QuestionConfig,
    QuestionReport,
    StudentResult,
)

logger = logging.getLogger(__name__)


def build_student_comment(records: list[GradingRecord]) -> str:
    """Aggregate per-question comments into a single student comment."""
    parts: list[str] = []
    for r in records:
        line = f"【{r.qid}】({r.score}分)"
        if r.comments:
            line += f" {r.comments}"
        parts.append(line)
    return "\n".join(parts)


def export_scores(cfg: AppConfig, results: list[StudentResult]) -> str:
    """Write graded results into the output xlsx, return the output path."""
    score_map: dict[str, tuple[int, str]] = {}
    for sr in results:
        comment = build_student_comment(sr.records)
        score_map[sr.student_id] = (sr.total_score, comment)

    in_path = cfg.assignment_configuration.excel_in_path
    out_path = cfg.assignment_report.excel_out_path
    export_graded_xlsx(in_path, score_map, out_path)
    logger.info("Exported graded xlsx to %s", out_path)
    return out_path


def compute_score_distribution(
    results: list[StudentResult], qid: str
) -> dict[int, int]:
    dist: dict[int, int] = defaultdict(int)
    for sr in results:
        for r in sr.records:
            if r.qid == qid:
                dist[r.score] += 1
    return dict(sorted(dist.items()))


# ── Data collection helpers ──────────────────────────────────────────


def _collect_question_data(
    question: QuestionConfig,
    results: list[StudentResult],
) -> tuple[list[dict], dict[str, list[str]], dict[str, list[str]], dict[int, int]]:
    """Collect enriched records, student_answers (relative), student_abs_answers (absolute), and score dist."""
    qid = question.qid
    name_map: dict[str, str] = {sr.filename: (sr.student_name or sr.filename) for sr in results}

    enriched: list[dict] = []
    student_answers: dict[str, list[str]] = {}
    student_abs_answers: dict[str, list[str]] = {}
    for sr in results:
        for r in sr.records:
            if r.qid != qid:
                continue
            d = r.model_dump()
            sname = name_map.get(r.filename, r.filename)
            d["student_name"] = sname
            enriched.append(d)
            student_answers[sname] = [to_relative_url_path(p) for p in r.answer]
            student_abs_answers[sname] = list(r.answer)

    score_dist = compute_score_distribution(results, qid)
    return enriched, student_answers, student_abs_answers, score_dist


def _extract_mentioned_names(text: str, all_names: list[str]) -> list[str]:
    """Return names from all_names that appear in text."""
    return [n for n in all_names if n in text]


def _split_report_and_reference(text: str) -> tuple[str, str]:
    """Split refined LLM output into (report_text, reference_answer) at --- or 参考作答 header."""
    import re

    text = text.strip()
    fence = re.match(r"^```(?:markdown|md)?\s*\n([\s\S]*?)\n```\s*$", text)
    if fence:
        text = fence.group(1).strip()

    # 参考作答/参考答案 标题行：支持 ## 参考作答、### 参考作答（满分示例）、三、参考作答 等
    ref_header_re = re.compile(
        r"^(?:#+\s*)?(?:[一二三四五六七八九十]+[、.]\s*)?"
        r"(?:参考作答|参考答案|标准答案|Reference Answer)"
        r"(?:[（(][^）)]*[）)])?\s*$",
        re.MULTILINE | re.IGNORECASE,
    )

    def extract_reference(rest: str) -> str:
        """从 rest 中提取参考作答内容（从标题行之后到结尾）。"""
        m = ref_header_re.search(rest)
        if m:
            return rest[m.end() :].strip()
        return rest.strip()

    # 1. 优先按 --- 分隔
    for marker in ("---", "***", "___"):
        idx = text.find(f"\n{marker}\n")
        if idx != -1:
            report_part = text[:idx].strip()
            rest = text[idx + len(marker) + 2 :].lstrip().lstrip("-*_ \t\n")
            return report_part, extract_reference(rest)

    # 2. 无 --- 时，按 参考作答 标题行拆分
    m = ref_header_re.search(text)
    if m:
        report_part = text[: m.start()].strip()
        reference = text[m.end() :].strip()
        return report_part, reference

    return text, ""


# ── Step 1: Initial report (text-only) ──────────────────────────────


def _step1_render_and_call(
    cfg: AppConfig,
    question: QuestionConfig,
    enriched_results: list[dict],
    score_dist: dict[int, int],
) -> str:
    """Step 1: render prompt from template and call LLM (text-only), return initial report."""
    report_cfg = cfg.assignment_report
    tpl = Template(Path(report_cfg.llm.prompt_template).read_text(encoding="utf-8"))
    prompt_text = tpl.render(
        qid=question.qid,
        max_score=question.score,
        question_text=question.question_text,
        rubric=question.rubric,
        score_distribution=score_dist,
        results=enriched_results,
    )
    llm = build_llm(report_cfg.llm)
    from langchain_core.messages import HumanMessage

    llm_bounded = llm.bind(max_tokens=2048)
    msg = HumanMessage(content=prompt_text)
    resp = invoke_with_log(llm_bounded, [msg], {"stage": "report-step1", "qid": question.qid})
    return resp.content


# ── Step 2: Refined report with vision ───────────────────────────────


def _step2_refine_with_images(
    cfg: AppConfig,
    question: QuestionConfig,
    initial_report: str,
    mentioned_names: list[str],
    enriched_results: list[dict],
    student_abs_answers: dict[str, list[str]],
) -> str:
    """Step 2: send initial report + mentioned students' answer images → refined report + reference answer."""
    report_cfg = cfg.assignment_report
    prompt_dir = Path(report_cfg.llm.prompt_template).parent
    refined_tpl_path = prompt_dir / "report_refined.jinja"
    if not refined_tpl_path.exists():
        logger.warning("report_refined.jinja not found at %s, skipping step 2", refined_tpl_path)
        return initial_report

    refined_tpl = Template(refined_tpl_path.read_text(encoding="utf-8"))
    enriched_by_name: dict[str, dict] = {r["student_name"]: r for r in enriched_results}

    mentioned_students = []
    for name in mentioned_names:
        rec = enriched_by_name.get(name)
        if rec:
            mentioned_students.append({
                "name": name,
                "score": rec["score"],
                "summary": rec.get("summary", ""),
                "comments": rec.get("comments", ""),
            })

    intro_text = refined_tpl.render(
        qid=question.qid,
        max_score=question.score,
        question_text=question.question_text,
        rubric=question.rubric,
        initial_report=initial_report,
        mentioned_students=mentioned_students,
    )

    # Build segmented vision message: intro+question images, then each student's images
    question_image_paths = [
        str(resolve_assignment_path(img)) for img in question.question_images
    ]

    # Split the rendered template at student image insertion points
    # We render the full prompt as one text block, then attach question images after the intro,
    # and attach each student's images after their section.
    # Simpler approach: build segments manually.

    segments: list[tuple[str, list[str]]] = []

    # Part 1: everything up to and including "题干图片" section → attach question images
    part1_end = "以下为该题的题干图片（若有），请结合上文理解题目："
    idx = intro_text.find(part1_end)
    if idx != -1:
        split_pos = idx + len(part1_end)
        segments.append((intro_text[:split_pos], question_image_paths))
        remaining = intro_text[split_pos:]
    else:
        segments.append((intro_text, question_image_paths))
        remaining = ""

    # Part 2: for each mentioned student, split at their image marker and attach their images
    for ms in mentioned_students:
        marker = f"（{ms['name']} 的作答图片如下）"
        idx = remaining.find(marker)
        if idx != -1:
            split_pos = idx + len(marker)
            text_before = remaining[:split_pos]
            abs_paths = student_abs_answers.get(ms["name"], [])
            segments.append((text_before, abs_paths))
            remaining = remaining[split_pos:]
        # else: no marker found, student's images won't be attached individually

    # Part 3: remaining text (the task description)
    if remaining.strip():
        segments.append((remaining, []))

    msg = build_vision_message_segmented(segments)
    llm = build_llm(report_cfg.llm)
    llm_bounded = llm.bind(max_tokens=4096)
    resp = invoke_with_log(llm_bounded, [msg], {"stage": "report-step2", "qid": question.qid})
    return resp.content


# ── Orchestration ────────────────────────────────────────────────────


def _generate_report_two_step(
    cfg: AppConfig,
    question: QuestionConfig,
    results: list[StudentResult],
) -> QuestionReport:
    """Two-step report generation: initial text report → refined vision report with reference answer."""
    enriched, student_answers, student_abs_answers, score_dist = _collect_question_data(question, results)

    # Step 1
    initial_report = _step1_render_and_call(cfg, question, enriched, score_dist)

    # Extract mentioned names
    all_names = list(student_answers.keys())
    mentioned = _extract_mentioned_names(initial_report, all_names)

    if mentioned:
        # Step 2
        refined_output = _step2_refine_with_images(
            cfg, question, initial_report, mentioned, enriched, student_abs_answers,
        )
        report_text, reference_answer = _split_report_and_reference(refined_output)
        # 兜底：若 report 过短或缺少解题思路/常见错误，回退使用初步报告
        if len(report_text.strip()) < 150 or ("解题思路" not in report_text and "常见错误" not in report_text):
            logger.warning(
                "Refined report for %s too short or missing 解题思路/常见错误 (len=%d), falling back to initial report",
                question.qid,
                len(report_text.strip()),
            )
            report_text = initial_report
    else:
        logger.info("No student names found in initial report for %s, skipping step 2", question.qid)
        report_text = initial_report
        reference_answer = ""

    return QuestionReport(
        qid=question.qid,
        question_index=getattr(question, "question_index", None),
        score_distribution=score_dist,
        report_text=report_text,
        reference_answer=reference_answer,
        student_answers=student_answers,
    )


async def generate_question_report(
    cfg: AppConfig,
    question: QuestionConfig,
    results: list[StudentResult],
) -> QuestionReport:
    """Generate an LLM-powered per-question analysis report (two-step)."""
    return _generate_report_two_step(cfg, question, results)


def _generate_question_report_sync(
    cfg: AppConfig,
    question: QuestionConfig,
    results: list[StudentResult],
) -> QuestionReport:
    """同步生成单题报告（两步），供线程池调用，不阻塞事件循环。"""
    return _generate_report_two_step(cfg, question, results)


def generate_question_report_sync(
    cfg: AppConfig,
    question: QuestionConfig,
    results: list[StudentResult],
) -> QuestionReport:
    """公开的单题同步报告生成接口。"""
    return _generate_question_report_sync(cfg, question, results)


def run_report_sync(
    cfg: AppConfig,
    questions: list[QuestionConfig],
    results: list[StudentResult],
    on_progress=None,
) -> list[QuestionReport]:
    """
    在后台线程中同步执行：先导出 Excel，再并发生成各题报告。
    on_progress(message, current, total) 会在工作线程中被调用，调用方需保证线程安全。
    """
    export_scores(cfg, results)
    num_workers = getattr(cfg.assignment_report, "num_workers", 3)
    total = len(questions)
    progress_lock = threading.Lock()
    completed = [0]  # 用 list 以便闭包内修改

    # 按题目顺序收集结果
    report_by_qid: dict[str, QuestionReport] = {}

    def _gen_one(q: QuestionConfig) -> QuestionReport:
        r = _generate_question_report_sync(cfg, q, results)
        with progress_lock:
            completed[0] += 1
            if on_progress:
                on_progress(f"生成报告: {q.qid} ({completed[0]}/{total})", completed[0], total)
        return r

    with ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="report-q") as ex:
        futures = {ex.submit(_gen_one, q): q for q in questions}
        for fut in as_completed(futures):
            report = fut.result()
            report_by_qid[report.qid] = report

    return [report_by_qid[q.qid] for q in questions]


async def generate_all_reports(
    cfg: AppConfig,
    questions: list[QuestionConfig],
    results: list[StudentResult],
    on_progress=None,
) -> list[QuestionReport]:
    reports: list[QuestionReport] = []
    total = len(questions)
    for i, q in enumerate(questions):
        if on_progress:
            await on_progress(f"生成报告: {q.qid} ({i + 1}/{total})", i + 1, total)
        report = await generate_question_report(cfg, q, results)
        reports.append(report)
    return reports
