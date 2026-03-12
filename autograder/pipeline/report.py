"""Report generation – per-student summary & per-question analysis."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

from jinja2 import Template

from autograder.config import AppConfig, to_relative_url_path
from autograder.excel_utils import export_graded_xlsx, read_student_roster
from autograder.llm import build_llm, invoke_with_log
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


def _collect_question_data(
    question: QuestionConfig,
    results: list[StudentResult],
) -> tuple[list[dict], dict[str, list[str]], dict[int, int]]:
    """Collect enriched records, student_answers mapping, and score distribution for a question."""
    qid = question.qid
    name_map: dict[str, str] = {sr.filename: (sr.student_name or sr.filename) for sr in results}

    enriched: list[dict] = []
    student_answers: dict[str, list[str]] = {}
    for sr in results:
        for r in sr.records:
            if r.qid != qid:
                continue
            d = r.model_dump()
            sname = name_map.get(r.filename, r.filename)
            d["student_name"] = sname
            enriched.append(d)
            student_answers[sname] = [to_relative_url_path(p) for p in r.answer]

    score_dist = compute_score_distribution(results, qid)
    return enriched, student_answers, score_dist


def _render_and_call_llm(
    cfg: AppConfig,
    question: QuestionConfig,
    enriched_results: list[dict],
    score_dist: dict[int, int],
) -> str:
    """Render prompt from template and call LLM, return response text."""
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
    resp = invoke_with_log(llm_bounded, [msg], {"stage": "report", "qid": question.qid})
    return resp.content


async def generate_question_report(
    cfg: AppConfig,
    question: QuestionConfig,
    results: list[StudentResult],
) -> QuestionReport:
    """Generate an LLM-powered per-question analysis report."""
    enriched, student_answers, score_dist = _collect_question_data(question, results)
    report_text = _render_and_call_llm(cfg, question, enriched, score_dist)

    return QuestionReport(
        qid=question.qid,
        question_index=getattr(question, "question_index", None),
        score_distribution=score_dist,
        report_text=report_text,
        student_answers=student_answers,
    )


def _generate_question_report_sync(
    cfg: AppConfig,
    question: QuestionConfig,
    results: list[StudentResult],
) -> QuestionReport:
    """同步生成单题报告，供线程池调用，不阻塞事件循环。"""
    enriched, student_answers, score_dist = _collect_question_data(question, results)
    report_text = _render_and_call_llm(cfg, question, enriched, score_dist)

    return QuestionReport(
        qid=question.qid,
        question_index=getattr(question, "question_index", None),
        score_distribution=score_dist,
        report_text=report_text,
        student_answers=student_answers,
    )


def run_report_sync(
    cfg: AppConfig,
    questions: list[QuestionConfig],
    results: list[StudentResult],
    on_progress=None,
) -> list[QuestionReport]:
    """
    在后台线程中同步执行：先导出 Excel，再逐题生成报告。
    on_progress(message, current, total) 会在工作线程中被调用，调用方需保证线程安全。
    """
    export_scores(cfg, results)
    reports: list[QuestionReport] = []
    for i, q in enumerate(questions):
        if on_progress:
            on_progress(f"生成报告: {q.qid} ({i + 1}/{len(questions)})", i + 1, len(questions))
        report = _generate_question_report_sync(cfg, q, results)
        reports.append(report)
    return reports


async def generate_all_reports(
    cfg: AppConfig,
    questions: list[QuestionConfig],
    results: list[StudentResult],
    on_progress=None,
) -> list[QuestionReport]:
    reports: list[QuestionReport] = []
    for i, q in enumerate(questions):
        if on_progress:
            await on_progress(f"生成报告: {q.qid} ({i + 1}/{len(questions)})", i + 1, len(questions))
        report = await generate_question_report(cfg, q, results)
        reports.append(report)
    return reports
