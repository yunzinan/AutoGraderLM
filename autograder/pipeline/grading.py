"""Grading pipeline – LLM-based answer grading with confidence-driven retry."""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import TypedDict

from jinja2 import Template
from langgraph.graph import END, StateGraph

from autograder.config import AppConfig, get_results_dir, resolve_assignment_path
from autograder.llm import build_llm, build_vision_message_segmented, extract_json, invoke_with_log
from autograder.models import GradingOutput, GradingRecord, QuestionConfig, StudentResult, _now_iso
from autograder.pdf_utils import parse_student_info, student_canonical_stem

logger = logging.getLogger(__name__)


class GradeState(TypedDict, total=False):
    qid: str
    max_score: int
    question_text: str
    rubric: str
    example_answer_text: str
    question_image_paths: list[str]
    answer_image_paths: list[str]
    example_image_paths: list[str]
    output: dict | None
    retry_count: int
    max_retry: int
    error: str


def build_grading_graph(cfg: AppConfig) -> StateGraph:
    llm_cfg = cfg.assignment_grading.llm
    llm = build_llm(llm_cfg)
    prompt_dir = Path(llm_cfg.prompt_template).parent
    intro_tpl = Template((prompt_dir / "grading_intro.jinja").read_text(encoding="utf-8"))
    rubric_tpl = Template((prompt_dir / "grading_rubric.jinja").read_text(encoding="utf-8"))
    example_tpl = Template((prompt_dir / "grading_example.jinja").read_text(encoding="utf-8"))
    tail_tpl = Template((prompt_dir / "grading_tail.jinja").read_text(encoding="utf-8"))

    def grade_node(state: GradeState) -> dict:
        # 各板块后紧接该板块的图片：题目信息+题目图 → 评分标准 → 参考答案+示例答案图 → 评阅流程与输出要求 → 学生作答图
        intro_text = intro_tpl.render(
            qid=state["qid"],
            max_score=state["max_score"],
            question_text=state.get("question_text") or "",
        )
        rubric_text = rubric_tpl.render(rubric=state.get("rubric") or "")
        has_example = bool(state.get("example_answer_text") or state.get("example_image_paths"))
        example_text = example_tpl.render(
            example_answer_text=state.get("example_answer_text") or "",
            has_example=has_example,
        )
        tail_text = tail_tpl.render(max_score=state["max_score"])

        segments: list[tuple[str, list[str]]] = [
            (intro_text, state.get("question_image_paths") or []),
            (rubric_text, []),
        ]
        if has_example:
            segments.append((example_text, state.get("example_image_paths") or []))
        segments.append((tail_text, state["answer_image_paths"]))

        msg = build_vision_message_segmented(segments)
        try:
            stem = Path(state["answer_image_paths"][0]).parent.name if state["answer_image_paths"] else ""
            ctx = {"stage": "grading", "qid": state["qid"], "pdf_stem": stem}
            resp = invoke_with_log(llm, [msg], ctx)
            raw = extract_json(resp.content)
            if raw is None:
                return {
                    "output": None,
                    "retry_count": state["retry_count"] + 1,
                    "error": "JSON parse failed",
                }
            parsed = GradingOutput(**raw)
            if parsed.score > state["max_score"]:
                parsed.score = state["max_score"]
            return {"output": parsed.model_dump(), "retry_count": state["retry_count"] + 1, "error": ""}
        except Exception as e:
            logger.exception("Grading LLM call failed for %s", state["qid"])
            return {
                "output": None,
                "retry_count": state["retry_count"] + 1,
                "error": str(e),
            }

    def should_retry(state: GradeState) -> str:
        if state["output"] is None:
            if state["retry_count"] < state["max_retry"]:
                return "retry"
            return "done"
        # confidence == 2 → retry until >= 3 or exhausted
        if state["output"]["confidence"] == 2 and state["retry_count"] < state["max_retry"]:
            return "retry"
        return "done"

    graph = StateGraph(GradeState)
    graph.add_node("grade", grade_node)
    graph.add_conditional_edges("grade", should_retry, {"retry": "grade", "done": END})
    graph.set_entry_point("grade")
    return graph


def _save_student_result(result: StudentResult) -> None:
    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    canonical = student_canonical_stem(result.filename)
    path = results_dir / f"{canonical}.json"
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")


def load_student_result(filename_stem: str) -> StudentResult | None:
    path = get_results_dir() / f"{filename_stem}.json"
    if not path.exists():
        return None
    return StudentResult.model_validate_json(path.read_text(encoding="utf-8"))


def load_all_results() -> list[StudentResult]:
    """加载 results 目录下所有学生评阅结果（持久化 JSON），不论是否有人工复核。
    仅加载学生结果文件，跳过 question_reports.json 等非学生结果文件。
    """
    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for p in sorted(results_dir.glob("*.json")):
        if p.name == "question_reports.json":
            continue
        try:
            results.append(StudentResult.model_validate_json(p.read_text(encoding="utf-8")))
        except Exception:
            logger.exception("Failed to load result %s", p)
    return results


def _grade_one_sync(
    app,
    grading_cfg,
    stem: str,
    qid: str,
    q: QuestionConfig,
    answer_paths: list[str],
) -> tuple[str, GradingRecord]:
    """同步执行单题评阅，供在线程中调用。返回 (stem, record)。"""
    question_image_paths = [
        str(resolve_assignment_path(p)) for p in (q.question_images or [])
        if p and resolve_assignment_path(p).exists()
    ]
    example_paths = [
        str(resolve_assignment_path(p)) for p in (q.example_answer_images or [])
        if p and resolve_assignment_path(p).exists()
    ]
    init_state: GradeState = {
        "qid": qid,
        "max_score": q.score,
        "question_text": q.question_text or "",
        "rubric": q.rubric or "",
        "example_answer_text": q.example_answer_text or "",
        "question_image_paths": question_image_paths,
        "answer_image_paths": answer_paths,
        "example_image_paths": example_paths,
        "output": None,
        "retry_count": 0,
        "max_retry": grading_cfg.max_retry,
        "error": "",
    }
    final = app.invoke(init_state)

    record = GradingRecord(
        filename=stem + ".pdf",
        qid=qid,
        question_index=getattr(q, "question_index", None),
        answer=answer_paths,
        grader=grading_cfg.llm.model,
        graded_at=_now_iso(),
    )
    if final["output"] is not None:
        out = GradingOutput(**final["output"])
        record.score = out.score
        record.confidence = out.confidence
        record.summary = out.summary
        record.comments = out.comments
    else:
        record.confidence = 0
        record.comments = f"评分失败: {final.get('error', 'unknown')}"
    return (stem, record)


async def run_grading(
    cfg: AppConfig,
    questions: list[QuestionConfig],
    answer_map: dict[str, dict[str, list[str]]],
    on_progress=None,
) -> list[StudentResult]:
    """Grade all students' answers.

    使用 num_workers 并发调用大模型评阅多道题。
    *answer_map*: pdf_stem -> {qid: [image_path, ...]}
    Returns list of StudentResult.
    """
    graph = build_grading_graph(cfg)
    app = graph.compile()
    grading_cfg = cfg.assignment_grading
    num_workers = grading_cfg.num_workers

    q_map = {q.qid: q for q in questions}
    all_students = sorted(answer_map.keys())
    total = sum(len(qids) for qids in answer_map.values())
    sem = asyncio.Semaphore(num_workers)
    completed = [0]

    # 展平为 (stem, qid, q, answer_paths) 任务列表
    tasks_args: list[tuple[str, str, QuestionConfig, list[str]]] = []
    for stem in all_students:
        qid_images = answer_map[stem]
        for qid in sorted(qid_images.keys()):
            q = q_map.get(qid)
            if q is None:
                logger.warning("Question config not found for %s", qid)
                continue
            tasks_args.append((stem, qid, q, qid_images[qid]))

    async def grade_one(stem: str, qid: str, q: QuestionConfig, answer_paths: list[str]) -> tuple[str, GradingRecord]:
        async with sem:
            stem_result, record = await asyncio.to_thread(
                _grade_one_sync,
                app,
                grading_cfg,
                stem,
                qid,
                q,
                answer_paths,
            )
        completed[0] += 1
        if on_progress:
            await on_progress(
                f"评分中: ({completed[0]}/{total})",
                completed[0],
                total,
            )
        return (stem_result, record)

    task_list = [grade_one(stem, qid, q, paths) for stem, qid, q, paths in tasks_args]
    raw_results = await asyncio.gather(*task_list)

    # 按 stem 分组，组装 StudentResult
    by_stem: dict[str, list[GradingRecord]] = defaultdict(list)
    for stem, record in raw_results:
        by_stem[stem].append(record)

    results: list[StudentResult] = []
    for stem in all_students:
        records = by_stem.get(stem, [])
        records.sort(key=lambda r: r.qid)
        student_id, student_name = parse_student_info(stem + ".pdf")
        student_result = StudentResult(
            filename=stem + ".pdf",
            student_id=student_id,
            student_name=student_name,
        )
        student_result.records = records
        student_result.total_score = sum(r.score for r in records)
        _save_student_result(student_result)
        results.append(student_result)

    return results
