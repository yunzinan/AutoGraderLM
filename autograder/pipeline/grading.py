"""Grading pipeline – LLM-based answer grading with confidence-driven retry."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TypedDict

from jinja2 import Template
from langgraph.graph import END, StateGraph

from autograder.config import AppConfig
from autograder.llm import build_llm, build_vision_message, extract_json
from autograder.models import GradingOutput, GradingRecord, QuestionConfig, StudentResult
from autograder.pdf_utils import parse_student_info

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("./results")


class GradeState(TypedDict):
    qid: str
    max_score: int
    prompt_text: str
    answer_image_paths: list[str]
    example_image_paths: list[str]
    output: dict | None
    retry_count: int
    max_retry: int
    error: str


def build_grading_graph(cfg: AppConfig) -> StateGraph:
    llm_cfg = cfg.assignment_grading.llm
    llm = build_llm(llm_cfg)

    def grade_node(state: GradeState) -> dict:
        images = state["example_image_paths"] + state["answer_image_paths"]
        msg = build_vision_message(state["prompt_text"], images)
        try:
            resp = llm.invoke([msg])
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
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{Path(result.filename).stem}.json"
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")


def load_student_result(filename_stem: str) -> StudentResult | None:
    path = RESULTS_DIR / f"{filename_stem}.json"
    if not path.exists():
        return None
    return StudentResult.model_validate_json(path.read_text(encoding="utf-8"))


def load_all_results() -> list[StudentResult]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for p in sorted(RESULTS_DIR.glob("*.json")):
        try:
            results.append(StudentResult.model_validate_json(p.read_text(encoding="utf-8")))
        except Exception:
            logger.exception("Failed to load result %s", p)
    return results


async def run_grading(
    cfg: AppConfig,
    questions: list[QuestionConfig],
    answer_map: dict[str, dict[str, list[str]]],
    on_progress=None,
) -> list[StudentResult]:
    """Grade all students' answers.

    *answer_map*: pdf_stem -> {qid: [image_path, ...]}
    Returns list of StudentResult.
    """
    graph = build_grading_graph(cfg)
    app = graph.compile()
    grading_cfg = cfg.assignment_grading
    prompt_tpl = Template(Path(grading_cfg.llm.prompt_template).read_text(encoding="utf-8"))

    q_map = {q.qid: q for q in questions}
    all_students = sorted(answer_map.keys())
    total = sum(len(qids) for qids in answer_map.values())
    progress_count = 0
    results: list[StudentResult] = []

    for stem in all_students:
        qid_images = answer_map[stem]
        student_id, student_name = parse_student_info(stem + ".pdf")

        student_result = StudentResult(
            filename=stem + ".pdf",
            student_id=student_id,
            student_name=student_name,
        )

        for qid in sorted(qid_images.keys()):
            progress_count += 1
            if on_progress:
                await on_progress(f"评分中: {student_name} - {qid} ({progress_count}/{total})", progress_count, total)

            q = q_map.get(qid)
            if q is None:
                logger.warning("Question config not found for %s", qid)
                continue

            prompt_text = prompt_tpl.render(
                qid=q.qid,
                max_score=q.score,
                question_text=q.question_text,
                rubric=q.rubric,
                example_answer_text=q.example_answer_text,
            )

            init_state: GradeState = {
                "qid": qid,
                "max_score": q.score,
                "prompt_text": prompt_text,
                "answer_image_paths": qid_images[qid],
                "example_image_paths": q.example_answer_images,
                "output": None,
                "retry_count": 0,
                "max_retry": grading_cfg.max_retry,
                "error": "",
            }
            final = app.invoke(init_state)

            record = GradingRecord(
                filename=stem + ".pdf",
                qid=qid,
                answer=qid_images[qid],
                grader=grading_cfg.llm.model,
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

            student_result.records.append(record)

        student_result.total_score = sum(r.score for r in student_result.records)
        _save_student_result(student_result)
        results.append(student_result)

    return results
