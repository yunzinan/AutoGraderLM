"""Router: question configuration CRUD + image upload."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel

from autograder.models import QuestionConfig

router = APIRouter(prefix="/api/questions", tags=["questions"])
QUESTIONS_DIR = Path("./questions")


def _q_dir(qid: str) -> Path:
    d = QUESTIONS_DIR / qid
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_config(qid: str) -> QuestionConfig | None:
    cfg_path = _q_dir(qid) / "config.json"
    if not cfg_path.exists():
        return None
    return QuestionConfig.model_validate_json(cfg_path.read_text(encoding="utf-8"))


def _save_config(q: QuestionConfig) -> None:
    cfg_path = _q_dir(q.qid) / "config.json"
    cfg_path.write_text(q.model_dump_json(indent=2), encoding="utf-8")


def load_all_questions() -> list[QuestionConfig]:
    QUESTIONS_DIR.mkdir(parents=True, exist_ok=True)
    qs: list[QuestionConfig] = []
    for d in sorted(QUESTIONS_DIR.iterdir()):
        if d.is_dir():
            q = _load_config(d.name)
            if q:
                qs.append(q)
    return qs


@router.get("")
def list_questions() -> list[QuestionConfig]:
    return load_all_questions()


@router.get("/{qid}")
def get_question(qid: str) -> QuestionConfig | dict:
    q = _load_config(qid)
    if q is None:
        return {"error": "not found"}
    return q


class QuestionUpdate(BaseModel):
    question_index: Optional[int] = None
    score: int = 0
    rubric: str = ""
    question_text: str = ""
    example_answer_text: str = ""


@router.post("/{qid}")
def upsert_question(qid: str, body: QuestionUpdate) -> QuestionConfig:
    existing = _load_config(qid)
    q = QuestionConfig(
        qid=qid,
        question_index=body.question_index,
        score=body.score,
        rubric=body.rubric,
        question_text=body.question_text,
        example_answer_text=body.example_answer_text,
        question_images=existing.question_images if existing else [],
        example_answer_images=existing.example_answer_images if existing else [],
    )
    _save_config(q)
    return q


@router.delete("/{qid}")
def delete_question(qid: str) -> dict:
    d = QUESTIONS_DIR / qid
    if d.exists():
        shutil.rmtree(d)
    return {"ok": True}


@router.post("/{qid}/upload_question_image")
async def upload_question_image(qid: str, file: UploadFile = File(...)) -> dict:
    d = _q_dir(qid)
    existing = sorted(d.glob("question_*.png"))
    idx = len(existing)
    dest = d / f"question_{idx}.png"
    dest.write_bytes(await file.read())

    q = _load_config(qid) or QuestionConfig(qid=qid, score=0)
    q.question_images.append(str(dest))
    _save_config(q)
    return {"path": str(dest)}


@router.post("/{qid}/upload_example_image")
async def upload_example_image(qid: str, file: UploadFile = File(...)) -> dict:
    d = _q_dir(qid)
    existing = sorted(d.glob("example_*.png"))
    idx = len(existing)
    dest = d / f"example_{idx}.png"
    dest.write_bytes(await file.read())

    q = _load_config(qid) or QuestionConfig(qid=qid, score=0)
    q.example_answer_images.append(str(dest))
    _save_config(q)
    return {"path": str(dest)}
