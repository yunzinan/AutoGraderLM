"""Router: question configuration CRUD + image upload."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel

from autograder.config import get_questions_dir
from autograder.models import QuestionConfig

router = APIRouter(prefix="/api/questions", tags=["questions"])


def _q_dir(qid: str) -> Path:
    d = get_questions_dir() / qid
    d.mkdir(parents=True, exist_ok=True)
    return d


def _normalize_image_path(path: str, qid: str) -> str:
    """将绝对路径规范为相对路径 questions/{qid}/{filename}，供前端 /files/ + path 使用。"""
    p = Path(path)
    if not p.is_absolute():
        return path
    return f"questions/{qid}/{p.name}"


def _load_config(qid: str) -> QuestionConfig | None:
    cfg_path = _q_dir(qid) / "config.json"
    if not cfg_path.exists():
        return None
    q = QuestionConfig.model_validate_json(cfg_path.read_text(encoding="utf-8"))
    # 兼容旧配置：绝对路径转为相对路径，保证前端 /files/ + path 正确
    q.question_images = [_normalize_image_path(p, qid) for p in (q.question_images or [])]
    q.example_answer_images = [_normalize_image_path(p, qid) for p in (q.example_answer_images or [])]
    return q


def _save_config(q: QuestionConfig) -> None:
    cfg_path = _q_dir(q.qid) / "config.json"
    cfg_path.write_text(q.model_dump_json(indent=2), encoding="utf-8")


def load_all_questions() -> list[QuestionConfig]:
    qdir = get_questions_dir()
    qdir.mkdir(parents=True, exist_ok=True)
    qs: list[QuestionConfig] = []
    for d in sorted(qdir.iterdir()):
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
    question_images: Optional[list[str]] = None
    example_answer_images: Optional[list[str]] = None


@router.post("/{qid}")
def upsert_question(qid: str, body: QuestionUpdate) -> QuestionConfig:
    existing = _load_config(qid)
    question_images = body.question_images if body.question_images is not None else (existing.question_images if existing else [])
    example_answer_images = (
        body.example_answer_images if body.example_answer_images is not None else (existing.example_answer_images if existing else [])
    )
    q = QuestionConfig(
        qid=qid,
        question_index=body.question_index,
        score=body.score,
        rubric=body.rubric,
        question_text=body.question_text,
        example_answer_text=body.example_answer_text,
        question_images=[_normalize_image_path(p, qid) for p in (question_images or [])],
        example_answer_images=[_normalize_image_path(p, qid) for p in (example_answer_images or [])],
    )
    _save_config(q)
    return q


@router.delete("/{qid}")
def delete_question(qid: str) -> dict:
    d = get_questions_dir() / qid
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
    # 存相对路径，前端用 /files/ + path 访问
    rel_path = f"questions/{qid}/{dest.name}"
    q = _load_config(qid) or QuestionConfig(qid=qid, score=0)
    q.question_images.append(rel_path)
    _save_config(q)
    return {"path": rel_path}


@router.post("/{qid}/upload_example_image")
async def upload_example_image(qid: str, file: UploadFile = File(...)) -> dict:
    d = _q_dir(qid)
    existing = sorted(d.glob("example_*.png"))
    idx = len(existing)
    dest = d / f"example_{idx}.png"
    dest.write_bytes(await file.read())
    rel_path = f"questions/{qid}/{dest.name}"
    q = _load_config(qid) or QuestionConfig(qid=qid, score=0)
    q.example_answer_images.append(rel_path)
    _save_config(q)
    return {"path": rel_path}
