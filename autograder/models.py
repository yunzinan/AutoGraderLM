"""Pydantic data models for AutoGraderLM."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


# ── Question Configuration ──────────────────────────────────────────


class QuestionConfig(BaseModel):
    qid: str
    question_index: Optional[int] = Field(
        default=None,
        description="题目在习题集中的序号，学生答卷上可能写该序号（如 2. 4. 7. 8.），留空表示按作业内顺序 1、2、3…",
    )
    score: int = Field(ge=0, le=100)
    rubric: str = ""
    question_text: str = ""
    question_images: list[str] = Field(default_factory=list)
    example_answer_text: str = ""
    example_answer_images: list[str] = Field(default_factory=list)


# ── Segmentation ────────────────────────────────────────────────────


class BBox(BaseModel):
    page: int = Field(ge=1)
    bbox: list[int] = Field(min_length=4, max_length=4)


class QuestionRegion(BaseModel):
    qid: str
    regions: list[BBox]


class SegmentationResult(BaseModel):
    questions: list[QuestionRegion]


# ── Grading ─────────────────────────────────────────────────────────


class Confidence(IntEnum):
    LOST_ALL = 0
    LOST_PARTIAL = 1
    COMPLETE_UNSURE = 2
    COMPLETE_MOSTLY = 3
    COMPLETE_FULLY = 4


class GradingOutput(BaseModel):
    score: int = Field(ge=0)
    confidence: int = Field(ge=0, le=4)
    summary: str = ""
    comments: str = ""


def _now_iso() -> str:
    """返回当前时间的 ISO 8601 格式字符串。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class GradingRecord(BaseModel):
    filename: str
    qid: str
    """题目在习题集中的序号，与 QuestionConfig.question_index 一致"""
    question_index: Optional[int] = None
    answer: list[str] = Field(default_factory=list)
    grader: str = ""
    """评阅人：LLM 时为模型名，人工复核后为 'human'"""
    graded_at: Optional[str] = None
    """评阅时间，ISO 8601 格式"""
    score: int = Field(ge=0, default=0)
    confidence: int = Field(ge=0, le=4, default=0)
    summary: str = ""
    comments: str = ""


class StudentResult(BaseModel):
    filename: str
    student_id: str = ""
    student_name: str = ""
    records: list[GradingRecord] = Field(default_factory=list)
    total_score: int = 0


# ── Pipeline Status ─────────────────────────────────────────────────


class PipelineStatus(BaseModel):
    stage: str = "idle"
    message: str = ""
    current: int = 0
    total: int = 0
    errors: list[str] = Field(default_factory=list)


# ── Question Report ─────────────────────────────────────────────────


class QuestionReport(BaseModel):
    qid: str
    """题目在习题集中的序号"""
    question_index: Optional[int] = None
    score_distribution: dict[int, int] = Field(default_factory=dict)
    report_text: str = ""
    reference_answer: str = ""
    student_answers: dict[str, list[str]] = Field(
        default_factory=dict,
        description="学生姓名 → 该题作答图片路径列表，供前端 auto-link 弹出作答图片",
    )
