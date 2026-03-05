"""Pydantic data models for AutoGraderLM."""

from __future__ import annotations

from enum import IntEnum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


# ── Question Configuration ──────────────────────────────────────────


class QuestionConfig(BaseModel):
    qid: str
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


class GradingRecord(BaseModel):
    filename: str
    qid: str
    answer: list[str] = Field(default_factory=list)
    grader: str = ""
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
    score_distribution: dict[int, int] = Field(default_factory=dict)
    report_text: str = ""
