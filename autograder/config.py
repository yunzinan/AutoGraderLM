"""Configuration management – loads config.yaml and env vars."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

# 优先从项目根目录加载 .env（main.py 所在目录）
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_path)

CONFIG_PATH = Path("config.yaml")
BASE_DIR = Path(".")


class LLMConfig(BaseModel):
    model: str = "gemini-3-flash-preview"
    endpoint: str = "/v1/chat/completions"
    prompt_template: str = ""
    base_url: str = ""
    api_key: str = ""


class WebServerConfig(BaseModel):
    port: int = 8081


class AssignmentConfigSection(BaseModel):
    num_question: int = 5
    num_student: int = 5
    pdf_folder_path: str = "./res/"
    xlsx_in_path: str = ""


class SegmentationConfig(BaseModel):
    num_workers: int = 5
    max_retry: int = 3
    llm: LLMConfig = Field(default_factory=LLMConfig)


class GradingConfig(BaseModel):
    num_workers: int = 5
    max_retry: int = 3
    llm: LLMConfig = Field(default_factory=LLMConfig)


class ReportConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    xlsx_out_path: str = ""


class LlmLogConfig(BaseModel):
    """大模型对话日志，用于后台调试。"""
    enabled: bool = False
    path: str = "./logs/llm_dialogue.log"


class AppConfig(BaseModel):
    assignment_name: str = "Assignment 1"
    web_server: WebServerConfig = Field(default_factory=WebServerConfig)
    assignment_configuration: AssignmentConfigSection = Field(default_factory=AssignmentConfigSection)
    assignment_segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    assignment_grading: GradingConfig = Field(default_factory=GradingConfig)
    assignment_report: ReportConfig = Field(default_factory=ReportConfig)
    llm_log: LlmLogConfig = Field(default_factory=LlmLogConfig)


def _normalize_base_url(url: str) -> str:
    """将「完整 endpoint URL」转为 OpenAI 客户端需要的 base_url。

    例如: https://poloai.top/v1/chat/completions -> https://poloai.top/v1
    客户端会自行追加 /chat/completions。
    """
    url = (url or "").strip().rstrip("/")
    if not url:
        return "https://api.openai.com/v1"
    # 若用户填的是完整 endpoint，去掉末尾的 /chat/completions
    if url.endswith("/chat/completions"):
        url = url[: -len("/chat/completions")].rstrip("/")
    # 若没有版本前缀，保留原样（部分代理只暴露 /v1）
    return url or "https://api.openai.com/v1"


def _inject_env(llm: LLMConfig) -> None:
    """从环境变量填充 base_url / api_key（.env 已通过 dotenv 加载）。"""
    if not llm.base_url:
        llm.base_url = _normalize_base_url(os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    if not llm.api_key:
        llm.api_key = os.getenv("OPENAI_API_KEY", "")


def load_config(path: Path | str = CONFIG_PATH) -> AppConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = AppConfig(**raw)
    for llm_cfg in (
        cfg.assignment_segmentation.llm,
        cfg.assignment_grading.llm,
        cfg.assignment_report.llm,
    ):
        _inject_env(llm_cfg)
    return cfg


_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config
