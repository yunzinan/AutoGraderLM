"""Configuration management – loads config from -c/--config and env vars."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field

# 优先从项目根目录加载 .env（main.py 所在目录）
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_path)

DEFAULT_CONFIG_PATH = Path("config.yaml")
_config_file: Optional[Path] = None
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
    pdf_folder_path: str = "./res/"
    excel_in_path: str = ""


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
    excel_out_path: str = ""
    num_workers: int = Field(default=3, ge=1, le=16, description="并发生成报告的题目数")


class AssignmentRegradeConfig(BaseModel):
    """全量评分后，得分 <= ratio * 题目总分的作答自动进入人工复核列表。"""
    add_to_regrade_when_below: float = Field(default=0.2, ge=0.0, le=1.0)


class LlmLogConfig(BaseModel):
    """大模型对话日志，用于后台调试。"""
    enabled: bool = False
    path: str = "./logs/llm_dialogue.log"


class AppConfig(BaseModel):
    assignment_name: str = "Assignment 1"
    assignment_path: str = "."
    web_server: WebServerConfig = Field(default_factory=WebServerConfig)
    assignment_configuration: AssignmentConfigSection = Field(default_factory=AssignmentConfigSection)
    assignment_segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    assignment_grading: GradingConfig = Field(default_factory=GradingConfig)
    assignment_regrade: AssignmentRegradeConfig = Field(default_factory=AssignmentRegradeConfig)
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


def _resolve_under_assignment(assignment_base: Path, subpath: str) -> str:
    """将相对路径解析到 assignment_base 下；已是绝对路径则原样返回。"""
    p = Path(subpath)
    if p.is_absolute():
        return subpath
    return str((assignment_base / subpath).resolve())


def load_config(path: Path | str | None = None) -> AppConfig:
    path = Path(path or _config_file or DEFAULT_CONFIG_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    config_dir = path.resolve().parent
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    assignment_path_raw = raw.get("assignment_path", ".")
    # 以配置文件所在目录为基准，解析 assignment_path，保证无论从哪启动都能找到文件
    assignment_base = (config_dir / assignment_path_raw).resolve()
    assignment_path = str(assignment_base)
    raw["assignment_path"] = assignment_path
    # 将 res / excel_in / excel_out 解析到 assignment_path 下
    ac = raw.get("assignment_configuration") or {}
    if "pdf_folder_path" in ac:
        ac["pdf_folder_path"] = _resolve_under_assignment(assignment_base, ac["pdf_folder_path"])
    if "excel_in_path" in ac:
        ac["excel_in_path"] = _resolve_under_assignment(assignment_base, ac["excel_in_path"])
    raw["assignment_configuration"] = ac
    ar = raw.get("assignment_report") or {}
    if "excel_out_path" in ar:
        ar["excel_out_path"] = _resolve_under_assignment(assignment_base, ar["excel_out_path"])
    raw["assignment_report"] = ar
    # 兼容拼写：assignmetn_regrade -> assignment_regrade
    regrade = raw.get("assignment_regrade") or raw.get("assignmetn_regrade")
    if regrade is not None:
        raw["assignment_regrade"] = regrade
    cfg = AppConfig(**raw)
    for llm_cfg in (
        cfg.assignment_segmentation.llm,
        cfg.assignment_grading.llm,
        cfg.assignment_report.llm,
    ):
        _inject_env(llm_cfg)
    return cfg


_config: Optional[AppConfig] = None


def set_config_file(path: str | Path) -> None:
    """通过 main.py -c/--config 指定配置文件路径。"""
    global _config_file
    _config_file = Path(path)
    # 清空已加载的配置，下次 get_config() 会重新从新路径加载
    global _config
    _config = None


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def get_assignment_path() -> Path:
    """当前作业根目录（res/questions/answers/results 等均在其下）。"""
    return Path(get_config().assignment_path)


def resolve_assignment_path(path: str) -> Path:
    """将配置中的相对路径（如 questions/Q1/question_0.png）解析为绝对路径；已是绝对路径则原样返回。"""
    p = Path(path)
    if p.is_absolute():
        return p
    return get_assignment_path() / path


def to_relative_url_path(path: str) -> str:
    """将绝对路径转为相对 assignment 的 URL 路径，供前端 /files/ + path 使用。"""
    if not path:
        return path
    p = Path(path)
    base = get_assignment_path()
    if p.is_absolute():
        try:
            rel = p.relative_to(base)
            return str(rel).replace("\\", "/")
        except ValueError:
            return path
    return path


def get_questions_dir() -> Path:
    return get_assignment_path() / "questions"


def get_answers_dir() -> Path:
    return get_assignment_path() / "answers"


def get_results_dir() -> Path:
    return get_assignment_path() / "results"


def get_num_questions() -> int:
    """根据 {assignment_path}/questions 下有效题目配置数量推断题目数（每个子目录含 config.json 计一题）。"""
    qdir = get_questions_dir()
    if not qdir.exists():
        return 0
    return sum(
        1 for d in qdir.iterdir()
        if d.is_dir() and (d / "config.json").exists()
    )


def get_num_student() -> int:
    """以 pdf_folder_path 下 PDF 数量作为学生数（PDF 为 excel 名单的子集，表格照常加载）。"""
    cfg = get_config()
    pdf_dir = Path(cfg.assignment_configuration.pdf_folder_path)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"pdf_folder_path 不存在: {pdf_dir}")
    return sum(1 for _ in pdf_dir.glob("*.pdf")) + sum(1 for _ in pdf_dir.glob("*.PDF"))
