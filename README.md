# AutoGraderLM

**Automatic LLM-based Grader for Homework Grading** — A web application that uses vision-capable LLMs to segment, grade, and analyze student PDF submissions with a graphical interface.

---

## Overview

AutoGraderLM automates the homework grading workflow:

1. **Configure** questions (stem, rubric, reference answers)
2. **Segment** PDFs into per-question images via LLM(also supports manual segmentation)
3. **Grade** each answer with scores, confidence, and comments
4. **Review** low-confidence items manually
5. **Export** scores and generate per-question analysis reports

See [use-guide.md](use-guide.md) for detailed usage (in Chinese).

---

## Quick Start

```bash
# Create environment
conda create -n autograder python=3.12
conda activate autograder

# Install dependencies
pip install -r requirements.txt

# Create .env with your LLM API credentials
echo "OPENAI_API_KEY=sk-your-key" > .env
echo "OPENAI_BASE_URL=https://api.openai.com/v1" >> .env

# Run (specify config for the assignment)
python main.py -c config_hw2.yaml
```

Open **http://localhost:8081** in your browser.

---

## Deployment

### Requirements

- Python 3.12+
- OpenAI-compatible API (vision model support)

### Steps

1. Clone the repo and `cd` into it
2. Create a virtual environment (conda or venv)
3. `pip install -r requirements.txt`
4. Add `.env` with `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL`
5. Prepare assignment directory (see [use-guide.md](use-guide.md))
6. Run: `python main.py -c <config.yaml>`

### Switching Assignments

Use different config files per assignment:

```bash
python main.py -c config_hw1.yaml
python main.py -c config_hw2.yaml
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           AutoGraderLM                                   │
├─────────────────────────────────────────────────────────────────────────┤
│  Frontend (Vue 3 + Tailwind)  │  Backend (FastAPI)                       │
│  static/index.html            │  autograder/                             │
│  - Question config            │  - app.py (routers, static mounts)       │
│  - Segmentation editor        │  - routers/: questions, pipeline,        │
│  - Review UI                  │    review, results, stats                │
│  - Results & stats            │  - pipeline/: segmentation, grading,     │
│  - Reports                    │    report                                │
└─────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LangGraph pipelines  │  LangChain (OpenAI-compatible)  │  PyMuPDF      │
│  - Segmentation       │  - Vision messages              │  - PDF split  │
│  - Grading (retry)    │  - JSON extraction              │  - Page crop │
│  - Report generation  │  - invoke_with_log                │               │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Role |
|-----------|------|
| `main.py` | Entry point; loads config via `-c`, runs uvicorn |
| `autograder/config.py` | YAML config + env vars; resolves paths under `assignment_path` |
| `autograder/llm.py` | LangChain ChatOpenAI wrapper; vision messages; JSON extraction |
| `autograder/pipeline/` | LangGraph workflows: segmentation, grading, report |
| `autograder/routers/` | REST API for questions, pipeline, review, results, stats |
| `prompts/*.jinja` | Jinja2 templates for LLM prompts |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| FastAPI, uvicorn | Web server |
| LangChain, LangGraph | LLM orchestration, pipeline graphs |
| PyMuPDF, Pillow | PDF rendering, image handling |
| Jinja2 | Prompt templates |
| openpyxl, xlrd, xlwt | Excel read/write (.xls, .xlsx) |
| PyYAML, python-dotenv | Config and env |

See [requirements.txt](requirements.txt) for versions.

---

## Project Structure

```
AutoGraderLM/
├── main.py                 # Entry point
├── config_hw1.yaml         # Assignment 1 config
├── config_hw2.yaml         # Assignment 2 config
├── .env                    # OPENAI_API_KEY, OPENAI_BASE_URL (gitignored)
├── autograder/
│   ├── app.py              # FastAPI app factory
│   ├── config.py           # Config loading, path resolution
│   ├── llm.py              # LLM client, vision messages
│   ├── models.py           # Pydantic models
│   ├── pdf_utils.py        # PDF parsing, student info
│   ├── excel_utils.py      # Roster read/write
│   ├── pipeline/
│   │   ├── segmentation.py # PDF → per-question images
│   │   ├── grading.py       # LLM grading with retry
│   │   └── report.py       # Per-question reports
│   └── routers/
│       ├── questions.py    # Question CRUD
│       ├── pipeline.py     # Segment, grade, report, segment editor
│       ├── review.py       # Manual review
│       ├── results.py      # Per-student results
│       └── stats.py        # Roster, export, summary
├── prompts/
│   ├── segmentation*.jinja # Segmentation prompts
│   ├── grading*.jinja      # Grading prompts
│   └── report.jinja        # Report prompt
├── static/
│   └── index.html          # SPA (Vue 3)
├── demand-analysis.md      # Requirements (Chinese)
├── use-guide.md            # Usage guide (Chinese)
└── requirements.txt
```

Per-assignment data (under `assignment_path`, e.g. `hw2/`):

```
hw2/
├── res/           # Student PDFs ({学号}_{姓名}.pdf or {学号}_{姓名}_*.pdf; 学号_姓名 used as key)
├── in.xls         # Student roster (学号, 姓名, 成绩, 评语)
├── questions/     # Question configs (Q1, Q2, ...)
├── answers/       # Segmented images per student (dirs named 学号_姓名; overwrite on re-run)
└── results/      # Grading JSON per student (学号_姓名.json; overwrite on re-run)
```

---

## License

See [LICENSE](LICENSE).
