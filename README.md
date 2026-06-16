# AutoGraderLM

<div align="center">
  <img src="static/badge.png" alt="AutoGraderLM" width="33%" />
</div>

**AutoGraderLM** is a local web application for LLM-assisted homework grading. It uses vision-capable OpenAI-compatible models to segment student PDF submissions, grade per-question answers, support manual review, export scores, and generate per-question analysis reports.

[中文 README](README.zh-CN.md) | [Detailed Chinese usage guide](use-guide.md)

---

## What It Does

AutoGraderLM is designed for a repeated homework grading workflow:

1. Configure questions, rubrics, scores, and reference answers.
2. Load student PDF submissions from an assignment folder.
3. Segment each PDF into per-question answer images with a vision LLM.
4. Manually adjust segmentation boxes when needed.
5. Grade each question with score, confidence, summary, and comments.
6. Review low-confidence or low-score items manually.
7. Browse per-student results and rerun single-question grading.
8. Export roster-compatible score sheets and generate per-question reports.

The app runs locally by default at `127.0.0.1:8081` and stores assignment data under a local `assignment_path`.

---

## Main UI Modules

| Module | Purpose |
|--------|---------|
| Overview | Shows expected/submitted/segmented/graded counts and per-submission status. |
| Question Config | Edits question text, score, rubric, reference answer, and images. |
| Segmentation | Runs PDF segmentation and provides a manual bounding-box editor. |
| Grading | Runs full or incremental LLM grading over segmented answers. |
| Manual Review | Fixes low-confidence or low-score grading records. |
| Results | Browses each student result and supports single-question regrading. |
| Statistics | Shows roster scores, distributions, Excel export, and report generation. |
| Reports | Displays per-question analysis reports and exports report PDF. |

---

## Quick Start

```bash
# Create environment
conda create -n autograder python=3.12
conda activate autograder

# Install dependencies
pip install -r requirements.txt

# Configure LLM credentials
echo "OPENAI_API_KEY=your-api-key" > .env
echo "OPENAI_BASE_URL=https://api.openai.com/v1" >> .env

# Copy and edit the example assignment config
cp config.example.yaml config.yaml

# Start the app
python main.py -c config.yaml
```

Open <http://localhost:8081>.

The example config uses:

```yaml
assignment_path: ./assignment_example/
web_server:
  host: "127.0.0.1"
  port: 8081
```

Use `0.0.0.0` only when intentionally exposing the app on a trusted LAN. The app has no built-in authentication.

---

## Assignment Folder Layout

Each config points to one assignment folder via `assignment_path`.

```text
assignment_example/
├── res/           # Student PDFs: {student_id}_{name}.pdf or {student_id}_{name}_*.pdf
├── in.xls         # Optional roster spreadsheet
├── questions/     # Created by the app
├── answers/       # Created by segmentation
└── results/       # Created by grading/report generation
```

Recommended PDF naming:

```text
2023000001_ZhangSan.pdf
2023000002_LiSi_1234.pdf
```

Only the first two underscore-separated parts are used as the canonical student key: `student_id_name`. This lets a newer PDF for the same student overwrite previous segmentation/results cleanly.

Generated assignment data and local configs are ignored by Git.

---

## Configuration

Start from [config.example.yaml](config.example.yaml). Important fields:

| Field | Meaning |
|-------|---------|
| `assignment_name` | Display name in the UI. |
| `assignment_path` | Local folder containing one assignment. |
| `web_server.host` / `port` | Local bind address and port. |
| `assignment_configuration.pdf_folder_path` | PDF folder relative to `assignment_path`. |
| `assignment_configuration.excel_in_path` | Roster file path relative to `assignment_path`. |
| `assignment_segmentation` | Segmentation model, prompt template, concurrency, retries. |
| `assignment_grading` | Grading model, prompt template, concurrency, retries. |
| `assignment_regrade.add_to_regrade_when_below` | Low-score threshold for manual review queue. |
| `assignment_report` | Report model, export path, concurrency, vision limits. |
| `llm_log` | Optional LLM request/response logging for debugging. |

For multiple assignments, keep separate local config files:

```bash
python main.py -c config.assignment1.yaml
python main.py -c config.assignment2.yaml
```

---

## Data Flow

```text
Question config
  -> questions/{Qid}/config.json

Segmentation
  -> answers/{student_id_name}/{Qid}-{idx}.png
  -> answers/{student_id_name}/_pages/segments.json

Grading and review
  -> results/{student_id_name}.json

Reports and export
  -> results/reports/
  -> configured Excel output path
```

The current public tree does not track assignment-specific `config_hw*.yaml`, `hw*/`, `.env`, logs, results, or generated answer files.

---

## Architecture

```text
AutoGraderLM/
├── main.py                 # CLI entrypoint, loads -c/--config
├── config.example.yaml     # Generic config template
├── autograder/
│   ├── app.py              # FastAPI app factory and static mounts
│   ├── config.py           # YAML/env config loading and path resolution
│   ├── llm.py              # OpenAI-compatible multimodal LLM wrapper
│   ├── models.py           # Pydantic models
│   ├── pdf_utils.py        # PDF rendering and cropping
│   ├── excel_utils.py      # Roster reading/export
│   ├── pipeline/
│   │   ├── segmentation.py # PDF -> answer image segmentation
│   │   ├── grading.py      # LLM grading graph
│   │   └── report.py       # Report generation
│   └── routers/            # REST API routers
├── prompts/                # Jinja2 prompt templates
├── static/index.html       # Vue 3 single-page app
├── use-guide.md            # Detailed Chinese guide
└── requirements.txt
```

Backend: FastAPI, LangGraph, LangChain OpenAI, PyMuPDF, Pillow, Jinja2.  
Frontend: Vue 3 + Tailwind from CDN in `static/index.html`.

---

## Notes For Public Use

- Use an OpenAI-compatible endpoint with vision support.
- The frontend loads Vue/Tailwind/Marked/KaTeX from public CDNs, so internet access is needed for the UI unless those assets are vendored.
- Keep `.env` local; `/api/config` does not return API keys.
- Raw `results/` files are not statically exposed by the app.
- The default server host is `127.0.0.1`. Treat `0.0.0.0` as a trusted-network setting.

---

## License

See [LICENSE](LICENSE).
