# AutoGraderLM

<div align="center">
  <img src="static/badge.png" alt="AutoGraderLM" width="33%" />
</div>

**AutoGraderLM** 是一个本地运行的 LLM 辅助作业评阅系统。它面向 PDF 作业批改场景，使用支持视觉输入的 OpenAI 兼容模型完成作答区域切分、逐题评分、人工复核、成绩导出和答题报告生成。

[English README](README.md) | [详细使用指南](use-guide.md)

---

## 功能概览

AutoGraderLM 覆盖一套完整的作业评阅流程：

1. 配置题目、分值、评分标准和参考答案。
2. 从作业目录读取学生 PDF。
3. 调用视觉 LLM 将 PDF 切分为逐题作答图片。
4. 在网页中手动调整切分框。
5. 调用 LLM 输出每题得分、置信度、总结和评语。
6. 对低置信度或低分作答进行人工复核。
7. 按学生查看评阅结果，并支持单题重新评阅。
8. 导出成绩表，生成按题目的答题分析报告和 PDF 报告。

应用默认只监听本机 `127.0.0.1:8081`，所有作业数据保存在本地 `assignment_path` 下。

---

## 页面模块

| 模块 | 作用 |
|------|------|
| 作业概览 | 查看应交、已提交、已切分、已评阅数量，以及每份作业状态。 |
| 作业配置 | 编辑题干、分值、评分标准、参考答案和图片。 |
| 作业解析 | 执行 PDF 切分，并提供人工框选编辑器。 |
| 作业评分 | 对切分后的作答进行全量或增量 LLM 评分。 |
| 人工复核 | 修正低置信度或低分评阅记录。 |
| 评阅结果 | 按学生浏览每题结果，支持单题 AI 重新评阅。 |
| 评分统计 | 汇总成绩、展示分布、导出 Excel、触发报告生成。 |
| 答题报告 | 查看每题得分分布和共性错误分析，并导出 PDF。 |

---

## 快速开始

```bash
# 创建环境
conda create -n autograder python=3.12
conda activate autograder

# 安装依赖
pip install -r requirements.txt

# 配置 LLM API
echo "OPENAI_API_KEY=your-api-key" > .env
echo "OPENAI_BASE_URL=https://api.openai.com/v1" >> .env

# 复制并修改示例配置
cp config.example.yaml config.yaml

# 启动应用
python main.py -c config.yaml
```

浏览器打开 <http://localhost:8081>。

示例配置默认包含：

```yaml
assignment_path: ./assignment_example/
web_server:
  host: "127.0.0.1"
  port: 8081
```

只有在可信局域网中明确需要对外访问时，才建议把 `host` 改为 `0.0.0.0`。当前应用没有内置登录认证。

---

## 作业目录结构

每个配置文件对应一个作业目录，即 `assignment_path`。

```text
assignment_example/
├── res/           # 学生 PDF：{学号}_{姓名}.pdf 或 {学号}_{姓名}_*.pdf
├── in.xls         # 可选学生名单
├── questions/     # 应用自动生成
├── answers/       # 切分后生成
└── results/       # 评分和报告生成后写入
```

推荐 PDF 命名：

```text
2023000001_张三.pdf
2023000002_李四_1234.pdf
```

系统只使用文件名中前两个下划线分隔字段作为学生唯一标识，即 `学号_姓名`。如果同一学生重新提交 PDF，重新切分和评分会覆盖对应的 `answers/` 与 `results/`。

本地作业数据、日志、`.env` 和本地配置不会被 Git 跟踪。

---

## 配置说明

从 [config.example.yaml](config.example.yaml) 复制一份本地配置后修改。主要字段如下：

| 字段 | 说明 |
|------|------|
| `assignment_name` | 页面展示的作业名称。 |
| `assignment_path` | 当前作业的本地目录。 |
| `web_server.host` / `port` | 服务监听地址和端口。 |
| `assignment_configuration.pdf_folder_path` | PDF 目录，相对 `assignment_path`。 |
| `assignment_configuration.excel_in_path` | 学生名单路径，相对 `assignment_path`。 |
| `assignment_segmentation` | 切分模型、prompt、并发数、重试次数。 |
| `assignment_grading` | 评分模型、prompt、并发数、重试次数。 |
| `assignment_regrade.add_to_regrade_when_below` | 自动进入人工复核的低分阈值。 |
| `assignment_report` | 报告模型、导出路径、并发数和图片限制。 |
| `llm_log` | 可选的大模型请求和响应日志。 |

多个作业可以使用多个本地配置：

```bash
python main.py -c config.assignment1.yaml
python main.py -c config.assignment2.yaml
```

---

## 数据流

```text
作业配置
  -> questions/{Qid}/config.json

作业解析
  -> answers/{学号_姓名}/{Qid}-{idx}.png
  -> answers/{学号_姓名}/_pages/segments.json

作业评分和人工复核
  -> results/{学号_姓名}.json

报告和导出
  -> results/reports/
  -> 配置中的 Excel 输出路径
```

当前公开仓库不跟踪具体作业的 `config_hw*.yaml`、`hw*/`、`.env`、日志、评分结果或切分图片。

---

## 项目结构

```text
AutoGraderLM/
├── main.py                 # 命令行入口，读取 -c/--config
├── config.example.yaml     # 通用配置模板
├── autograder/
│   ├── app.py              # FastAPI 应用和静态资源挂载
│   ├── config.py           # YAML/env 配置加载和路径解析
│   ├── llm.py              # OpenAI 兼容多模态 LLM 封装
│   ├── models.py           # Pydantic 数据模型
│   ├── pdf_utils.py        # PDF 渲染和裁剪
│   ├── excel_utils.py      # 名单读取和成绩导出
│   ├── pipeline/
│   │   ├── segmentation.py # PDF -> 逐题作答图片
│   │   ├── grading.py      # LLM 评分流程
│   │   └── report.py       # 答题报告生成
│   └── routers/            # REST API 路由
├── prompts/                # Jinja2 prompt 模板
├── static/index.html       # Vue 3 单页应用
├── use-guide.md            # 详细中文使用指南
└── requirements.txt
```

后端使用 FastAPI、LangGraph、LangChain OpenAI、PyMuPDF、Pillow、Jinja2。  
前端是 `static/index.html` 中的 Vue 3 + Tailwind 单页应用。

---

## 公开使用注意事项

- 需要支持视觉输入的 OpenAI 兼容模型。
- 前端依赖 Vue/Tailwind/Marked/KaTeX 的公共 CDN；若在离线环境使用，需要自行本地化这些资源。
- `.env` 只应保存在本地；`/api/config` 不会返回 API key。
- `results/` 原始结果文件不会通过静态文件接口暴露。
- 默认只监听 `127.0.0.1`；把服务暴露到局域网前，请确认网络环境可信。

---

## 许可证

见 [LICENSE](LICENSE)。
