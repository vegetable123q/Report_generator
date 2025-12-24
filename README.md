# TianGong AI Workspace

## 项目简介
- 通用的 AI CLI 工作区，用于统一管理 Codex、Gemini CLI、Claude Code 等智能体工具，以及日常文档创作任务。
- 基于 `uv` 管理 Python 依赖，内置 LangChain、LangGraph 等现代化 Agent 组件，便于快速扩展。
- 提供 `tiangong-workspace` 命令行应用：可查看环境信息、执行文档工作流、触发 Tavily 联网检索、调用 Crossref/OpenAlex 文献元数据、运行自主工作流智能体。
- 集成 Tavily MCP 搜索 + LangGraph 自主 Agent，支持联网调研、Shell/Python 执行、LangChain 工作流协同，能够灵活处理未预设的复杂任务。
- 跨平台安装脚本覆盖 Ubuntu、macOS 与 Windows，可按需安装 Node.js、Pandoc/MiKTeX 等可选组件。

## 目录结构
- `install_*.sh` / `install_windows.ps1`：一键安装脚本。
- `src/tiangong_ai_workspace/`：工作区 Python 包与 CLI 入口。
  - `cli.py`：Typer CLI，包含 `docs`、`agents`、`research`、`crossref`、`openalex` 与 `mcp` 子命令。
  - `agents/`：LangGraph 文档工作流 (`workflows.py`)、LangGraph/DeepAgents 双引擎自主智能体 (`deep_agent.py`)、具备 Pydantic 入参与输出校验的 LangChain Tool 封装 (`tools.py`)。
  - `tooling/`：响应封装、工作区配置加载 (`config.py`)、工具注册表、模型路由器 (`llm.py`)、统一 Tool Schema (`tool_schemas.py`)、Tavily MCP 搜索客户端、Crossref Works API 客户端 (`crossref.py`)、OpenAlex Works/Cited-by 客户端 (`openalex.py`)、Dify 知识库客户端 (`dify.py`)、Neo4j 图数据库客户端 (`neo4j.py`) 以及带审计的 Shell/Python 执行器。
  - `templates/`：不同文档类型的结构提示。
  - `mcp_client.py`：同步封装的 MCP 客户端。
  - `secrets.py`：凭证加载逻辑。
- `pyproject.toml`：项目元数据与依赖定义（请使用 `uv add/remove` 维护）。
- `AGENTS.md` / `README.md`：面向 Agent 与用户的人机协作指南。

## 快速安装
1. 克隆或下载本仓库。
2. 根据操作系统执行对应脚本（支持 `--full`、`--minimal`、`--with-pdf`、`--with-node` 等参数）：

```bash
# Ubuntu
bash install_ubuntu.sh --full

# macOS
bash install_macos.sh --full

# Windows（在管理员 PowerShell 中）
PowerShell -ExecutionPolicy Bypass -File install_windows.ps1 -Full
```

脚本会检查/安装：
- Python 3.12+
- Astral `uv`
- Node.js 22+（仅在需要 Node 生态 CLI 时安装）
- Pandoc + LaTeX/MiKTeX（用于 PDF/DOCX 导出，可选）

## 手动安装
若不使用脚本，可手动执行以下步骤：

```bash
uv sync                 # 安装 Python 依赖
uv run tiangong-workspace info
```

如需激活虚拟环境可运行 `source .venv/bin/activate`（在 Windows 上为 `.venv\Scripts\activate`）。

## CLI 使用
常用命令：

```bash
uv run tiangong-workspace --help
uv run tiangong-workspace info          # 查看版本、项目路径等信息
uv run tiangong-workspace check         # 检查 Python、uv、Node.js 以及外部 CLI
uv run tiangong-workspace tools         # 查看已配置的外部 CLI 列表
uv run tiangong-workspace tools --catalog   # 查看内部工作流与工具注册表
uv run tiangong-workspace agents list       # 查看自主智能体与运行时代码执行器
uv run tiangong-workspace knowledge retrieve "查询关键词"  # 直接检索 Dify 知识库
uv run tiangong-workspace newsletter generate --output-dir outputs  # 生成法规 newsletter（含柱状图+政策表）
uv run tiangong-workspace crossref journal-works "1234-5678" --query "LLM"  # 查询 Crossref 期刊文献
uv run tiangong-workspace openalex work "10.1016/S0921-3449(00)00060-4"      # 获取 OpenAlex 元数据
uv run tiangong-workspace openalex cited-by "W2072484418" --from 2020-01-01 --to 2021-01-01  # 时间窗引用统计
uv run tiangong-workspace embeddings generate "示例文本"   # 调用 OpenAI 兼容 embedding 服务
```

所有支持的命令都提供 `--json` 选项，可输出结构化响应，方便被其他智能体消费。

### 工作区配置
- `pyproject.toml` 中的 `[tool.tiangong.workspace.cli_tools]` / `[tool.tiangong.workspace.tool_registry]` 控制 CLI 检测与 Agent Catalog，无需修改源码即可扩充。
- 注册表中的工具会自动附带输入/输出 JSON Schema，`tiangong-workspace tools --catalog` 会展示完整结构，方便其他智能体静态校验参数。

## 自主智能体与运行时执行
`agents` 子命令使用 LangGraph 构建的多工具智能体，可根据任务动态规划、调用 Shell/Python、联网检索并生成文档：

```bash
uv run tiangong-workspace agents run "为新能源项目生成市场调研与实施计划"
uv run tiangong-workspace agents run "统计 data.csv 中的指标并绘图" --no-tavily
```

默认工具：
- Shell 执行器：运行命令行工具、脚本、`uv`/`git` 等指令，返回结构化 stdout/stderr。
- Python 执行器：在共享解释器中运行脚本，可直接使用 `pandas`、`matplotlib`、`seaborn` 等依赖。
- Tavily 搜索：通过 MCP 获取实时互联网情报。
- Crossref 文献：调用 Crossref `/journals/{issn}/works` 接口查询指定期刊文章。
- OpenAlex 文献：按 DOI 获取元数据，或按时间窗口统计引用（cites + from/to_publication_date）。
- Dify 知识库：在本地 HTTP 直连指定 Dify 数据集以获取企业知识，无需 MCP。
- LangGraph 文档工作流：生成报告、计划书、专利交底书、项目申报书。
- Neo4j 图数据库：通过 `neo4j` 官方驱动执行 Cypher，并支持 create/read/update/delete 全流程操作。
- OpenAI 兼容向量嵌入：调用本地或远程 OpenAI API 生成 embedding，默认返回结构化 JSON，可直接写入向量工作流。

可使用 `--no-shell`、`--no-python`、`--no-tavily`、`--no-dify`、`--no-crossref`、`--no-openalex`、`--no-document` 分别关闭对应工具；`--engine langgraph|deepagents` 切换运行后端；`--system-prompt` 和 `--model` 可自定义智能体设定。

## 文档工作流
`docs` 子命令调用 LangGraph 工作流（检索→大纲→草稿→AI 审核[可选]），支持报告、计划书、专利交底书、项目申报书等：

```bash
uv run tiangong-workspace docs list

uv run tiangong-workspace docs run report \
  --topic "季度项目总结" \
  --audience "研发团队与产品负责人" \
  --instructions "突出关键指标变化，附带行动建议"
```

常用参数：
- `--skip-research`：跳过 Tavily 联网检索。
- `--search-query`：自定义检索关键字（默认使用 topic）。
- `--purpose`：模型用途提示（`general`、`deep_research`、`creative`）。
- `--language`：设置输出语言（默认中文）。
- `--ai-review`：在草稿生成后执行一次自动评审，输出可执行的修改建议。

## 法规 Newsletter 自动生成
- 命令：`uv run tiangong-workspace newsletter generate --output-dir outputs`
- 首次生成：加 `--first-run` 时仅展示 Impact（High/Medium/Low）单柱堆叠，不显示 Previous。
- 工作流：加 `--workflow 2025` 时仅统计实施日期在 `2024-01` 到 `2025-12`（含）范围内的条目；图表改为单系列柱状图（不堆叠），Policy Table 仅按日期窗口筛选。
- Policy Table：默认最多输出 12 条（可用 `--max-policies` 覆盖）；默认启用 `--ai-emphasis` 仅用于用 LLM 将摘要压缩为 50 字以内的“一句完整短句”（不使用省略号；不可用时回退到规则摘要）；加粗强调由 Codex 执行 Prompt 在生成后统一处理，且只允许出现在「内容概要/链接」与「Key Insights」中。
- 实施日期：只保留不超过 `2025-12` 的完整日期，并优先选择更接近 2025 年的条目。
- 产出：`regulation_update_chart.png`（簇状堆叠柱形图）+ `regulation_newsletter.md`（含 Policy Table）。
- 图表：默认为簇状堆叠（Previous 灰色单柱 vs Impact 堆叠），`--first-run` 仅显示 Impact 单柱堆叠；图例位于图内右上角。
- Word：默认额外导出 `regulation_newsletter.docx`（`--no-docx` 可关闭），图表会被直接嵌入到 Word 中。
- Word（仅导出）：`uv run tiangong-workspace newsletter export-docx --markdown-path outputs/regulation_newsletter.md --output-dir outputs`
- 政策表：生态设计指标类别、政策名称、内容概要/链接、适用产品、适用行业、实施日期；适用产品/适用行业会自动做英文→中文映射，且适用产品超过 5 项时会自动精简并追加“等电力装备”；Word 导出会统一将字体设为黑色并应用三线表样式。
- 图表标题：Word 导出会将 `Regulation Update Chart` 标题居中且不额外加粗（加粗仅来自 Markdown 的 `**...**`）。
- Codex 执行 Prompt：`prompts/newsletter/workflow.md`（按该文件步骤即可一键生成最终 Word 交付物）。
- Codex 执行 Prompt（2024-01~2025-12）：`prompts/newsletter/workflow_2025.md`。

执行成功后 CLI 会输出结构化结果，包含草稿，启用 `--ai-review` 时还会附带评审意见。

## 联网研究
`research` 命令直接调用 Tavily MCP 工具，可独立进行资料搜集：

```bash
uv run tiangong-workspace research "新能源车市场份额"
uv run tiangong-workspace research "AI 写作辅助工具对比" --json
```

如需使用自定义 MCP 服务名称或工具名称，可分别通过 `--service` 与 `--tool-name` 覆盖。

## Crossref 文献
`crossref` 子命令直接访问 Crossref Works API 的 `/journals/{issn}/works` 接口，便于在智能体中查询期刊文献元数据：

```bash
uv run tiangong-workspace crossref journal-works "1234-5678" \
  --query "large language model" \
  --filters '{"from-pub-date": "2023-01-01"}' \
  --rows 5 \
  --select '["title", "DOI", "author"]' \
  --mailto me@example.com
```

`--filters` 接受 JSON 对象/数组或直接传入 Crossref filter 字符串（如 `from-pub-date:2020-01-01,until-pub-date:2020-12-31`），`--select` 支持 JSON 数组或逗号分隔字段。推荐传入 `--mailto` 以符合 Crossref 最佳实践。

## OpenAlex 文献/引用
`openalex` 子命令用于快速获取元数据与引用统计：

```bash
# 按 DOI 查元数据
uv run tiangong-workspace openalex work "10.1016/S0921-3449(00)00060-4" --mailto me@example.com

# 统计时间窗口内的引用（需 OpenAlex work_id，如 W2072484418）
uv run tiangong-workspace openalex cited-by "W2072484418" \
  --from 2000-09-01 \
  --to 2002-09-01 \
  --per-page 200 \
  --cursor "*"
```

`cited-by` 使用 `cites:work_id` + `from/to_publication_date` 过滤引用方的发表日期，`meta.count` 即为窗口内被引次数；可用 cursor 深翻页获取完整结果。

## 知识库检索
`knowledge` 子命令直接通过 HTTP 访问 Dify 知识库，无需再配置 `dify_knowledge_base_mcp`：

```bash
# 调整检索策略与语义/关键词权重
uv run tiangong-workspace knowledge retrieve "大数据治理" --top-k 8 --search-method semantic_search --semantic-weight 0.35

# 结合重排模型与元数据过滤
uv run tiangong-workspace knowledge retrieve \
  "新能源" \
  --reranking \
  --reranking-provider openai \
  --reranking-model text-embedding-3-large \
  --metadata '{"logical_operator": "and", "conditions": [{"name": "doc_type", "comparison_operator": "eq", "value": "白皮书"}]}'
```

命令会输出结构化 `WorkspaceResponse`，若搭配 `--json` 可方便地串接其他 Agent。除了沿用 `--options` 透传底层参数外，现在也可以通过 `--search-method`、`--reranking/--no-reranking`、`--reranking-provider/--reranking-model`、`--score-threshold`、`--score-threshold-enabled/--no-score-threshold-enabled`、`--semantic-weight` 与 `--metadata` 等旗标直接组装 Dify `retrieval_model` 以及 `metadata_filtering_conditions`，使检索精准可控。

## 向量嵌入
`embeddings` 子命令调用 `.sercrets/secrets.toml` 中配置的 OpenAI 兼容服务，供本地或私有环境快速生成向量：

```bash
# 基础示例
uv run tiangong-workspace embeddings generate "面向企业的数据治理方案"

# 批量文本 + 自定义模型 + JSON 输出
uv run tiangong-workspace embeddings generate "text A" "text B" \
  --model Qwen/Qwen3-Embedding-0.6B \
  --json
```

命令默认输出摘要信息，追加 `--json` 会返回包含 `embeddings`、`model`、`dimensions`、`usage` 的结构化 `WorkspaceResponse`，方便直接写入向量数据库或串接 Agent 工具。若连接到无鉴权的本地模型，可将 `api_key` 置为空字符串即可兼容。

## Secrets 配置
1. 复制 `.sercrets/secrets.example.toml` 为 `.sercrets/secrets.toml`（保持文件不入库）。
2. 填写 `openai.api_key`，可选配置 `model`、`chat_model`、`deep_research_model`。
3. 按示例补充 Tavily MCP 区块：

```toml
[tavily_web_mcp]
transport = "streamable_http"
service_name = "tavily"
url = "https://mcp.tavily.com/mcp"
api_key = "<YOUR_TAVILY_API_KEY>"
api_key_header = "Authorization"
api_key_prefix = "Bearer"
```

4. 按需在 `[neo4j]` 区块填入图数据库连接信息（未配置时智能体会自动跳过 Neo4j 工具）：

```toml
[neo4j]
uri = "bolt://localhost:7687"
username = "neo4j"
password = "<YOUR_NEO4J_PASSWORD>"
database = "neo4j"
```

5. 若需直接访问 Dify 知识库，请新增以下配置（示例值可替换为实际实例）：

```toml
[dify_knowledge_base]
api_base_url = "https://thuenv.tiangong.world:7001/v1"
api_key = "dataset-XXXX"
dataset_id = "53a90891-853c-4bf0-bf39-96dd84e11501"
```

配置完成后即可使用 `knowledge retrieve` 命令或 Dify LangChain Tool。

6. 若需启用 OpenAI 兼容的 embedding 生成（支持本地/私有部署服务），可按以下示例添加配置，其中 `api_key` 允许留空：

```toml
[openai_compatitble_embedding]
url = "http://192.168.1.140:8004/v1/"
api_key = ""
model = "Qwen/Qwen3-Embedding-0.6B"
```

## 自定义集成
1. 在 `pyproject.toml` 的 `[tool.tiangong.workspace.cli_tools]` 中新增/修改 CLI 监测项，即可立刻反映到 `tiangong-workspace tools`。
2. 通过 `[tool.tiangong.workspace.tool_registry]` 或 `tooling/registry.py` 注册新的内部工具，`tools --catalog` 会连同 JSON Schema 一起暴露给其他 Agent。
3. 通过 `agents/tools.py` / `tooling/executors.py` 构建新的 Tool 或执行器，`agents/deep_agent.py` 可将其纳入 LangGraph / DeepAgents 智能体。
4. 扩展 `agents/` 内的 LangGraph 工作流或新增模板，满足更多写作场景。
5. 同步更新 `AGENTS.md` 与 `README.md`，确保文档与代码一致。

## 开发提示
- 依赖管理：使用 `uv add <package>` / `uv remove <package>` 调整依赖。
- 每次修改程序后，务必按以下顺序运行并全部通过：

```bash
uv run black .
uv run ruff check
uv run pytest
```

- 测试运行：`uv run pytest`。
- 任何含代码的改动需同步维护本文件与 `AGENTS.md`。

## 许可证
本项目采用 [MIT License](LICENSE)。
