# DocMind：可溯源多文档 RAG 问答系统

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-green)](https://flask.palletsprojects.com/)
[![LangChain](https://img.shields.io/badge/LangChain-Community-orange)](https://python.langchain.com/)
[![Chroma](https://img.shields.io/badge/Chroma-Vector%20DB-red)](https://www.trychroma.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

基于 RAG（检索增强生成）架构的本地文档智能问答系统。支持 PDF、Word、Markdown、TXT 多格式文档上传解析，采用**混合检索**（向量稠密检索 + BM25 词法检索 + RRF 融合排序）提升召回准确率，内置多轮对话查询改写、增量索引、来源引用追溯与 Prompt 注入防御机制，并提供可复现的 Recall@K / MRR 评估框架。

## 功能特性

### 文档处理与存储
- **多格式解析**：支持 PDF（PyPDF）、DOCX（docx2txt）、Markdown、TXT 自动编码检测
- **智能切分**：500 字符/块 + 80 字符重叠，适配中文标点（句号、感叹号、问号）递归分隔
- **增量索引**：新增文档无需重建知识库，支持按文档 ID 删除及其关联向量
- **持久化存储**：Chroma 本地向量数据库持久化，重启后知识库不丢失
- **元数据管理**：每个块记录文件名、页码、块索引、文档 ID，支持来源精确定位

### 混合检索引擎
- **稠密向量检索**：基于智谱 embedding-2 模型生成向量，Chroma 相似度检索
- **BM25 词法检索**：自研轻量 BM25 实现（无额外依赖），支持中英文混合分词（英文单词 + 中文 unigram/bigram）
- **RRF 融合排序**：加权 Reciprocal Rank Fusion（稠密 0.65 / 词法 0.35）融合两路检索结果
- **多样性排序**：相邻块去重机制，避免同一文档相邻片段垄断上下文窗口
- **检索追踪**：返回检索耗时（ms）、生成耗时、命中通道（dense / lexical / both）等性能指标

### 多轮对话与生成
- **查询改写**：利用 GLM-4-Flash 将多轮对话中的追问改写为独立检索 query，提升上下文关联问题的召回率
- **对话历史管理**：保留最近 6 轮对话，自动截断超长内容
- **引用标注**：模型回答中标注 [1] [2] 等引用编号，对应具体来源片段
- **Prompt 注入防御**：系统提示约束模型忽略文档中嵌入的恶意指令
- **无依据拒答**：检索结果低于相关度阈值时，明确返回"未找到相关信息"而非编造答案

### 工程化与评估
- **评估框架**：`evaluate.py` 基于 JSONL 标注数据集计算 Recall@K 与 MRR 指标
- **单元测试**：pytest 覆盖 API 接口、嵌入服务、检索逻辑三大模块
- **代码规范**：ruff 静态检查，保持代码风格统一
- **配置管理**：.env 环境变量配置（API Key、端口、上传限制、Top-K、历史轮数等）
- **安全防护**：文件上传大小限制（默认 20MB）、文件名安全处理、异常统一处理

## 技术架构

```
用户上传文档
    │
    ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  文档解析    │────▶│  智能切分     │────▶│  向量嵌入     │
│ (多格式)     │     │ (500+80)     │     │ (embedding-2)│
└─────────────┘     └──────────────┘     └──────┬───────┘
                                                 │
                                                 ▼
                                          ┌──────────────┐
                                          │  Chroma 存储  │
                                          │  (持久化)     │
                                          └──────┬───────┘
                                                 │
用户提问 ◀─── 引用来源 ◀─── 生成回答 ◀─── GLM-4-Flash ◀─── 混合检索 ◀───┘
    │                                              │
    └──────── 查询改写（多轮对话）◀─────────────────┘
              (稠密检索 + BM25 + RRF 融合)
```

## 项目结构

```
ai-doc-qa/
├── app.py                  # Flask 应用入口，路由与请求处理
├── rag_service.py          # RAG 核心服务：文档处理、检索、问答、查询改写
├── retrieval.py            # 混合检索：BM25 索引 + RRF 融合排序 + 多样性重排
├── evaluate.py             # 评估脚本：Recall@K / MRR 计算
├── requirements.txt        # 运行依赖
├── requirements-dev.txt    # 开发依赖（含 pytest、ruff）
├── pytest.ini              # pytest 配置
├── .env.example            # 环境变量示例
├── templates/
│   └── index.html          # Web 交互界面
├── tests/
│   ├── test_app.py         # API 接口测试
│   ├── test_embeddings.py  # 嵌入服务测试
│   └── test_retrieval.py   # 检索逻辑测试
└── data/
    ├── uploads/            # 上传文档存储
    └── chroma/             # Chroma 向量数据库
```

## 快速开始

### 环境要求
- Python 3.10+
- 智谱 AI API Key（[申请地址](https://open.bigmodel.cn/)）

### 安装与运行

```bash
# 1. 克隆仓库
git clone https://github.com/zywooo666/ai-doc-qa.git
cd ai-doc-qa

# 2. 创建虚拟环境
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements-dev.txt

# 4. 配置环境变量
copy .env.example .env
# 编辑 .env，填入你的 ZHIPU_API_KEY

# 5. 启动服务
python app.py
```

启动后打开浏览器访问 **http://127.0.0.1:5000**，上传文档即可开始问答。

### 环境变量说明

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `ZHIPU_API_KEY` | - | 智谱 AI API Key（必填） |
| `PORT` | 5000 | 服务端口 |
| `MAX_UPLOAD_MB` | 20 | 单文件上传大小限制（MB） |
| `TOP_K` | 4 | 检索返回相关片段数量 |
| `MAX_HISTORY_TURNS` | 6 | 多轮对话保留轮数 |
| `LOG_LEVEL` | INFO | 日志级别 |

## API 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/health` | 健康检查与知识库就绪状态 |
| GET | `/api/documents` | 获取已索引文档列表 |
| POST | `/api/documents` | 多文件增量索引上传（`files`） |
| DELETE | `/api/documents/<id>` | 删除指定文档及其向量 |
| POST | `/api/chat` | 问答接口，参数 `question` + `history` |

### 问答示例

```bash
curl -X POST http://127.0.0.1:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "项目支持哪些文档格式？", "history": []}'
```

返回内容包含：回答文本、来源引用列表（文件名、页码、块索引、相似度分数、检索通道）、检索与生成耗时。

## 评估

创建 UTF-8 编码的 JSONL 评估数据集，每行包含 `query` 和 `relevant_chunk_ids`（格式 `document_id:chunk_index`）：

```jsonl
{"query": "支持哪些文档格式？", "relevant_chunk_ids": ["abc123:0", "abc123:1"]}
{"query": "如何配置API Key？", "relevant_chunk_ids": ["def456:3"]}
```

运行评估：

```bash
python evaluate.py eval.jsonl
```

输出指标：查询数量、Recall@K、MRR（平均倒数排名）。

> 注意：请勿在无固定评估数据集的情况下宣称准确率百分比。

## 测试

```bash
# 运行单元测试
pytest

# 代码规范检查
ruff check .
```

## 技术栈

| 类别 | 技术 |
| --- | --- |
| 后端框架 | Flask 3.0 |
| RAG 框架 | LangChain Community |
| 向量数据库 | Chroma |
| 大模型 | 智谱 AI GLM-4-Flash |
| 嵌入模型 | 智谱 AI embedding-2 |
| 文档解析 | PyPDF、docx2txt |
| 测试 | pytest、ruff |
| 部署 | 本地运行（Werkzeug 开发服务器） |

## License

MIT License - 详见 [LICENSE](LICENSE) 文件。
