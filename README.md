# RAG 智能文档助手

基于大模型的企业文档问答系统。上传 Markdown 格式的内部说明书，系统自动构建向量索引，支持实时检索增强生成（RAG）。

## 功能

- **智能问答** — 基于 RAG 检索公司内部文档，结合大模型生成准确回答
- **文档管理** — 支持拖拽上传 Markdown 文件，自动分片并建立向量索引
- **流式输出** — 回答实时流式展示，支持 Markdown 渲染和代码高亮
- **模型配置** — 可通过 UI 实时切换 LLM 模型、Embedding 模型、API 地址等
<img width="1278" height="696" alt="image" src="https://github.com/user-attachments/assets/39e853e8-874c-4df5-afa4-2ade0e1166e6" />
<img width="888" height="572" alt="image" src="https://github.com/user-attachments/assets/98ad3d4e-295e-4175-b1da-cc742eb3f82e" />

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp env.example .env
# 编辑 .env，填入你的 DashScope API Key

# 3. 启动服务
python main.py
```

浏览器访问 `http://localhost:8000`

## 配置说明

### 环境变量 (.env)

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_API_KEY` | 大模型 API Key | - |
| `LLM_BASE_URL` | API 地址（OpenAI 兼容） | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `LLM_MODEL` | LLM 模型名 | `qwen-plus` |
| `EMBEDDING_MODEL` | Embedding 模型名 | `text-embedding-v3` |
| `TEMPERATURE` | 生成温度 | `0.7` |
| `CHUNK_SIZE` | 文档分片大小（字符） | `500` |
| `CHUNK_OVERLAP` | 分片重叠 | `50` |
| `TOP_K` | 检索返回片段数 | `5` |

### 运行时配置

启动后点击页面右上角 **⚙️ 模型配置** 按钮，可在线修改所有配置参数，无需重启服务。

## 技术栈

- **后端**: FastAPI + uvicorn
- **向量数据库**: ChromaDB（本地持久化）
- **LLM/Embedding**: 通过 OpenAI 兼容接口调用 Qwen（DashScope），也支持任何兼容端点
- **前端**: 原生 HTML/CSS/JS 单页应用
