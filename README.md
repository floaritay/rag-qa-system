# 智能课程助手

基于 RAG（检索增强生成）技术的智能问答系统，帮助学生快速查询课程资料中的内容。

## 项目简介

智能课程助手是一个 AI 驱动的问答系统，能够从课程资料（PDF/PPTX/DOCX/MD）中检索相关信息并生成准确回答。系统采用 FastAPI + LangChain 构建 RAG 服务，支持混合检索、查询优化、重排序等高级功能，同时提供 OpenAI 兼容 API 以接入 Open WebUI 等客户端。

## 技术栈

| 组件     | 技术选型                     |
| ------ | ------------------------ |
| 后端框架   | FastAPI                  |
| RAG 框架 | LangChain                |
| 向量数据库  | FAISS（本地存储）              |
| 关键词检索  | BM25（rank_bm25）          |
| 大语言模型  | 阿里云百炼 qwen3.5-122b-a10b |
| 嵌入模型   | 硅基流动 BAAI/bge-m3        |
| 重排序模型  | 硅基流动 BAAI/bge-reranker-v2-m3 |
| 前端界面   | 原生 Web 界面（HTML+CSS+JS） |

## 项目结构

```
├── backend/                            # 后端服务
│   ├── main_siliconflow_rag.py         # 主服务（完整版，含混合检索、重排序、会话记忆）
│   ├── main.py                         # 基础版本（百炼嵌入）
│   ├── main_siliconflow.py             # 硅基流动嵌入版本
│   ├── main_siliconflow_memory.py      # 带会话记忆版本
│   ├── evaluate.py                     # RAG 评估脚本
│   └── requirements.txt                # Python 依赖
├── web/                                # 前端界面
│   ├── index.html                      # 页面结构
│   ├── app.js                          # 交互逻辑
│   └── styles.css                      # 样式
├── docs/                               # 文档目录
├── course_materials/                   # 课程资料存放目录（不存在请手动创建）
├── course_knowledge_base/              # 向量库 + BM25 索引（自动生成）
├── start.bat                           # 一键启动脚本
└── README.md
```

## 快速开始

### 环境要求

- Python 3.10+
- 阿里云百炼平台 API 密钥（用于大语言模型）
- 硅基流动 API 密钥（用于大语言模型，嵌入模型和重排序模型）

### 安装步骤

1. **克隆项目**

```bash
git clone <repository-url>
cd 你的文件目录
```

2. **安装依赖**

```bash
pip install -r backend/requirements.txt
```

3. **配置环境变量**

在项目根目录创建 `.env` 文件：

```
BAILIAN_API_KEY=你的百炼平台API密钥
SILICONFLOW_API_KEY=你的硅基流动API密钥
```
或在前端页面设置

4. **准备课程资料**

将课程资料（PDF/PPTX/DOCX/MD）放入 `course_materials` 文件夹（如不存在请手动创建）。

5. **启动服务**

```bash
# Windows - 一键启动（后端 + 前端 + 自动打开浏览器）
start.bat

# 或手动启动
python backend/main_siliconflow_rag.py          # 后端 http://localhost:8001
python -m http.server 8080 --directory web      # 前端 http://localhost:8080
```

6. **初始化知识库**

```bash
# 初始化知识库
curl -X POST http://localhost:8001/init

# 强制重建知识库（课程资料变更后使用）
curl -X POST "http://localhost:8001/init?force_rebuild=true"
```

或在前端页面点击初始化按钮。

## 功能特性

- **混合检索**：向量语义检索（FAISS）+ 关键词检索（BM25），通过 RRF 融合排序
- **查询优化**：查询改写（将口语化问题转为标准检索词）和 HyDE（生成假设答案后检索）
- **重排序**：使用硅基流动 reranker 模型对检索结果二次排序，提升相关性
- **会话记忆**：基于 SQLite 的多轮对话管理，自动摘要历史消息
- **多格式支持**：支持 PDF、PPTX、DOCX、Markdown 格式的课程资料
- **知识库管理**：支持文件上传、删除、知识库初始化和重建
- **OpenAI 兼容 API**：支持 Open WebUI 等客户端接入
- **RAG 评估**：提供评估脚本，计算精确率、召回率等指标

## 工作原理

```
用户提问
  ├── 查询改写（可选）：口语化 → 标准检索词
  ├── HyDE（可选）：生成假设答案，用其向量检索
  │
  ▼
混合检索（可选）
  ├── FAISS 向量检索（权重 0.7）
  └── BM25 关键词检索（权重 0.3）
  │
  ▼ RRF 融合
重排序（可选）：reranker 模型二次排序
  │
  ▼
上下文组装 → LLM 生成 → 返回答案（含来源引用）
```

**文档入库流程**：课程资料 → 文本分割（chunk_size=500, overlap=50）→ 嵌入向量化 → 存入 FAISS + BM25 索引

## API 接口

### 问答接口

| 接口                     | 方法   | 说明                        |
| ---------------------- | ---- | ------------------------- |
| `/ask`                 | POST | 问答接口（支持检索策略参数）         |
| `/health`              | GET  | 健康检查                     |
| `/init`                | POST | 初始化/重建知识库（`?force_rebuild=true` 强制重建） |

### 会话管理

| 接口                     | 方法   | 说明              |
| ---------------------- | ---- | --------------- |
| `/sessions`            | POST | 创建新会话          |
| `/sessions`            | GET  | 获取所有会话列表      |
| `/sessions/{session_id}` | GET  | 获取指定会话详情     |
| `/sessions/{session_id}` | DELETE | 删除指定会话       |
| `/sessions/{session_id}/messages` | GET | 获取会话消息历史 |

### 知识库管理

| 接口                     | 方法   | 说明              |
| ---------------------- | ---- | --------------- |
| `/materials`           | GET  | 列出所有课程资料文件   |
| `/materials/upload`    | POST | 上传课程资料文件      |
| `/materials/{filename}` | DELETE | 删除指定课程资料文件 |

### 配置与兼容

| 接口                     | 方法   | 说明                        |
| ---------------------- | ---- | ------------------------- |
| `/config`              | GET  | 获取当前配置                   |
| `/config`              | POST | 更新配置                     |
| `/v1/models`           | GET  | 返回可用模型列表（OpenAI 兼容）     |
| `/v1/chat/completions` | POST | 处理 OpenAI 格式的聊天请求（OpenAI 兼容） |

### 问答示例

```bash
curl -X POST http://localhost:8001/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是多AGV调度算法？"}'
```

## 评估工具

```bash
python backend/evaluate.py
```

输出指标：Precision@1/3/5、Recall@1/3/5、平均响应时间。

## 常见问题

**需要重建知识库的情况：**
- 新增/删除/修改了课程资料文件
- 更换了嵌入模型
- 修改了文本分割参数（chunk_size, chunk_overlap）

**后端版本说明：**
- `main_siliconflow_rag.py` — 完整版，推荐使用
- `main_siliconflow_memory.py` — 仅会话记忆，无 RAG 优化
- `main_siliconflow.py` — 基础 RAG，硅基流动嵌入
- `main.py` — 基础 RAG，百炼嵌入

## 详细文档

- [部署指南](./docs/DEPLOYMENT_GUIDE.md)
- [评估指南](./docs/EVALUATION_GUIDE.md)
- [RAG 优化指南](./docs/RAG_OPTIMIZATION_GUIDE.md)
- [会话记忆指南](./docs/SESSION_MEMORY_GUIDE.md)
- [来源显示指南](./docs/SOURCES_DISPLAY_GUIDE.md)
- [问题解决记录](./docs/solve.txt)

## License

MIT
