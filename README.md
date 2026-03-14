# 智能课程助手

基于 RAG（检索增强生成）技术的智能问答系统，帮助学生快速查询课程资料中的内容。

## 项目简介

智能课程助手是一个 AI 驱动的问答系统，能够从课程 PDF 资料中检索相关信息并生成准确回答。系统采用前后端分离架构，后端使用 FastAPI + LangChain 构建 RAG 服务，前端使用 Gradio 提供友好的 Web 界面。

## 技术栈

| 组件     | 技术选型                    |
| ------ | ----------------------- |
| 后端框架   | FastAPI                 |
| RAG 框架 | LangChain               |
| 向量数据库  | FAISS（本地存储）             |
| 大语言模型  | 阿里云百炼 qwen-plus         |
| 嵌入模型   | 阿里云百炼 text-embedding-v2 |
| 前端界面   | Gradio                  |

## 项目结构

```
CV1/
├── backend/                    # 后端服务
│   ├── main.py                # FastAPI 主服务
│   ├── knowledge_base.py      # 知识库构建模块
│   ├── qa_chain.py            # 问答链模块
│   └── requirements.txt       # Python 依赖
├── frontend/                   # 前端服务
│   ├── app.py                 # Gradio Web 界面
│   └── requirements.txt       # Python 依赖
├── course_materials/           # 课程资料存放目录
│   └── *.pdf                  # PDF 课程文件
├── course_knowledge_base/      # 向量库存储目录（自动生成）
├── DEPLOYMENT_GUIDE.md         # 详细部署指南
└── README.md                   # 项目说明文档
```

## 快速开始

### 环境要求

- Python 3.10+
- 阿里云百炼平台 API 密钥

### 安装步骤

1. **克隆项目**

```bash
git clone <repository-url>
cd CV1
```

1. **安装后端依赖**

```bash
cd backend
pip install -r requirements.txt
```

1. **安装前端依赖**

```bash
cd ../frontend
pip install -r requirements.txt
```

1. **配置环境变量**

```bash
# Windows
set BAILIAN_API_KEY=你的百炼平台API密钥

# Linux/Mac
export BAILIAN_API_KEY=你的百炼平台API密钥
```

1. **准备课程资料**

将 PDF 格式的课程资料放入 `course_materials` 文件夹。

1. **启动后端服务**

```bash
cd backend
python main.py
```

后端服务将在 <http://localhost:8001> 运行。

1. **启动前端界面**

```bash
cd frontend
python app.py
```

前端界面将在 <http://localhost:7860> 运行。

## API 接口

| 接口        | 方法   | 说明        |
| --------- | ---- | --------- |
| `/`       | GET  | 服务信息      |
| `/health` | GET  | 健康检查      |
| `/ask`    | POST | 问答接口      |
| `/init`   | POST | 初始化/重建知识库 |

### 问答示例

```bash
curl -X POST http://localhost:8001/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "机器学习课程的成绩构成是怎样的？"}'
```

### 初始化知识库

```bash
# 初始化知识库
curl -X POST http://localhost:8001/init

# 强制重建知识库
curl -X POST "http://localhost:8001/init?force_rebuild=true"
```

## 功能特性

- **智能问答**：基于课程资料进行精准问答，回答有据可依
- **知识库管理**：支持知识库初始化和重建
- **批量处理**：支持批量加载 PDF 文件构建向量库
- **美观界面**：现代化的 Gradio Web 界面
- **服务监控**：提供健康检查接口

## 工作原理

```
用户提问 → 文本嵌入 → 向量检索 → 上下文组装 → LLM生成 → 返回答案
              ↓
         FAISS向量库
              ↑
    PDF文档 → 文本分割 → 向量化 → 存储
```

1. **文档处理**：使用 PyPDFLoader 加载 PDF，RecursiveCharacterTextSplitter 分割文本
2. **向量化**：调用百炼平台嵌入模型将文本转换为向量
3. **存储**：使用 FAISS 本地向量数据库存储
4. **检索**：基于语义相似度检索相关文档片段
5. **生成**：LLM 基于检索内容生成回答

## 详细文档

详细的部署和使用说明请参阅 [DEPLOYMENT\_GUIDE.md](./DEPLOYMENT_GUIDE.md)。

## 注意事项

- 确保 `BAILIAN_API_KEY` 环境变量已正确设置
- 课程资料变更后需调用 `/init` 接口重建知识库
- 向量库文件存储在 `course_knowledge_base` 目录

## License

MIT
