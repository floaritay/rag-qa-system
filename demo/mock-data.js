// ============================================================
// Mock Data for Demo
// ============================================================

const MOCK_DATA = {
    // 知识库列表
    knowledgeBases: [
        { id: 'default', name: '默认知识库', file_count: 5 },
        { id: 'tech', name: '技术文档', file_count: 3 },
    ],

    // 会话列表 (session_id / updated_at 格式匹配原始 app.js)
    sessions: {
        'default': [
            { session_id: 's1', title: 'RAG 技术原理', message_count: 4, updated_at: new Date().toISOString() },
            { session_id: 's2', title: '项目架构分析', message_count: 2, updated_at: new Date(Date.now() - 86400000).toISOString() },
        ],
        'tech': [
            { session_id: 's3', title: 'FastAPI 部署方案', message_count: 2, updated_at: new Date().toISOString() },
        ],
    },

    // 会话消息
    messages: {
        's1': [
            { role: 'user', content: '什么是 RAG 技术？它的核心原理是什么？' },
            {
                role: 'assistant',
                content: '**RAG（Retrieval-Augmented Generation）** 是一种将信息检索与大语言模型生成相结合的技术架构。\n\n## 核心原理\n\nRAG 的工作流程分为三个阶段：\n\n1. **检索阶段**：将用户问题转换为向量，在知识库中搜索最相关的文档片段\n2. **增强阶段**：将检索到的文档片段作为上下文，与原始问题一起组成 prompt\n3. **生成阶段**：大语言模型基于增强后的 prompt 生成回答\n\n## 主要优势\n\n- **减少幻觉**：回答基于真实文档，而非模型的参数记忆\n- **知识可更新**：只需更新文档库，无需重新训练模型\n- **可溯源**：每个回答都能追溯到具体的来源文档\n\n> RAG 技术是当前企业级 AI 应用的主流方案，特别适合需要基于私有知识库进行问答的场景。',
                sources: [
                    { source: 'docs/RAG技术概述.pdf', page: 2, rank: 1, vector_score: 0.92, rrf_score: 0.88, rerank_score: 0.95, content: 'RAG（Retrieval-Augmented Generation）是一种结合了信息检索和文本生成的技术框架。它通过在生成回答之前，先从外部知识库中检索相关文档片段，将这些片段作为上下文信息提供给大语言模型，从而生成更准确、更有依据的回答。' },
                    { source: 'docs/大模型应用实践.docx', page: 11, rank: 2, vector_score: 0.85, rrf_score: 0.82, rerank_score: 0.88, content: 'RAG 技术的核心优势在于它能够将大语言模型的生成能力与外部知识库的精确信息相结合，有效减少模型幻觉（hallucination）问题，同时保持知识的可更新性。' },
                    { source: 'docs/检索增强生成.pdf', page: 0, rank: 3, vector_score: 0.78, rrf_score: 0.75, rerank_score: 0.80, content: '检索增强生成的基本流程：用户提问 → 查询编码 → 向量检索 → 上下文增强 → LLM 生成 → 返回回答。' },
                ],
            },
            { role: 'user', content: '向量检索和关键词检索有什么区别？' },
            {
                role: 'assistant',
                content: '向量检索和关键词检索是两种不同的信息检索方式，各有优劣：\n\n## 向量检索（语义检索）\n\n- 将文本编码为高维向量，通过**余弦相似度**或**L2距离**计算语义相似性\n- 能理解同义词、近义表达（如"汽车"和"轿车"）\n- 适合模糊查询和语义理解\n- 本项目使用 FAISS 进行高效的向量检索\n\n## 关键词检索（BM25）\n\n- 基于**词频（TF）**和**逆文档频率（IDF）**的经典算法\n- 对精确匹配和专有名词效果更好\n- 不依赖模型推理，速度更快\n- 对中文需要特殊的分词处理\n\n## 混合检索策略\n\n本项目采用 **Reciprocal Rank Fusion（RRF）** 将两种检索结果融合：\n\n```\nRRF_score = Σ 1/(k + rank_i)\n```\n\n- 向量检索权重：0.7\n- BM25 检索权重：0.3\n- 融合后能兼顾语义理解和精确匹配',
                sources: [
                    { source: 'docs/检索算法详解.pdf', page: 4, rank: 1, vector_score: 0.90, rrf_score: 0.86, rerank_score: 0.92, content: '向量检索通过将文本映射到高维向量空间，利用余弦相似度或欧氏距离进行语义匹配。BM25 则基于词频和逆文档频率进行关键词匹配，对精确术语检索效果更好。' },
                    { source: 'docs/RAG技术概述.pdf', page: 6, rank: 2, vector_score: 0.82, rrf_score: 0.78, rerank_score: 0.85, content: '混合检索策略将向量检索和 BM25 检索的结果通过 RRF（Reciprocal Rank Fusion）算法融合，兼顾语义理解和精确匹配。' },
                ],
            },
        ],
        's2': [
            { role: 'user', content: '这个项目的整体架构是怎样的？' },
            {
                role: 'assistant',
                content: '## 项目架构概览\n\n本项目是一个基于 RAG 的个人知识库问答系统，采用前后端分离架构：\n\n### 后端（FastAPI）\n\n| 模块 | 技术栈 | 功能 |\n|------|--------|------|\n| Web 框架 | FastAPI | RESTful API + SSE 流式输出 |\n| 文档处理 | LangChain | PDF/PPTX/DOCX/MD 加载与分块 |\n| 向量存储 | FAISS | 高效向量检索 |\n| 关键词检索 | BM25 | 中英文混合关键词匹配 |\n| 重排序 | BGE-Reranker | 检索结果精排 |\n| 会话管理 | SQLite | 多轮对话历史 |\n\n### 前端（Vanilla JS）\n\n- 纯 HTML/CSS/JS，无框架依赖\n- 暗色主题，铜色/琥珀色调\n- 自定义 Markdown 渲染器\n- SSE 流式输出实时显示\n\n### RAG 流程\n\n```\n文档上传 → 文本分块(500字/块) → 向量化(BGE-M3) → FAISS 索引\n                                                    ↓\n用户提问 → 查询优化 → 混合检索 → 重排序 → LLM 生成回答\n```',
                sources: [
                    { source: 'docs/项目设计文档.md', page: 0, rank: 1, vector_score: 0.95, rrf_score: 0.91, rerank_score: 0.97, content: '本项目采用前后端分离架构。后端使用 FastAPI 框架，集成 LangChain 进行文档处理，FAISS 进行向量检索，SQLite 管理会话历史。' },
                    { source: 'docs/系统架构图.pdf', page: 1, rank: 2, vector_score: 0.88, rrf_score: 0.84, rerank_score: 0.90, content: '系统架构分为四层：接口层（FastAPI REST API）、检索层（FAISS + BM25 混合检索）、生成层（LLM 调用）、存储层（向量库 + SQLite）。' },
                ],
            },
        ],
        's3': [
            { role: 'user', content: 'FastAPI 如何实现流式输出？' },
            {
                role: 'assistant',
                content: '## FastAPI 流式输出实现\n\n本项目使用 **Server-Sent Events（SSE）** 实现流式输出，核心代码如下：\n\n### 后端实现\n\n```python\nfrom fastapi.responses import StreamingResponse\n\nasync def rag_stream(query: str):\n    async def generate():\n        # 1. 先发送检索到的来源文档\n        sources = retrieve(query)\n        yield f"data: {json.dumps({\'type\': \'sources\', \'data\': sources})}\\n\\n"\n        \n        # 2. 逐 token 流式输出 LLM 回答\n        async for chunk in llm.astream(prompt):\n            yield f"data: {json.dumps({\'type\': \'token\', \'data\': chunk})}\\n\\n"\n    \n    return StreamingResponse(generate(), media_type="text/event-stream")\n```\n\n### 前端消费\n\n```javascript\nconst res = await fetch(\'/ask/stream\', { method: \'POST\', body: JSON.stringify({question}) });\nconst reader = res.body.getReader();\n\nwhile (true) {\n    const {done, value} = await reader.read();\n    if (done) break;\n    const text = new TextDecoder().decode(value);\n    // 解析 SSE 事件并逐字显示\n}\n```\n\n### 优势\n\n- **低延迟**：用户无需等待完整回答生成\n- **实时反馈**：流式显示提供更好的交互体验\n- **来源优先**：先展示检索来源，再流式输出回答',
                sources: [
                    { source: 'docs/FastAPI实战指南.pdf', page: 7, rank: 1, vector_score: 0.91, rrf_score: 0.87, rerank_score: 0.93, content: 'FastAPI 通过 StreamingResponse 支持流式输出。将 media_type 设置为 text/event-stream 即可实现 SSE 协议，前端可通过 EventSource 或 fetch ReadableStream 接收数据。' },
                    { source: 'docs/项目设计文档.md', page: 14, rank: 2, vector_score: 0.84, rrf_score: 0.80, rerank_score: 0.86, content: '流式输出采用 SSE 协议，事件格式为 data: {JSON}\\n\\n。支持三种事件类型：sources（来源文档）、token（生成文本）、error（错误信息）。' },
                ],
            },
        ],
    },

    // 预设问答对（用于新对话和默认回答）
    presetQA: {
        '这门课程的主要内容是什么？': {
            answer: '## 课程主要内容\n\n本课程围绕 **RAG（检索增强生成）技术**展开，涵盖以下核心模块：\n\n### 第一部分：基础理论\n- 大语言模型（LLM）原理与局限性\n- 向量检索与语义搜索基础\n- RAG 架构设计思想\n\n### 第二部分：核心技术\n- 文档加载与智能分块策略\n- 向量化模型（Embedding）选型与使用\n- 混合检索：向量检索 + BM25 关键词检索\n- 检索结果重排序（Reranking）\n\n### 第三部分：工程实践\n- FastAPI 后端开发\n- FAISS 向量数据库实战\n- 流式输出与实时交互\n- 多轮对话与会话管理\n\n### 第四部分：优化与评估\n- 查询优化：查询改写与 HyDE\n- RAG 效果评估指标（Precision@K, Recall@K）\n- 生产环境部署与调优\n\n> 课程注重理论与实践结合，每个模块都配有完整的代码实现。',
            sources: [
                { source: 'docs/课程大纲.pdf', page: 0, rank: 1, vector_score: 0.96, rrf_score: 0.93, rerank_score: 0.98, content: '本课程共分为四个部分：基础理论、核心技术、工程实践、优化与评估。涵盖从 RAG 概念到生产部署的完整知识体系。' },
                { source: 'docs/教学计划.docx', page: 1, rank: 2, vector_score: 0.88, rrf_score: 0.84, rerank_score: 0.90, content: '教学计划安排：第 1-3 周基础理论，第 4-7 周核心技术，第 8-10 周工程实践，第 11-12 周优化与评估。' },
                { source: 'docs/RAG技术概述.pdf', page: 0, rank: 3, vector_score: 0.82, rrf_score: 0.78, rerank_score: 0.85, content: 'RAG 技术概述：检索增强生成是当前最主流的大模型应用架构，通过将外部知识库与 LLM 结合，实现基于私有数据的精准问答。' },
            ],
        },
        '请总结一下最近讲的知识点': {
            answer: '## 近期知识点总结\n\n### 1. 混合检索策略\n- 向量检索（FAISS）与关键词检索（BM25）的融合\n- 使用 **Reciprocal Rank Fusion（RRF）** 算法合并排序\n- 权重配置：向量 0.7 + BM25 0.3\n\n### 2. 查询优化技术\n- **查询改写**：将口语化问题转换为规范检索词\n- **HyDE**：生成假设性回答，用其向量进行检索\n- 注意：HyDE 与混合检索互斥\n\n### 3. 检索结果重排序\n- 使用 BGE-Reranker 交叉编码器\n- 对候选文档进行精排，保留 Top-3\n- 显著提升答案准确率\n\n### 4. 流式输出实现\n- 基于 SSE（Server-Sent Events）协议\n- 先发送来源文档，再逐 token 输出回答\n- 前端使用 ReadableStream 实时渲染\n\n### 5. 会话管理\n- SQLite 存储对话历史\n- 每 10 条消息自动摘要\n- 7 天自动过期机制',
            sources: [
                { source: 'docs/课程笔记_第8周.md', page: 0, rank: 1, vector_score: 0.94, rrf_score: 0.90, rerank_score: 0.96, content: '本周重点：混合检索策略的实现与优化，包括 RRF 融合算法、权重调优、以及与纯向量检索的效果对比。' },
                { source: 'docs/检索算法详解.pdf', page: 9, rank: 2, vector_score: 0.86, rrf_score: 0.82, rerank_score: 0.88, content: 'BGE-Reranker 使用交叉编码器对 query-document 对进行精排，相比向量检索的双塔模型，精度更高但速度较慢。' },
                { source: 'docs/RAG技术概述.pdf', page: 4, rank: 3, vector_score: 0.79, rrf_score: 0.75, rerank_score: 0.81, content: 'HyDE（Hypothetical Document Embeddings）技术：先让 LLM 生成一个假设性回答，再用该回答的向量进行检索，可以提升语义匹配精度。' },
            ],
        },
        '有哪些重要的概念需要掌握？': {
            answer: '## 重要概念清单\n\n### 核心概念\n\n| 概念 | 说明 | 重要度 |\n|------|------|--------|\n| **RAG** | 检索增强生成，结合检索与生成的 AI 架构 | ⭐⭐⭐ |\n| **Embedding** | 将文本转换为高维向量的技术 | ⭐⭐⭐ |\n| **FAISS** | Facebook 开源的高效向量检索库 | ⭐⭐⭐ |\n| **BM25** | 经典的关键词检索算法 | ⭐⭐ |\n| **Reranking** | 对检索结果进行精排的技术 | ⭐⭐⭐ |\n\n### 进阶概念\n\n| 概念 | 说明 | 重要度 |\n|------|------|--------|\n| **RRF** | Reciprocal Rank Fusion，多路检索融合算法 | ⭐⭐ |\n| **HyDE** | Hypothetical Documents Embedding，假设文档嵌入 | ⭐⭐ |\n| **查询改写** | 将口语化问题转换为规范检索词 | ⭐⭐ |\n| **Chunking** | 文档分块策略，影响检索质量 | ⭐⭐⭐ |\n| **SSE** | Server-Sent Events，流式输出协议 | ⭐ |\n\n### 必须理解的公式\n\n```\n# RRF 融合公式\nRRF_score(d) = Σ 1/(k + rank_i(d))    # k 通常取 60\n\n# BM25 评分公式  \nBM25(q,d) = Σ IDF(qi) · (f(qi,d) · (k1+1)) / (f(qi,d) + k1·(1-b+b·|d|/avgdl))\n```',
            sources: [
                { source: 'docs/核心概念手册.pdf', page: 0, rank: 1, vector_score: 0.97, rrf_score: 0.94, rerank_score: 0.99, content: 'RAG 核心概念：Retrieval-Augmented Generation，通过检索外部知识来增强大语言模型的生成能力，减少幻觉，提高准确性。' },
                { source: 'docs/检索算法详解.pdf', page: 0, rank: 2, vector_score: 0.89, rrf_score: 0.85, rerank_score: 0.91, content: 'BM25 算法是信息检索领域的经典算法，基于概率检索模型，通过词频（TF）和逆文档频率（IDF）计算文档相关性。' },
                { source: 'docs/RAG技术概述.pdf', page: 1, rank: 3, vector_score: 0.83, rrf_score: 0.79, rerank_score: 0.86, content: 'Embedding 模型将文本映射到稠密向量空间，语义相近的文本在向量空间中距离更近。常用模型包括 BGE、OpenAI text-embedding 等。' },
            ],
        },
    },

    // 通用默认回答（当用户输入不在预设中时使用）
    defaultAnswer: {
        answer: '这是一个很好的问题。根据知识库中的文档，我来为您解答。\n\n基于 RAG 技术的检索结果，相关内容主要涉及以下几个方面：\n\n1. **技术原理**：RAG 通过检索外部知识库来增强大语言模型的回答能力\n2. **实现方式**：使用向量检索（FAISS）+ 关键词检索（BM25）的混合策略\n3. **优化手段**：包括查询改写、HyDE、重排序等预处理和后处理技术\n\n如果您想了解更多细节，可以尝试：\n- 查看侧边栏中的历史对话\n- 使用更具体的问题进行提问\n- 切换到不同的知识库查看相关文档',
        sources: [
            { source: 'docs/RAG技术概述.pdf', page: 0, rank: 1, vector_score: 0.75, rrf_score: 0.70, rerank_score: 0.78, content: 'RAG 技术通过检索-增强-生成的流程，将外部知识与大语言模型结合，实现基于私有数据的精准问答。' },
            { source: 'docs/项目设计文档.md', page: 0, rank: 2, vector_score: 0.68, rrf_score: 0.65, rerank_score: 0.72, content: '本项目实现了完整的 RAG 流程，包括文档加载、分块、向量化、混合检索、重排序和流式生成。' },
        ],
    },

    // 知识库文件列表 (name/size/modified 格式匹配原始 app.js 的 renderFileList)
    files: {
        'default': [
            { name: 'RAG技术概述.pdf', size: 2456789, modified: 1747733400 },
            { name: '检索算法详解.pdf', size: 1823456, modified: 1747563600 },
            { name: '大模型应用实践.docx', size: 987654, modified: 1747277700 },
            { name: '项目设计文档.md', size: 45678, modified: 1747919100 },
            { name: '课程笔记_第8周.md', size: 23456, modified: 1748149200 },
        ],
        'tech': [
            { name: 'FastAPI实战指南.pdf', size: 3456789, modified: 1747637400 },
            { name: '系统架构图.pdf', size: 1234567, modified: 1747818000 },
            { name: '部署文档.md', size: 34567, modified: 1747981800 },
        ],
    },

    // 模型配置
    config: {
        llm: { model: 'Qwen/Qwen3-8B', base_url: 'https://api.siliconflow.cn/v1', api_key: '' },
        embedding: { model: 'BAAI/bge-m3', base_url: 'https://api.siliconflow.cn/v1', api_key: '' },
        reranker: { model: 'BAAI/bge-reranker-v2-m3', base_url: 'https://api.siliconflow.cn/v1', api_key: '' },
    },

    // 硅基流动模型列表
    siliconflowModels: {
        llm: [
            { id: 'Qwen/Qwen3-8B', name: 'Qwen3-8B', is_free: true },
            { id: 'Qwen/Qwen3-14B', name: 'Qwen3-14B', is_free: false },
            { id: 'Qwen/Qwen3-32B', name: 'Qwen3-32B', is_free: false },
            { id: 'deepseek-ai/DeepSeek-V3', name: 'DeepSeek-V3', is_free: false },
            { id: 'THUDM/GLM-Z1-9B-0414', name: 'GLM-Z1-9B', is_free: true },
        ],
        embedding: [
            { id: 'BAAI/bge-m3', name: 'BGE-M3', is_free: true },
            { id: 'BAAI/bge-large-zh-v1.5', name: 'BGE-Large-ZH', is_free: true },
        ],
        reranker: [
            { id: 'BAAI/bge-reranker-v2-m3', name: 'BGE-Reranker-v2-M3', is_free: true },
        ],
    },

    // 会话 ID 计数器
    _sessionIdCounter: 10,
};
