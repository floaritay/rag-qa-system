# 会话管理与短期记忆 — 实现原理与流程

## 概述

`backend/main_siliconflow_memory.py` 在原有无状态 RAG 问答基础上，新增了**会话管理**和**短期记忆**功能。核心目标是让 LLM 能理解对话上下文，支持追问（如"请详细解释"、"那它有什么优缺点？"）。

前端已适配，用户**无需手动创建会话**——首次提问时自动创建，后续追问自动复用同一会话。

## 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                      前端 (web/)                             │
│                                                              │
│  ┌─────────────┐  首次提问自动创建会话  ┌──────────────────┐ │
│  │ 会话侧边栏   │ ←───────────────────→ │    聊天区域       │ │
│  │ 列表/切换/删除│   session_id 传递     │ 问答+历史展示     │ │
│  └─────────────┘                       └──────────────────┘ │
└──────────────────────────┬──────────────────────────────────┘
                           │ POST /ask { question, session_id? }
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI 服务端                             │
│                                                              │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │ 会话管理  │    │  SQLite DB    │    │  RAG 查询引擎     │  │
│  │ /sessions │    │ sessions表    │    │ retriever + LLM   │  │
│  │ CRUD 端点 │    │ messages表    │    │ 带历史注入         │  │
│  └──────────┘    └──────────────┘    └───────────────────┘  │
│       │               │                      │               │
│       ▼               ▼                      ▼               │
│  创建/查询/       读写对话历史          检索文档+生成回答      │
│  删除会话         注入到 prompt         结合上下文理解代词     │
└─────────────────────────────────────────────────────────────┘
```

## 一、会话管理

### 1.1 什么是会话

一个**会话（Session）**代表一次完整的对话，由 `session_id`（UUID）唯一标识。同一个会话内的所有问答共享上下文，LLM 可以引用之前的对话内容。

### 1.2 会话生命周期

```
用户首次提问 ──→ 前端自动创建会话 ──→ 后端存储问答 ──→ 后续追问复用会话
      │               │                  │                │
  不需要session_id   POST /sessions     POST /ask        session_id自动传递
  前端自动处理       返回session_id     带session_id     历史自动注入prompt
                                                    │
                                                    ▼
                                              自动摘要（每10条消息）
                                              过期清理（7天）
```

**关键设计：用户无需关心会话的存在。** 前端在首次提问时自动调用 `/sessions` 创建会话，后续所有请求自动携带 `session_id`。

### 1.3 数据库存储

使用 SQLite（Python 内置，无额外依赖），数据库文件位于 `backend/sessions.db`。

**sessions 表：**
| 字段 | 类型 | 说明 |
|------|------|------|
| session_id | TEXT (PK) | UUID 唯一标识 |
| title | TEXT | 会话标题，首次问答后自动取用户消息前30字 |
| created_at | TEXT | 创建时间（ISO 8601） |
| updated_at | TEXT | 最后更新时间 |
| summary | TEXT | LLM 生成的对话摘要（可为空） |

**messages 表：**
| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER (PK) | 自增主键 |
| session_id | TEXT (FK) | 关联会话，级联删除 |
| role | TEXT | "user" 或 "assistant" |
| content | TEXT | 消息内容 |
| created_at | TEXT | 创建时间 |

### 1.4 API 端点

| 方法 | 路径 | 功能 |
|------|------|------|
| POST | `/sessions` | 创建新会话，返回 session_id |
| GET | `/sessions` | 列出所有会话（按更新时间倒序） |
| GET | `/sessions/{id}` | 获取单个会话详情（含摘要和消息数） |
| DELETE | `/sessions/{id}` | 删除会话及其所有消息 |
| GET | `/sessions/{id}/messages` | 获取会话的完整消息历史 |
| POST | `/sessions/cleanup` | 清理超过指定天数的过期会话 |

## 二、短期记忆

### 2.1 记忆原理

传统 RAG 每次问答都是独立的，LLM 看不到之前的对话。短期记忆的实现方式是：**将历史对话格式化后注入到 prompt 中**，让 LLM 在回答时能参考上下文。

```
无记忆 prompt：
┌─────────────────────────┐
│ 参考资料：{context}       │
│                          │
│ 学生问题：{question}      │
└─────────────────────────┘

有记忆 prompt：
┌─────────────────────────┐
│ 参考资料：{context}       │
│                          │
│ 对话历史：               │
│ 用户：什么是多AGV调度？   │
│ 助手：多AGV调度是指...    │
│                          │
│ 学生问题：那它有什么优缺点？│
└─────────────────────────┘
```

LLM 看到对话历史后，能理解"它"指代"多AGV调度"，从而给出有意义的回答。

### 2.2 历史注入流程

```
用户提问 "那它有什么优缺点？"
         │
         ▼
┌─────────────────────────┐
│ 1. 从 SQLite 加载该会话   │
│    最近 10 轮对话历史     │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ 2. 格式化为文本：         │
│    "### 对话历史：       │
│     用户：什么是多AGV？  │
│     助手：多AGV调度是..." │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ 3. 检索相关文档           │
│    retriever.invoke()    │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ 4. 组装完整 prompt        │
│    {context} + {history} │
│    + {question}          │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ 5. 调用 LLM 生成回答     │
│    LLM 理解"它"指代      │
│    "多AGV调度"           │
└─────────────────────────┘
```

### 2.3 历史截断策略

- 最多保留最近 **10 轮对话**（20 条消息：10 条 user + 10 条 assistant）
- 超出部分自动丢弃，避免超出 LLM token 限制
- 截断时保留最新的对话，丢弃最旧的

## 三、RAG 查询重构

### 3.1 原版（无状态）

原版使用 LangChain LCEL 链式调用：

```python
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | PROMPT
    | llm
    | StrOutputParser()
)
answer = rag_chain.invoke(question)
```

问题：LCEL chain 的输入只有 `question` 一个变量，无法注入历史。

### 3.2 新版（有记忆）

改用函数手动组合，支持三个输入变量：

```python
def rag_query(question, session_id=None):
    # 1. 检索
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    # 2. 加载历史
    history = format_history(session_id)  # 从 SQLite 读取

    # 3. 组装 prompt
    prompt = PROMPT_WITH_HISTORY.format(
        context=context, history=history, question=question
    )

    # 4. 调用 LLM
    answer = llm.invoke(prompt)
    return answer.content
```

### 3.3 向后兼容

当 `session_id` 为 `None` 时，`format_history()` 返回空字符串，prompt 中 `{history}` 为空，行为与原版完全一致。

## 四、Prompt 模板设计

带历史的 prompt 模板增加了来源标注规范，要求 LLM 严格区分知识库内容和通用知识：

```
你是一个专业的课程助教。你的核心职责是基于提供的参考资料回答学生问题，
并结合对话上下文理解意图。

### 回答原则：
1. 严禁编造课程资料中的信息。
2. 无论问题是否与参考资料相关，你都必须给出有益的回复，
   但同时必须严格区分并标注信息的来源。

### 来源标注规范（必须严格执行）：
- 来自知识库：标注 [来源:参考资料]
- 超出知识库：先声明"该问题未在课程知识库中找到相关资料"，
  再用通用知识补充，标注 [来源:通用知识]
- 混合情况：分别标注，不能混淆

参考资料：
{context}

{history}
学生问题：
{question}
```

## 五、自动摘要

### 5.1 触发条件

每累积 **10 条消息**（5 轮问答）自动触发一次摘要生成。

### 5.2 实现方式

使用 FastAPI 的 `BackgroundTasks`，在后台线程中异步执行，不阻塞用户请求：

```python
background_tasks.add_task(generate_summary_task, session_id)
```

### 5.3 摘要用途

摘要存储在 `sessions.summary` 字段，前端可以展示在会话列表中，帮助用户快速了解每个会话的内容。

## 六、线程安全

### 6.1 SQLite 并发

FastAPI 使用 `asyncio.to_thread()` 将同步调用放到线程池执行，多个请求可能同时访问数据库。

解决方案：每次 DB 操作独立创建连接，使用 `check_same_thread=False`：

```python
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
```

### 6.2 WAL 模式

启用 SQLite 的 WAL（Write-Ahead Logging）模式，提升并发读写性能：

```python
conn.execute("PRAGMA journal_mode=WAL")
```

## 七、配置参数

| 常量 | 默认值 | 说明 |
|------|--------|------|
| `MAX_HISTORY_EXCHANGES` | 10 | 发给 LLM 的最近对话轮数 |
| `SESSION_MAX_AGE_DAYS` | 7 | 会话过期天数 |
| `SUMMARY_TRIGGER_COUNT` | 10 | 每 N 条消息触发自动摘要 |

## 八、前端适配

### 8.1 会话侧边栏

`web/index.html` 新增左侧会话列表面板，显示所有会话，支持点击切换和删除。

### 8.2 自动会话管理

`web/app.js` 的核心改动：

```javascript
// 首次提问时自动创建会话
if (!currentSessionId) {
    const sessRes = await fetch(`${API_URL}/sessions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: '新对话' })
    });
    const sess = await sessRes.json();
    currentSessionId = sess.session_id;
}

// 请求时自动携带 session_id
const body = { question: question };
if (currentSessionId) {
    body.session_id = currentSessionId;
}
```

### 8.3 用户操作流程

1. 打开页面，左侧显示会话列表
2. 直接输入问题并发送（无需手动创建会话）
3. 系统自动创建会话，后续追问自动关联
4. 可在左侧切换历史会话，查看对话记录
5. 可删除不需要的会话

## 九、使用示例

### Web 前端（推荐）

直接打开 `web/index.html`，输入问题即可。会话自动管理。

### API 调用

```bash
# 方式一：自动创建会话（不传 session_id，后端无状态处理）
curl -X POST http://localhost:8001/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是多AGV调度算法？"}'

# 方式二：手动创建会话并关联
# 1. 创建会话
curl -X POST http://localhost:8001/sessions \
  -H "Content-Type: application/json" \
  -d '{"title": "AGV调度讨论"}'
# 返回: {"session_id": "abc-123-...", ...}

# 2. 第一轮提问
curl -X POST http://localhost:8001/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是多AGV调度算法？", "session_id": "abc-123-..."}'

# 3. 追问（LLM 能理解"它"指代多AGV调度）
curl -X POST http://localhost:8001/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "那它有什么优缺点？", "session_id": "abc-123-..."}'

# 4. 查看对话历史
curl http://localhost:8001/sessions/abc-123-.../messages

# 5. 列出所有会话
curl http://localhost:8001/sessions

# 6. 清理过期会话
curl -X POST "http://localhost:8001/sessions/cleanup?max_age_days=7"
```

## 十、与原版的差异

| 特性 | main_siliconflow.py | main_siliconflow_memory.py |
|------|---------------------|---------------------------|
| 对话历史 | 无 | SQLite 持久化 |
| 上下文理解 | 每次独立，不理解追问 | 自动注入历史，理解指代 |
| RAG 链 | LCEL chain | 手动函数组合 |
| Prompt | 2 变量 (context, question) | 3 变量 (+history)，含来源标注规范 |
| 新增依赖 | 无 | sqlite3（Python 内置） |
| session_id | 不支持 | 可选参数，前端自动管理 |
| 会话管理 | 无 | CRUD + 自动标题 + 自动摘要 |
| 前端 | 无状态问答 | 会话侧边栏 + 自动创建 + 切换/删除 |
