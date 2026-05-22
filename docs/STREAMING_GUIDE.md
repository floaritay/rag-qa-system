# 流式输出（Streaming）— 实现原理与改动说明

## 概述

系统支持 SSE（Server-Sent Events）流式输出，LLM 生成的回答逐 token 实时推送到前端，用户无需等待完整回答即可看到内容逐步呈现。同时兼容非流式模式，`/ask` 和 `/v1/chat/completions` 均支持流式。

## 实现原理

### 整体架构

```
用户提问
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  检索阶段（同步，与非流式完全一致）                        │
│  预检索优化 → 混合检索 → 重排序 → 上下文组装 → Prompt 构建  │
└─────────────────────────┬───────────────────────────┘
                          │ prompt_text 已就绪
                          ▼
┌─────────────────────────────────────────────────────┐
│  生成阶段（流式）                                      │
│  llm.stream(prompt_text) → AIMessageChunk 迭代器      │
│  每个 chunk 的 .content 是一小段文本（非累积）            │
└─────────────────────────┬───────────────────────────┘
                          │ SSE 事件流
                          ▼
┌─────────────────────────────────────────────────────┐
│  前端渲染                                             │
│  fetch + ReadableStream → 逐 token 累积 → renderMarkdown │
│  每次收到 token 都用完整累积文本重新渲染 Markdown          │
└─────────────────────────────────────────────────────┘
```

### 关键设计决策

**检索阶段不流式，生成阶段才流式。** 检索（向量搜索、BM25、重排序）必须在 LLM 生成之前完成，因为需要完整的上下文才能构建 Prompt。流式只作用于最后一步 `llm.stream()`。

**同步生成器 + StreamingResponse。** LangChain 的 `ChatOpenAI.stream()` 返回同步迭代器。使用 FastAPI 的 `StreamingResponse` 消费同步生成器，FastAPI 会在线程池中运行生成器，无需手动管理线程。

**每次用完整文本重新渲染 Markdown。** Markdown 结构（表格、代码块、列表）需要完整的定界符才能正确渲染。收到每个 token 后，用完整的累积文本调用 `renderMarkdown()` 重新生成 HTML。这是流式渲染的固有取舍——部分 Markdown 会产生短暂的渲染闪烁。

## SSE 协议

### `/ask/stream` 端点

每条 SSE 事件格式为 `data: {JSON}\n\n`，JSON 结构：

```jsonc
// 1. 来源事件（首条，检索完成后立即发送）
{"type": "sources", "data": [{"rank": 1, "source": "file.pdf", "page": 0, "content": "...", "vector_score": 1.23}]}

// 2. Token 事件（逐个发送）
{"type": "token", "data": "根据"}

// 3. 流结束标记
data: [DONE]

// 4. 错误事件（可选）
{"type": "error", "data": "错误描述"}
```

### `/v1/chat/completions` 端点（`stream: true`）

遵循 OpenAI SSE 格式：

```jsonc
// Token chunk
{"id": "chatcmpl-xxx", "object": "chat.completion.chunk", "created": 1234567890, "model": "knowledge-base",
 "choices": [{"index": 0, "delta": {"content": "根据"}, "finish_reason": null}]}

// 结束 chunk
{"id": "chatcmpl-xxx", "object": "chat.completion.chunk", "created": 1234567890, "model": "knowledge-base",
 "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

data: [DONE]
```

## 改动文件清单

### `backend/main_siliconflow_rag.py`

| 位置 | 改动 |
|------|------|
| 导入区 | 新增 `StreamingResponse`、`json` |
| `_retrieve_and_build_prompt()` | 新增函数，提取公共检索逻辑（预检索优化 → 检索 → 重排序 → Prompt 构建），供流式和非流式共用 |
| `rag_query_stream()` | 新增函数，同步生成器，先 yield sources 事件，再逐 token yield |
| `rag_query_stateless_stream()` | 新增函数，无状态版本的流式 RAG 查询 |
| `POST /ask/stream` | 新增端点，返回 `StreamingResponse(media_type="text/event-stream")` |
| `POST /v1/chat/completions` | 重构，当 `request.stream=True` 时返回 SSE 流（OpenAI chunk 格式），否则保持原有非流式行为 |

### `web/app.js`

| 位置 | 改动 |
|------|------|
| `askQuestion()` | 改为调用 `/ask/stream`，使用 `fetch` + `ReadableStream` 消费 SSE，逐 token 更新消息内容 |
| `createStreamingMessage()` | 新增函数，预创建空消息壳（含光标动画），返回 DOM 引用供流式更新 |
| `renderSourcesPanel()` | 新增函数，独立渲染来源面板，流式过程中收到 sources 事件时调用 |

### `web/styles.css`

| 位置 | 改动 |
|------|------|
| 文件末尾 | 新增 `.streaming-cursor` 光标闪烁动画、`.sources-panel-slot:empty` 隐藏样式 |

## 前端流式渲染流程

```
1. 用户发送问题
2. 创建空消息壳（含闪烁光标 ▋）
3. fetch /ask/stream，获取 ReadableStream
4. 循环读取流数据：
   ├─ 收到 sources 事件 → 渲染来源面板
   ├─ 收到 token 事件 → 累积文本 → renderMarkdown(完整文本) → 更新消息内容
   └─ 收到 [DONE] → 移除光标，结束
```

**前端 SSE 消费方式：** 不使用 `EventSource`（不支持 POST），而是 `fetch` + `response.body.getReader()` + `TextDecoder`，手动解析 SSE 行。

**缓冲区处理：** TCP 分包可能导致一条 SSE 事件被拆成多个 chunk，因此使用 `buffer` 变量缓存不完整的行，下次读取时拼接。

## 非流式兼容

- `POST /ask` 保持原有非流式行为不变，返回完整 JSON 响应
- `POST /v1/chat/completions` 当 `stream` 为 `false` 或省略时，返回原有 JSON 响应
- 前端设置面板中的检索策略参数（retrieval_strategy、pre_retrieval、post_retrieval）对流式和非流式均生效

## 注意事项

- **消息持久化：** 流式完成后才将完整回答保存到 SQLite，不会保存中间状态
- **会话摘要：** 流式完成后才触发自动摘要计数，行为与非流式一致
- **超时：** 流式端点无单独超时限制（检索阶段通常几秒内完成，LLM 生成通过流式逐 chunk 返回不会超时）
- **并发安全：** 全局 `llm` 实例可安全并发调用 `.stream()`，每次调用内部创建独立连接
- **错误处理：** 流式过程中出错会发送 `{"type": "error", "data": "..."}` 事件，前端跳过该事件
