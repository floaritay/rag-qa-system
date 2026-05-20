# 检索来源展示功能

## 概述

在 RAG 问答的响应中新增 `sources` 字段，返回检索阶段命中的文档片段及其元信息（排名、来源文件、页码、重排序分数）。前端以可折叠面板的形式展示，便于用户验证答案的依据和调试检索质量。

## 变更范围

| 文件 | 变更内容 |
|------|---------|
| `backend/main_siliconflow_rag.py` | 新增 `extract_sources()`；`rag_query` / `rag_query_stateless` 返回值由 `str` 改为 `tuple`；重排序器附加 `rerank_score` 到 metadata |
| `web/app.js` | `addMessage()` 支持渲染 sources 面板；`askQuestion()` 传递 sources 数据 |
| `web/styles.css` | 新增 `.sources-panel`、`.source-item`、`.source-score` 等样式 |

## 后端实现

### extract_sources()

```python
def extract_sources(docs, top_k=DEFAULT_TOP_K):
    sources = []
    for i, doc in enumerate(docs[:top_k]):
        meta = doc.metadata if hasattr(doc, 'metadata') else {}
        source = {
            "rank": i + 1,
            "content": doc.page_content[:500],
            "source": meta.get("source", ""),
            "page": meta.get("page", None),
        }
        if "vector_score" in meta:
            source["vector_score"] = meta["vector_score"]
        if "rrf_score" in meta:
            source["rrf_score"] = meta["rrf_score"]
        if "rerank_score" in meta:
            source["rerank_score"] = meta["rerank_score"]
        sources.append(source)
    return sources
```

从 LangChain `Document` 对象列表中提取以下字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `rank` | `int` | 在最终排序中的位次（从 1 开始） |
| `content` | `str` | 文档片段内容，截取前 500 字符 |
| `source` | `str` | 源文件路径（PDF 文件名） |
| `page` | `int \| None` | PDF 页码（PyPDFLoader 提供，从 0 开始） |
| `vector_score` | `float` | FAISS 向量检索的 L2 距离，越小越相关（默认/HyDE 路径存在） |
| `rrf_score` | `float` | RRF 融合分数，越大越相关（仅混合检索路径存在） |
| `rerank_score` | `float` | 重排序相关度分数，0~1，越大越相关（仅启用后检索重排序时存在） |

### 函数签名变更

`rag_query` 和 `rag_query_stateless` 的返回值从 `str` 改为 `tuple[str, list]`：

```python
# 变更前
def rag_query(...) -> str:
    ...
    return answer_text

# 变更后
def rag_query(...) -> tuple:
    ...
    return answer_text, sources
```

调用方（`/ask` 和 `/v1/chat/completions`）已同步更新为解构赋值：

```python
answer, sources = await asyncio.wait_for(
    asyncio.to_thread(rag_query, ...), timeout=180
)
```

### 重排序分数透传

`SiliconFlowReranker.rerank()` 在排序后将 `relevance_score` 写入 `doc.metadata`：

```python
doc.metadata['rerank_score'] = round(score, 4)
```

这样 `extract_sources()` 可以统一读取，无需为重排序单独处理返回格式。

### 向量检索分数

默认检索和 HyDE 路径改用 `similarity_search_with_score` / `similarity_search_with_score_by_vector`，返回 `(Document, float)` 元组。FAISS 返回的分数为 L2 距离（欧氏距离），值越小表示越相似。分数写入 `doc.metadata['vector_score']`。

```python
scored = vectorstore.similarity_search_with_score(search_query, k=HYBRID_CANDIDATE_K)
for doc, score in scored:
    doc.metadata['vector_score'] = round(float(score), 4)
    docs.append(doc)
```

### RRF 融合分数

`hybrid_retrieve` 在倒数排名融合阶段已计算每个文档的累积分数 `doc_scores[key]["score"]`，现在将该分数写入 `doc.metadata['rrf_score']`。RRF 分数越大表示在两个检索通道中的综合排名越靠前。

```python
for item in sorted_items[:k]:
    doc = item["doc"]
    doc.metadata['rrf_score'] = round(item["score"], 6)
    result.append(doc)
```

## API 响应格式

`POST /ask` 响应中的 `sources` 字段：

```json
{
  "answer": "...",
  "session_id": "...",
  "sources": [
    {
      "rank": 1,
      "content": "多AGV调度系统的核心算法包括...",
      "source": "D:\\course_materials\\chapter3.pdf",
      "page": 12,
      "vector_score": 0.8234,
      "rerank_score": 0.8934
    },
    {
      "rank": 2,
      "content": "路径规划采用A*算法...",
      "source": "D:\\course_materials\\chapter3.pdf",
      "page": 15,
      "vector_score": 1.0521,
      "rerank_score": 0.7621
    }
  ]
}
```

不同检索路径返回的分数字段不同：

| 路径 | vector_score | rrf_score | rerank_score |
|------|:---:|:---:|:---:|
| 默认向量检索 | ✓ | — | — |
| HyDE | ✓ | — | — |
| 混合检索 | — | ✓ | — |
| 默认 + 重排序 | ✓ | — | ✓ |
| 混合 + 重排序 | — | ✓ | ✓ |

## 前端展示

每条助手回复下方显示「检索来源（N 条）」折叠按钮，点击展开后呈现：

- **排名标签**：`#1` `#2` ... 以铜色高亮
- **文件名与页码**：从完整路径中提取文件名，页码 +1 显示（适配用户习惯）
- **分数标签**：
  - 蓝色 `距离 x.xx`：FAISS L2 距离（默认/HyDE 路径）
  - 紫色 `RRF x.xxxx`：RRF 融合分数（混合检索路径）
  - 绿色 `xx.x%`：重排序相关度（启用重排序时）
- **内容预览**：最多显示 4 行，超出部分以省略号截断
- 各标签悬停显示 tooltip，说明含义和方向（越大越好 / 越小越好）

## 与历史消息的关系

来源信息仅在实时问答时返回，不持久化到数据库。从会话历史加载的消息不显示来源面板——这符合设计意图：来源信息用于即时验证，而非长期参考。
