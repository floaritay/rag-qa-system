# 检索优化版 — 实现原理与流程

## 概述

`backend/main_siliconflow_rag.py` 在会话管理与短期记忆版本基础上，新增了三类检索优化能力：

1. **预检索优化**：查询改写（Query Rewriting）、假设性文档嵌入（HyDE）
2. **检索策略优化**：混合检索（向量 + BM25 关键词融合）
3. **后检索优化**：重排序（Reranking）

所有优化均可通过 API 参数按需组合，默认行为与原版完全一致。

## 整体架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                         前端 (web/)                                   │
│                                                                       │
│  ┌─────────────┐   ┌──────────────────┐   ┌───────────────────────┐  │
│  │ 会话侧边栏   │   │    聊天区域       │   │   检索策略配置面板     │  │
│  │              │   │                  │   │ 检索方式/预检索/后检索  │  │
│  └─────────────┘   └──────────────────┘   └───────────────────────┘  │
└────────────────────────────┬──────────────────────────────────────────┘
                             │ POST /ask { question, session_id?,
                             │            retrieval_strategy,
                             │            pre_retrieval, post_retrieval }
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        FastAPI 服务端                                 │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │                    RAG 查询引擎                                │    │
│  │                                                               │    │
│  │  ┌─────────────┐   ┌──────────────┐   ┌───────────────────┐  │    │
│  │  │ 预检索优化    │   │  检索阶段     │   │  后检索优化        │  │    │
│  │  │ 查询改写     │──→│ 向量/混合检索  │──→│  重排序            │  │    │
│  │  │ HyDE        │   │              │   │                   │  │    │
│  │  └─────────────┘   └──────────────┘   └───────────────────┘  │    │
│  │           │              │                    │               │    │
│  │           ▼              ▼                    ▼               │    │
│  │     改写/生成       FAISS + BM25         BAAI/bge-           │    │
│  │     检索词          RRF 融合             reranker-v2-m3      │    │
│  │                                            │               │    │
│  │                                            ▼               │    │
│  │                                     ┌──────────────┐       │    │
│  │                                     │  Top-3 文档   │       │    │
│  │                                     └──────┬───────┘       │    │
│  └────────────────────────────────────────────┼───────────────┘    │
│                                               │                     │
│  ┌──────────────┐    ┌──────────────┐         │                     │
│  │  会话管理     │    │  SQLite DB    │         ▼                     │
│  │ /sessions    │    │              │   ┌──────────────┐            │
│  │ CRUD 端点    │    │              │   │  组装 Prompt   │            │
│  └──────────────┘    └──────────────┘   │ context+history│            │
│                                         │ +question      │            │
│                                         └──────┬───────┘            │
│                                                │                     │
│                                                ▼                     │
│                                          LLM 生成回答                │
└──────────────────────────────────────────────────────────────────────┘
```

## 一、预检索优化

预检索优化在用户问题送入检索引擎之前，对查询进行变换，提升检索命中率。

### 1.1 查询改写（Query Rewriting）

#### 问题场景

用户口语化提问包含代词和省略，直接作为检索词效果差：

```
用户第一轮：什么是多AGV调度算法？
用户第二轮：那它有什么优缺点？
                              ↑
                         "它"指代不明，检索命中率低
```

#### 实现原理

调用 LLM 将口语化问题改写为规范化检索词：

```python
QUERY_REWRITE_PROMPT = """你是一个检索优化助手。请将以下用户问题改写为更适合向量检索的规范化查询词。
要求：
1. 去除口语化表达和代词（如"它"、"这个"、"那"）
2. 补充缺失的主语（结合对话历史推断代词指代）
3. 保留核心专业术语
4. 只输出改写后的查询词，不要解释

对话历史：
{history}

用户问题：{question}

改写后的查询词："""
```

#### 流程

```
用户问题："那它有什么优缺点？"
         │
         ▼
┌─────────────────────────┐
│ 1. 加载对话历史           │
│    "什么是多AGV调度算法？" │
│    "多AGV调度是指..."     │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ 2. LLM 改写              │
│    → "多AGV调度算法优缺点" │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ 3. 改写结果仅用于检索     │
│    原始问题仍送入最终prompt│
└─────────────────────────┘
```

#### 关键设计

- 改写后的查询词**仅用于检索**，最终 prompt 仍使用用户的原始问题，保持回答的自然性
- 有 session 时注入对话历史，帮助 LLM 解析代词；无 session 时仅做去口语化处理

### 1.2 HyDE（Hypothetical Document Embedding）

#### 问题场景

用户问题通常简短且口语化（如"AGV怎么调度"），而课程资料中的表述更专业冗长（如"多载量自动导引车系统的任务分配与路径规划方法"）。两者在向量空间中距离较远。

#### 实现原理

让 LLM 生成一段假设性答案，其用词更接近课程资料的表述风格，用假设答案的 embedding 做检索，命中率更高。

```python
HYDE_PROMPT = """请根据以下问题，写一段简短的假设性答案（约100字）。
这段答案不需要准确，只需包含可能出现在课程资料中的专业术语和表述方式。

问题：{question}

假设性答案："""
```

#### 流程

```
用户问题："AGV怎么调度？"
         │
         ▼
┌──────────────────────────────────────────┐
│ 1. LLM 生成假设性答案                      │
│    "多AGV调度系统采用遗传算法等智能优化方法  │
│     进行任务分配与路径规划，通过冲突检测     │
│     与解决策略实现高效协同运输..."           │
└────────────┬─────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────┐
│ 2. 对假设答案做 embedding                  │
│    embeddings.embed_query(假设答案)        │
└────────────┬─────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────┐
│ 3. 用假设答案向量在 FAISS 中检索            │
│    vectorstore.similarity_search_by_vector│
└────────────┬─────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────┐
│ 4. 检索结果组装 context                    │
│    最终 prompt 使用用户原始问题（非假设答案）│
└──────────────────────────────────────────┘
```

#### 关键设计

- 假设答案**仅用于生成检索向量**，不会出现在最终 prompt 中
- 假设答案不需要准确，关键是包含领域术语，拉近与课程资料的向量距离

## 二、检索策略优化

### 2.1 默认策略（向量检索）

原版方案，使用 FAISS 向量相似度检索：

```
用户问题 → embedding → FAISS similarity_search(k=15) → Top-3
```

### 2.2 混合检索（Hybrid Retrieval）

#### 问题场景

纯向量检索依赖语义相似度，对精确关键词匹配能力不足。例如用户搜索"遗传算法"，向量检索可能返回包含"智能优化方法"的段落，但遗漏直接提及"遗传算法"的段落。

#### 实现原理

将 FAISS 向量检索与 BM25 关键词检索的结果通过**倒数排名融合（Reciprocal Rank Fusion, RRF）**合并：

```
                    用户查询
                   ╱        ╲
                  ╱          ╲
                 ▼            ▼
        ┌──────────────┐  ┌──────────────┐
        │ FAISS 向量检索 │  │ BM25 关键词检索│
        │  Top-15      │  │  Top-15      │
        └──────┬───────┘  └──────┬───────┘
               │                 │
               ▼                 ▼
        ┌──────────────────────────────────┐
        │    倒数排名融合（RRF）              │
        │                                  │
        │  score(d) = Σ wᵢ / (rankᵢ + k)  │
        │                                  │
        │  向量权重 w₁ = 0.7               │
        │  BM25 权重 w₂ = 0.3              │
        │  常数 k = 60                     │
        └──────────────┬───────────────────┘
                       │
                       ▼
                 融合后 Top-3
```

#### BM25 索引构建

在 `create_vectorstore()` 构建 FAISS 向量库时，同步构建 BM25 索引并持久化：

```python
# 中文分词：字符 bigram + 英文单词
def tokenize(text: str) -> list:
    text = text.lower().strip()
    tokens = []
    en_tokens = re.findall(r'[a-zA-Z]+', text)     # 英文按空格
    tokens.extend(en_tokens)
    cn_chars = re.findall(r'[一-鿿]', text)          # 中文字符
    for i in range(len(cn_chars) - 1):
        tokens.append(cn_chars[i] + cn_chars[i + 1]) # bigram
    tokens.extend(cn_chars)
    return tokens

# 构建并保存
tokenized_corpus = [tokenize(doc.page_content) for doc in all_docs]
bm25_data = {"tokenized_corpus": tokenized_corpus, "documents": all_docs}
pickle.dump(bm25_data, open("course_knowledge_base/bm25_index.pkl", "wb"))
```

#### RRF 融合算法

```python
def hybrid_retrieve(query, k=3):
    # 向量检索 Top-15
    vector_docs = vectorstore.similarity_search(query, k=15)
    # BM25 检索 Top-15
    scores = bm25_index.get_scores(tokenize(query))
    bm25_top = sorted(...)[:15]

    # RRF 融合
    for rank, doc in enumerate(vector_docs):
        score += VECTOR_WEIGHT / (rank + 60)   # 向量排名贡献
    for rank, doc in enumerate(bm25_top):
        score += BM25_WEIGHT / (rank + 60)     # BM25 排名贡献

    # 按融合分数排序，取 Top-k
    return sorted(by_score)[:k]
```

#### 设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 分词方案 | 字符 bigram | 无需额外依赖（jieba），对专业术语覆盖好 |
| 融合算法 | RRF | 不依赖绝对分数，只看排名，对不同检索器的分数尺度差异免疫 |
| 候选数量 | 15 | 给后续重排序足够候选，同时控制计算量 |
| 权重分配 | 向量 0.7 / BM25 0.3 | 向量检索语义理解更强，作为主导；BM25 补充精确匹配 |

## 三、后检索优化

### 3.1 重排序（Reranking）

#### 问题场景

向量检索的相似度分数不等于相关性排序。一个文档可能与查询向量距离近但实际不相关（共享通用词汇），而真正相关的文档排名靠后。

#### 实现原理

使用硅基流动的重排序模型 `BAAI/bge-reranker-v2-m3`，对检索结果做精排。重排序模型是交叉编码器（cross-encoder），同时编码 query 和 document，相关性判断更准确。

```python
class SiliconFlowReranker:
    def rerank(self, query, documents, top_n=3):
        # POST https://api.siliconflow.cn/v1/rerank
        # {
        #   "model": "BAAI/bge-reranker-v2-m3",
        #   "query": "用户问题",
        #   "documents": ["文档1内容", "文档2内容", ...],
        #   "top_n": 3,
        #   "return_documents": false
        # }
        # 返回: [{"index": 2, "relevance_score": 0.95}, ...]
        # 按 index 映射回原始 Document 对象
```

#### 流程

```
检索阶段 Top-15 文档
         │
         ▼
┌──────────────────────────────────────┐
│ SiliconFlow Rerank API               │
│                                      │
│ 输入：                                │
│   query = 用户原始问题（非改写结果）    │
│   documents = 15 篇文档内容           │
│   top_n = 3                          │
│                                      │
│ 处理：                                │
│   Cross-encoder 逐篇计算相关性分数     │
│                                      │
│ 输出：                                │
│   按相关性重排后的 Top-3 文档          │
└──────────────┬───────────────────────┘
               │
               ▼
         重排后 Top-3 → 组装 context
```

#### 关键设计

- 重排序使用**用户原始问题**，而非查询改写后的检索词。重排序模型需要精确的语义匹配，改写词可能丢失细节
- 检索阶段取 15 篇候选，重排序后取 3 篇，给精排模型足够选择空间
- API 调用失败时降级返回原始检索结果的 Top-3，不影响可用性

## 四、优化策略组合

### 4.1 可配置参数

`Query` 模型新增三个可选参数：

```python
class Query(BaseModel):
    question: str
    session_id: Optional[str] = None
    retrieval_strategy: Optional[str] = "default"   # "default" | "hybrid"
    pre_retrieval: Optional[str] = "none"            # "none" | "rewrite" | "hyde"
    post_retrieval: Optional[str] = "none"           # "none" | "rerank"
```

### 4.2 策略组合矩阵

| 组合 | pre_retrieval | retrieval_strategy | post_retrieval | 适用场景 |
|------|--------------|-------------------|---------------|---------|
| 默认 | none | default | none | 快速问答，无需优化 |
| 改写+默认 | rewrite | default | none | 追问/代词多的对话 |
| HyDE+默认 | hyde | default | none | 简短口语化问题 |
| 默认+混合 | none | hybrid | none | 精确关键词匹配需求 |
| 默认+重排 | none | default | rerank | 提升排序精度 |
| 混合+重排 | none | hybrid | rerank | 全面提升检索质量 |
| 全组合 | rewrite | hybrid | rerank | 最高质量，延迟较高 |

### 4.3 完整 rag_query 流程

```python
def rag_query(question, session_id=None,
              retrieval_strategy="default",
              pre_retrieval="none",
              post_retrieval="none"):

    # ── 预检索优化 ──
    search_query = question
    if pre_retrieval == "rewrite":
        search_query = rewrite_query(question, session_id)  # +1 LLM 调用
    elif pre_retrieval == "hyde":
        search_query = hyde_generate(question)               # +1 LLM 调用

    # ── 检索 ──
    if retrieval_strategy == "hybrid":
        docs = hybrid_retrieve(search_query, k=15)           # FAISS + BM25
    elif pre_retrieval == "hyde":
        hyde_embedding = embeddings.embed_query(search_query)
        docs = vectorstore.similarity_search_by_vector(hyde_embedding, k=15)
    else:
        docs = retriever.invoke(search_query)                # 纯 FAISS

    # ── 后检索优化 ──
    if post_retrieval == "rerank":
        docs = reranker.rerank(question, docs, top_n=3)      # +1 API 调用

    # ── 组装 Prompt ──
    top_docs = docs[:3]
    context = "\n\n".join([doc.page_content for doc in top_docs])
    history_text = format_history(session_id)
    prompt = PROMPT_WITH_HISTORY.format(
        context=context, history=history_text, question=question
    )

    # ── LLM 生成 ──
    answer = llm.invoke(prompt)
    return answer.content
```

### 4.4 延迟与代价

| 策略 | 额外 LLM 调用 | 额外 API 调用 | 预估延迟增加 |
|------|--------------|--------------|-------------|
| 默认 | 0 | 0 | 0s |
| rewrite | 1 (改写) | 0 | +3~5s |
| hyde | 1 (生成) | 0 | +3~5s |
| hybrid | 0 | 0 | +1~2s |
| rerank | 0 | 1 (重排序) | +1~2s |
| 全组合 | 2 (改写+LLM) | 1 (重排序) | +8~12s |

## 五、前端适配

### 5.1 检索策略配置面板

`web/index.html` 右侧边栏新增"检索策略"卡片，包含三个下拉框：

```
┌─────────────────────────┐
│ 🔧 检索策略              │
│                         │
│ 检索方式                 │
│ ┌─────────────────────┐ │
│ │ 默认（向量检索）   ▾ │ │
│ └─────────────────────┘ │
│                         │
│ 预检索优化               │
│ ┌─────────────────────┐ │
│ │ 无                ▾ │ │
│ └─────────────────────┘ │
│                         │
│ 后检索优化               │
│ ┌─────────────────────┐ │
│ │ 无                ▾ │ │
│ └─────────────────────┘ │
└─────────────────────────┘
```

### 5.2 请求传递

`web/app.js` 在发送请求时自动读取用户选择：

```javascript
const body = { question: question };
if (currentSessionId) body.session_id = currentSessionId;
body.retrieval_strategy = elements.retrievalStrategy.value;
body.pre_retrieval = elements.preRetrieval.value;
body.post_retrieval = elements.postRetrieval.value;
```

默认值均为"无"/"默认"，不影响现有行为。

## 六、新增依赖

| 包 | 用途 | 安装 |
|---|------|------|
| `rank_bm25` | BM25 关键词检索算法 | `pip install rank_bm25` |

硅基流动重排序模型通过 HTTP API 调用，无需额外安装。

## 七、配置参数汇总

| 常量 | 默认值 | 说明 |
|------|--------|------|
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | 重排序模型 |
| `VECTOR_WEIGHT` | 0.7 | 混合检索中向量权重 |
| `BM25_WEIGHT` | 0.3 | 混合检索中 BM25 权重 |
| `HYBRID_CANDIDATE_K` | 15 | 检索阶段候选文档数 |
| `DEFAULT_TOP_K` | 3 | 最终返回文档数 |
| `MAX_HISTORY_EXCHANGES` | 10 | 发给 LLM 的最近对话轮数 |
| `SESSION_MAX_AGE_DAYS` | 7 | 会话过期天数 |
| `SUMMARY_TRIGGER_COUNT` | 10 | 每 N 条消息触发自动摘要 |

## 八、API 端点

与会话管理版本一致，`/ask` 端点新增三个可选参数：

```
POST /ask
{
    "question": "多AGV调度的优缺点？",
    "session_id": "可选",
    "retrieval_strategy": "default" | "hybrid",
    "pre_retrieval": "none" | "rewrite" | "hyde",
    "post_retrieval": "none" | "rerank"
}
```

其余端点（`/sessions`、`/health`、`/init`、`/v1/chat/completions` 等）行为不变。

## 九、与会话管理版本的差异

| 特性 | main_siliconflow_memory.py | main_siliconflow_rag.py |
|------|---------------------------|------------------------|
| 检索方式 | 纯 FAISS 向量检索 | 向量检索 + BM25 混合检索 |
| 查询优化 | 无 | 查询改写 / HyDE |
| 结果精排 | 无 | 硅基流动重排序模型 |
| 检索候选数 | k=3 | k=15，重排序后取 3 |
| BM25 索引 | 无 | 构建并持久化 bm25_index.pkl |
| API 参数 | question, session_id | +retrieval_strategy, pre_retrieval, post_retrieval |
| 新增依赖 | 无 | rank_bm25 |
| 超时设置 | 120s | 180s（考虑额外 LLM 调用） |

## 十、使用示例

### 默认策略（向后兼容）

```bash
curl -X POST http://localhost:8001/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是多AGV调度算法？"}'
```

### 查询改写（追问场景）

```bash
curl -X POST http://localhost:8001/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "那它有什么优缺点？", "session_id": "xxx", "pre_retrieval": "rewrite"}'
```

### HyDE（口语化问题）

```bash
curl -X POST http://localhost:8001/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "AGV怎么调度？", "pre_retrieval": "hyde"}'
```

### 混合检索 + 重排序

```bash
curl -X POST http://localhost:8001/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "矩阵制造车间的调度方法？", "retrieval_strategy": "hybrid", "post_retrieval": "rerank"}'
```

### 全部优化组合

```bash
curl -X POST http://localhost:8001/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "那它的优化效果怎么样？", "session_id": "xxx", "retrieval_strategy": "hybrid", "pre_retrieval": "rewrite", "post_retrieval": "rerank"}'
```

### 强制重建知识库（含 BM25 索引）

```bash
curl -X POST "http://localhost:8001/init?force_rebuild=true"
```
