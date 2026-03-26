
# RAG系统评估工具使用说明

## 文件说明

`backend/evaluate.py` - RAG系统评估脚本，用于计算Top-N召回率、精确率等指标

## 功能特性

1. **Top-K精确率评估**：计算 Precision@1, Precision@3, Precision@5
2. **Top-K召回率评估**：计算 Recall@1, Recall@3, Recall@5
3. **响应时间统计**：统计平均检索响应时间
4. **批量评估**：支持多个测试用例的批量评估
5. **报告生成**：自动生成评估报告

## 使用方法

### 1. 准备工作

确保：
- 已设置 `BAILIAN_API_KEY` 环境变量
- `course_knowledge_base` 目录存在且包含有效向量库
- 已安装所有依赖（在 `backend/requirements.txt` 中）

### 2. 设置环境变量

```bash
# Windows
set BAILIAN_API_KEY=你的百炼平台API密钥

# Linux/Mac
export BAILIAN_API_KEY=你的百炼平台API密钥
```

### 3. 配置测试用例

编辑 `evaluate.py` 中的 `get_sample_test_cases()` 函数，修改测试用例：

```python
def get_sample_test_cases():
    return [
        {
            "query": "您的问题1",
            "relevant_keywords": ["关键词1", "关键词2", "关键词3"]
        },
        {
            "query": "您的问题2",
            "relevant_keywords": ["关键词A", "关键词B"]
        }
    ]
```

**测试用例说明**：
- `query`: 要测试的查询问题
- `relevant_keywords`: 相关关键词列表，用于判断检索结果是否相关

### 4. 运行评估

```bash
cd d:\CV1
python backend\evaluate.py
```

## 评估指标说明

| 指标 | 说明 | 公式 |
|------|------|
| **Precision@1 | Top-1 精确率：检索到的第1个结果中，相关结果的比例 | 检索到的相关文档数 / 检索到的文档总数 |
| **Precision@3 | Top-3 精确率：检索到的前3个结果中，相关结果的比例 | 检索到的相关文档数 / 检索到的文档总数 |
| **Precision@5 | Top-5 精确率：检索到的前5个结果中，相关结果的比例 | 检索到的相关文档数 / 检索到的文档总数 |
| **Recall@1 | Top-1 召回率：在检索结果中找到的不同关键词数 / 总关键词数 | 找到的不同关键词数 / 总关键词数 |
| **Recall@3 | Top-3 召回率：在检索结果中找到的不同关键词数 / 总关键词数 | 找到的不同关键词数 / 总关键词数 |
| **Recall@5 | Top-5 召回率：在检索结果中找到的不同关键词数 / 总关键词数 | 找到的不同关键词数 / 总关键词数 |
| avg_response_time | 平均响应时间（秒） | - |

## 评估报告示例

```
============================================================
RAG系统评估报告
============================================================

处理测试用例 1/3...
  查询: 什么是矩阵制造车间？...
  Precision@3: 0.67
  Recall@3: 0.50

处理测试用例 2/3...
  查询: 多AGV调度系统研究的核心算法是什么？...
  Precision@3: 1.00
  Recall@3: 1.00

处理测试用例 3/3...
  查询: 单载量无人搬运车优化调度方法是怎样的？...
  Precision@3: 0.33
  Recall@3: 0.67

平均指标:
------------------------------------------------------------
precision@1              : 0.6667 (66.67%)
precision@3              : 0.6667 (66.67%)
precision@5              : 0.6667 (66.67%)
recall@1                 : 0.3333 (33.33%)
recall@3                 : 0.7222 (72.22%)
recall@5                 : 0.7778 (77.78%)
avg_response_time      : 0.2345 秒
============================================================
```

## 自定义评估

如果需要更复杂的评估逻辑，可以继承 `RAGEvaluator` 类并覆盖相应方法：

```python
class CustomRAGEvaluator(RAGEvaluator):
    def evaluate_recall_at_k(self, query: str, relevant_keywords: List[str], k: int = 3) -&gt; float:
        # 自定义召回率计算逻辑
        pass
    
    def evaluate_precision_at_k(self, query: str, relevant_keywords: List[str], k: int = 3) -&gt; float:
        # 自定义精确率计算逻辑
        pass
```

## 注意事项

1. 确保测试用例的关键词与课程资料内容匹配
2. 召回率和精确率基于关键词匹配，更准确的评估需要人工标注
3. 建议至少准备10-20个测试用例以获得更可靠的统计结果
4. 评估前确保知识库已正确初始化

## 扩展建议

1. 添加F1值计算（Precision和Recall的调和平均）
2. 添加MRR（Mean Reciprocal Rank）指标
3. 支持从JSON/CSV文件加载测试用例
4. 生成可视化图表展示评估结果
5. 添加端到端问答质量评估（不仅限于检索）

<!--
============================================================
RAG系统评估报告
============================================================

处理测试用例 1/3...
  查询: 什么是矩阵制造车间？...
  Precision@3: 1.00
  Recall@3: 1.00

处理测试用例 2/3...
  查询: 多AGV调度系统研究的核心算法是什么？...
  Precision@3: 1.00
  Recall@3: 0.40

处理测试用例 3/3...
  查询: 单载量无人搬运车优化调度方法是怎样的？...
  Precision@3: 1.00
  Recall@3: 0.50

平均指标:
------------------------------------------------------------
precision@1              : 1.0000 (100.00%)
precision@3              : 1.0000 (100.00%)
precision@5              : 1.0000 (100.00%)
recall@1                 : 0.5111 (51.11%)
recall@3                 : 0.6333 (63.33%)
recall@5                 : 0.6333 (63.33%)
avg_response_time        : 4.2801 秒
============================================================

增加检索数量(如果召回率足够，可以减少 k 值,降低响应时间)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

添加相似度阈值
# 使用相似度分数阈值，放宽匹配条件
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 5,
        "score_threshold": 0.3  # 相似度阈值，越低越宽松
    }
)
优化测试用例关键词
{
    "query": "什么是矩阵制造车间？",
    "relevant_keywords": ["矩阵", "制造车间", "生产", "布局", "调度", "优化"]  # 添加更多相关词
}

使用本地缓存
from functools import lru_cache
@lru_cache(maxsize=100)
def cached_retrieve(query: str):
    return retriever.invoke(query)

优化LLM参数
llm = ChatOpenAI(
    openai_api_key=bailian_api_key,
    openai_api_base=bailian_base_url,
    model_name=bailian_model,
    temperature=0.3,
    max_tokens=500,  # 限制输出长度
    request_timeout=10  # 设置超时
)    
 -->



<!-- 
文本嵌入模型（Embedding Model）
支持的模型：

MTEB、CMTEB 是 Embedding 模型的通用评估指标，数值越大，模型效果越好。text-embedding-v3 与 text-embedding-v4模型当前无法通过 LangChain 框架接口指定向量维度，默认采用 1024 维度作为输出向量维度值。

| 模型 | MTEB（Retrieval task） | CMTEB | CMTEB (Retrieval task) |
| --- | --- | --- | --- |
| text-embedding-v1 | 58.30 | 45.47 | 56.59 |
| text-embedding-v2 | 60.13 | 49.49 | 62.78 |
| text-embedding-v3（1024维度） | 63.39 | 55.41 | 73.23 |
| text-embedding-v4（1024维度） | 68.36 | 59.30 | 73.98 |


使用前需要安装以下依赖：
pip install langchain-community
pip install dashscope

模型调用：
from langchain_community.embeddings import DashScopeEmbeddings
embeddings = DashScopeEmbeddings(
    model="text-embedding-v4",
    # other params...
)

text = "This is a test document."

query_result = embeddings.embed_query(text)
print("文本向量长度：", len(query_result), sep='')

doc_results = embeddings.embed_documents(
    [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!"
    ])
print("文本向量数量：", len(doc_results), "，文本向量长度：", len(doc_results[0]), sep='')

详细介绍与更多使用方式请前往 LangChain 官方的 DashScope Embeddings。
完整的 API参考文档请前往 LangChain 官方的 Embedding API Reference。 
 -->