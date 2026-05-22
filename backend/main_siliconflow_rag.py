from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, timedelta
import uuid
import time
import asyncio
import sqlite3
import pickle
import json
import re
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi
import os
import uvicorn
import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

app = FastAPI(title="知识库API（检索优化版）")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 配置常量
# ============================================================
MAX_HISTORY_EXCHANGES = 10
SESSION_MAX_AGE_DAYS = 7
SUMMARY_TRIGGER_COUNT = 10

# 检索优化配置
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
VECTOR_WEIGHT = 0.7       # 混合检索中向量检索权重
BM25_WEIGHT = 0.3          # 混合检索中 BM25 权重
HYBRID_CANDIDATE_K = 15    # 混合检索候选数量
DEFAULT_TOP_K = 3          # 最终返回文档数

# ============================================================
# Pydantic 模型
# ============================================================

class Query(BaseModel):
    question: str
    session_id: Optional[str] = None
    retrieval_strategy: Optional[str] = "default"   # "default" | "hybrid"
    pre_retrieval: Optional[str] = "none"            # "none" | "rewrite" | "hyde"
    post_retrieval: Optional[str] = "none"           # "none" | "rerank"

class Response(BaseModel):
    answer: str
    sources: list = []
    session_id: Optional[str] = None

class OpenAIModel(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "knowledge-base"

class OpenAIModelsResponse(BaseModel):
    object: str = "list"
    data: List[OpenAIModel]

class OpenAIMessage(BaseModel):
    role: str
    content: str

class OpenAIChatRequest(BaseModel):
    model: str = "knowledge-base"
    messages: List[OpenAIMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    session_id: Optional[str] = None
    retrieval_strategy: Optional[str] = "default"
    pre_retrieval: Optional[str] = "none"
    post_retrieval: Optional[str] = "none"

class OpenAIChatChoice(BaseModel):
    index: int = 0
    message: OpenAIMessage
    finish_reason: str = "stop"

class OpenAIUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class OpenAIChatResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "knowledge-base"
    choices: List[OpenAIChatChoice]
    usage: OpenAIUsage = OpenAIUsage()

# 会话相关模型
class SessionCreate(BaseModel):
    title: Optional[str] = "新对话"

class SessionInfo(BaseModel):
    session_id: str
    title: str
    created_at: str
    updated_at: str
    summary: Optional[str] = None
    message_count: int = 0

class SessionListResponse(BaseModel):
    sessions: List[SessionInfo]

class MessageInfo(BaseModel):
    role: str
    content: str
    created_at: str

class SessionHistoryResponse(BaseModel):
    session_id: str
    messages: List[MessageInfo]

# ============================================================
# Prompt 模板
# ============================================================

prompt_template = """你是一个专业的 AI 助手。你的核心职责是基于提供的参考资料回答用户问题。

### 回答原则：
1. 严禁编造知识库中的信息。
2. 无论问题是否与参考资料相关，你都必须给出有益的回复，但同时必须严格区分并标注信息的来源。

### 来源标注规范（必须严格执行）：
- **来自知识库**：陈述参考资料中的内容时，必须在对应句子或段落末尾标注，如 [来源:xx文件xx节]。
- **超出知识库**：如果问题无法从参考资料中找到答案，你必须先明确声明"该问题未在知识库中找到相关资料"，然后可以调用你的通用知识进行补充解答，并在补充内容后标注 [来源:通用知识]。
- **混合情况**：如果回答中既有参考资料的内容，又有你补充的通用知识，必须分别标注，绝不能混淆。

### 回答格式要求：
1. 语言简洁，逻辑清晰，使用列表或分段提升可读性。

---
参考资料：
{context}
---
用户问题：
{question}
---
### 回答"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

prompt_template_with_history = """你是一个专业的 AI 助手。你的核心职责是基于提供的参考资料回答用户问题，并结合对话上下文理解意图。

### 回答原则：
1. 严禁编造知识库中的信息。
2. 无论问题是否与参考资料相关，你都必须给出有益的回复，但同时必须严格区分并标注信息的来源。

### 来源标注规范（必须严格执行）：
- **来自知识库**：陈述参考资料中的内容时，必须在对应句子或段落末尾标注，如 [来源:xx文件xx节]。
- **超出知识库**：如果问题无法从参考资料中找到答案，你必须先明确声明"该问题未在知识库中找到相关资料"，然后可以调用你的通用知识进行补充解答，并在补充内容后标注 [来源:通用知识]。
- **混合情况**：如果回答中既有参考资料的内容，又有你补充的通用知识，必须分别标注，绝不能混淆。

### 回答格式要求：
1. 语言简洁，逻辑清晰，使用列表或分段提升可读性。
2. 结合对话上下文，准确理解代词和省略的指代对象（如"它"、"这个方法"、"请详细解释"等）。

---
参考资料：
{context}
---
历史对话：
{history}
---
用户问题：
{question}
---
### 回答："""

PROMPT_WITH_HISTORY = PromptTemplate(
    template=prompt_template_with_history,
    input_variables=["context", "history", "question"]
)

SUMMARY_PROMPT_TEMPLATE = "请用2-3句话总结以下对话的主要内容：\n\n{conversation}\n\n总结："

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

HYDE_PROMPT = """请根据以下问题，写一段简短的假设性答案（约100字）。
这段答案不需要准确，只需包含可能出现在文档中的专业术语和表述方式。

问题：{question}

假设性答案："""

# ============================================================
# 全局变量
# ============================================================
vectorstore = None
retriever = None
embeddings = None
llm = None
reranker = None

# BM25 相关
bm25_index = None
bm25_docs = None

# 嵌入模型配置（OpenAI 兼容 API）
embedding_base_url = os.getenv("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1")
embedding_api_key = os.getenv("EMBEDDING_API_KEY")
embedding_model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# 重排序模型配置（OpenAI 兼容 API，默认使用嵌入模型的 API）
reranker_base_url = os.getenv("RERANKER_BASE_URL") or embedding_base_url
reranker_api_key = os.getenv("RERANKER_API_KEY") or embedding_api_key

# LLM 配置（OpenAI 兼容 API）
llm_base_url = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
llm_api_key = os.getenv("LLM_API_KEY")
llm_model = os.getenv("LLM_MODEL", "qwen3.5-122b-a10b")

# 数据库路径
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(base_dir, "backend", "sessions.db")

# ============================================================
# OpenAICompatibleEmbeddings 类
# ============================================================

class OpenAICompatibleEmbeddings(Embeddings):
    """OpenAI 兼容嵌入模型（默认硅基流动 BAAI/bge-m3）"""
    def __init__(self, api_key=None, base_url=None, model=None):
        self.api_key = api_key or embedding_api_key
        self.base_url = base_url or embedding_base_url
        self.model = model or embedding_model_name

    def _get_embeddings(self, texts_list):
        url = f"{self.base_url}/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": self.model,
            "input": texts_list
        }

        try:
            response = requests.post(url, headers=headers, json=data, timeout=120)
            response.raise_for_status()
            result = response.json()
            embeddings = [item['embedding'] for item in result['data']]
            return embeddings
        except Exception as e:
            print(f"调用嵌入模型API失败: {e}")
            print(f"响应内容: {response.text if 'response' in locals() else '无响应'}")
            return None

    def embed_documents(self, texts):
        result = self._get_embeddings(texts)
        return result if result else []

    def embed_query(self, text):
        result = self._get_embeddings([text])
        return result[0] if result else []

# ============================================================
# OpenAICompatibleReranker 类（后检索优化）
# ============================================================

class OpenAICompatibleReranker:
    """OpenAI 兼容重排序模型（默认硅基流动 BAAI/bge-reranker-v2-m3）"""
    def __init__(self, api_key=None, base_url=None, model=None, top_n=DEFAULT_TOP_K):
        self.api_key = api_key or reranker_api_key
        self.base_url = base_url or reranker_base_url
        self.model = model or RERANKER_MODEL
        self.top_n = top_n

    def rerank(self, query: str, documents: list, top_n: int = None) -> list:
        if not documents:
            return []
        top_n = top_n or self.top_n

        doc_texts = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in documents]

        url = f"{self.base_url}/rerank"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": self.model,
            "query": query,
            "documents": doc_texts,
            "top_n": min(top_n, len(doc_texts)),
            "return_documents": False
        }

        try:
            response = requests.post(url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            result = response.json()
            reranked = []
            for item in result['results']:
                idx = item['index']
                score = item['relevance_score']
                doc = documents[idx]
                # Attach rerank score to metadata
                if hasattr(doc, 'metadata'):
                    doc.metadata['rerank_score'] = round(score, 4)
                reranked.append(doc)
            return reranked
        except Exception as e:
            print(f"重排序API调用失败: {e}")
            print(f"响应内容: {response.text if 'response' in locals() else '无响应'}")
            return documents[:top_n]

# ============================================================
# 中文分词工具（BM25 用）
# ============================================================

def tokenize(text: str) -> list:
    """中英文混合分词：中文按字符 bigram，英文按空格"""
    text = text.lower().strip()
    tokens = []
    # 英文单词
    en_tokens = re.findall(r'[a-zA-Z]+', text)
    tokens.extend(en_tokens)
    # 中文字符 bigram
    cn_chars = re.findall(r'[一-鿿]', text)
    for i in range(len(cn_chars) - 1):
        tokens.append(cn_chars[i] + cn_chars[i + 1])
    # 单个中文字符也保留
    tokens.extend(cn_chars)
    return tokens

# ============================================================
# SQLite 数据库函数
# ============================================================

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            title TEXT DEFAULT '新对话',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            summary TEXT DEFAULT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
    """)
    conn.commit()
    conn.close()
    print("数据库初始化完成")

def create_session(session_id: str, title: str = "新对话") -> dict:
    now = datetime.now().isoformat()
    conn = get_db()
    conn.execute(
        "INSERT INTO sessions (session_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (session_id, title, now, now)
    )
    conn.commit()
    conn.close()
    return {"session_id": session_id, "title": title, "created_at": now, "updated_at": now}

def get_session(session_id: str) -> Optional[dict]:
    conn = get_db()
    row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def list_sessions() -> List[dict]:
    conn = get_db()
    rows = conn.execute("""
        SELECT s.*, COUNT(m.id) as message_count
        FROM sessions s
        LEFT JOIN messages m ON s.session_id = m.session_id
        GROUP BY s.session_id
        ORDER BY s.updated_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def delete_session(session_id: str) -> bool:
    conn = get_db()
    cursor = conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted

def add_message(session_id: str, role: str, content: str):
    now = datetime.now().isoformat()
    conn = get_db()
    conn.execute(
        "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (session_id, role, content, now)
    )
    conn.execute(
        "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
        (now, session_id)
    )
    conn.commit()
    conn.close()

def get_recent_messages(session_id: str, limit: int = 20) -> List[dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
        (session_id, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in reversed(rows)]

def get_message_count(session_id: str) -> int:
    conn = get_db()
    row = conn.execute("SELECT COUNT(*) as cnt FROM messages WHERE session_id = ?", (session_id,)).fetchone()
    conn.close()
    return row["cnt"] if row else 0

def update_session_title(session_id: str, title: str):
    conn = get_db()
    conn.execute("UPDATE sessions SET title = ? WHERE session_id = ?", (title, session_id))
    conn.commit()
    conn.close()

def update_session_summary(session_id: str, summary: str):
    conn = get_db()
    conn.execute("UPDATE sessions SET summary = ? WHERE session_id = ?", (summary, session_id))
    conn.commit()
    conn.close()

def cleanup_old_sessions(max_age_days: int = SESSION_MAX_AGE_DAYS) -> int:
    cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat()
    conn = get_db()
    cursor = conn.execute("DELETE FROM sessions WHERE updated_at < ?", (cutoff,))
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    return deleted

# ============================================================
# 文档加载与分割
# ============================================================

def _load_pptx(file_path):
    """加载 .pptx 文件，每个幻灯片生成一个 Document。"""
    from pptx import Presentation
    prs = Presentation(file_path)
    docs = []
    for slide_num, slide in enumerate(prs.slides, start=1):
        texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    text = paragraph.text.strip()
                    if text:
                        texts.append(text)
        if texts:
            page_content = "\n".join(texts)
            docs.append(Document(
                page_content=page_content,
                metadata={"source": file_path, "page": slide_num}
            ))
    return docs


def _load_docx(file_path):
    """加载 .docx 文件，整个文档生成一个 Document（由下游文本分割器分块）。"""
    from docx import Document as DocxDocument
    doc = DocxDocument(file_path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    if not paragraphs:
        return []
    page_content = "\n\n".join(paragraphs)
    return [Document(
        page_content=page_content,
        metadata={"source": file_path, "page": 1}
    )]


def load_documents(directory_path):
    """加载目录中的文档，支持 PDF、PPTX、DOCX、MD 格式。"""
    LOADER_MAP = {
        ".pdf": lambda fp: PyPDFLoader(fp).load(),
        ".pptx": _load_pptx,
        ".docx": _load_docx,
        ".md": lambda fp: TextLoader(fp, autodetect_encoding=True).load(),
    }

    all_documents = []
    stats = {}

    if not os.path.isdir(directory_path):
        print(f"错误：目录不存在 {directory_path}")
        return []

    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in LOADER_MAP:
                continue

            file_path = os.path.join(root, filename)
            if ext not in stats:
                stats[ext] = {"loaded": 0, "skipped": 0, "errors": []}

            try:
                docs = LOADER_MAP[ext](file_path)
                if docs:
                    all_documents.extend(docs)
                    stats[ext]["loaded"] += 1
                    print(f"  [OK] {ext.upper()}: {filename} -> {len(docs)} 个文档")
                else:
                    stats[ext]["skipped"] += 1
                    print(f"  [SKIP] {ext.upper()}: {filename} (无有效文本)")
            except Exception as e:
                stats[ext]["skipped"] += 1
                stats[ext]["errors"].append(f"{filename}: {e}")
                print(f"  [ERROR] {ext.upper()}: {filename} -> {e}")

    print(f"\n文档加载汇总:")
    print(f"  总计加载: {len(all_documents)} 个文档片段")
    for ext, s in sorted(stats.items()):
        print(f"  {ext.upper()}: 加载={s['loaded']}, 跳过={s['skipped']}")
        for err in s["errors"]:
            print(f"    错误: {err}")

    return all_documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
    )
    texts = text_splitter.split_documents(documents)
    print(f"分割为 {len(texts)} 个文本块")
    return texts

# ============================================================
# 向量库创建与初始化（含 BM25 索引构建）
# ============================================================

def create_vectorstore(texts):
    global embedding_api_key

    if not embedding_api_key:
        print("错误：未设置EMBEDDING_API_KEY环境变量")
        return None

    from langchain_core.documents import Document

    if texts and not isinstance(texts[0], Document):
        print("警告：texts不是Document对象列表，尝试转换")
        new_texts = []
        for i, text in enumerate(texts):
            if isinstance(text, str):
                cleaned_text = str(text).strip()
                cleaned_text = ''.join(c for c in cleaned_text if ord(c) >= 32 or c in '\n\t')
                if cleaned_text:
                    new_texts.append(Document(page_content=cleaned_text, metadata={}))
                else:
                    print(f"跳过空文本元素: {i}")
            else:
                print(f"跳过非字符串元素: {type(text)}")
        texts = new_texts
    else:
        cleaned_texts = []
        for doc in texts:
            if hasattr(doc, 'page_content'):
                cleaned_content = str(doc.page_content).strip()
                cleaned_content = ''.join(c for c in cleaned_content if ord(c) >= 32 or c in '\n\t')
                if cleaned_content:
                    cleaned_texts.append(Document(page_content=cleaned_content, metadata=doc.metadata))
                else:
                    print("跳过空文本Document")
            else:
                print(f"跳过非Document对象: {type(doc)}")
        texts = cleaned_texts

    if not texts:
        print("错误：没有有效文本可处理")
        return None

    print(f"清理后文本块数量: {len(texts)}")

    emb = OpenAICompatibleEmbeddings()
    batch_size = 32
    vs = None

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        print(f"处理第 {i//batch_size + 1} 批文本块，共 {len(batch_texts)} 个")

        valid_docs = []
        for j, doc in enumerate(batch_texts):
            if hasattr(doc, 'page_content') and isinstance(doc.page_content, str) and doc.page_content.strip():
                valid_docs.append(doc)
            else:
                print(f"跳过无效的Document对象: {j}")

        if valid_docs:
            if vs is None:
                vs = FAISS.from_documents(valid_docs, emb)
            else:
                temp_vs = FAISS.from_documents(valid_docs, emb)
                vs.merge_from(temp_vs)

    if vs:
        kb_path = os.path.join(base_dir, "course_knowledge_base")
        vs.save_local(kb_path)
        print(f"向量库已保存到 {kb_path}")

        # 构建 BM25 索引并持久化
        print("正在构建 BM25 索引...")
        tokenized_corpus = [tokenize(doc.page_content) for doc in texts]
        bm25_data = {"tokenized_corpus": tokenized_corpus, "documents": texts}
        bm25_path = os.path.join(kb_path, "bm25_index.pkl")
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25_data, f)
        print(f"BM25 索引已保存到 {bm25_path}")

        return vs
    else:
        print("错误：向量库创建失败")
        return None

def init_vectorstore(force_rebuild=False):
    global vectorstore, retriever, embeddings, llm, reranker
    global bm25_index, bm25_docs
    global embedding_api_key, llm_api_key, llm_base_url, llm_model
    try:
        if not embedding_api_key:
            print("未设置EMBEDDING_API_KEY环境变量")
            return False

        if not llm_api_key:
            print("未设置LLM_API_KEY环境变量")
            return False

        from langchain_openai import ChatOpenAI

        embeddings = OpenAICompatibleEmbeddings()
        llm = ChatOpenAI(
            openai_api_key=llm_api_key,
            openai_api_base=llm_base_url,
            model_name=llm_model,
            temperature=0.3,
            request_timeout=120,
            max_retries=2
        )

        # 初始化重排序器
        reranker = OpenAICompatibleReranker()
        print(f"重排序器已初始化 (模型: {RERANKER_MODEL})")

        kb_path = os.path.join(base_dir, "course_knowledge_base")
        materials_path = os.path.join(base_dir, "course_materials")

        if force_rebuild and os.path.exists(kb_path):
            import shutil
            shutil.rmtree(kb_path)
            print("已删除旧的向量库，准备重新创建")

        if not force_rebuild and os.path.exists(kb_path):
            vectorstore = FAISS.load_local(kb_path, embeddings, allow_dangerous_deserialization=True)
            print("成功加载现有向量库")

            # 加载 BM25 索引
            bm25_path = os.path.join(kb_path, "bm25_index.pkl")
            if os.path.exists(bm25_path):
                with open(bm25_path, "rb") as f:
                    bm25_data = pickle.load(f)
                bm25_index = BM25Okapi(bm25_data["tokenized_corpus"])
                bm25_docs = bm25_data["documents"]
                print(f"BM25 索引已加载 ({len(bm25_docs)} 个文档)")
            else:
                print("警告：BM25 索引文件不存在，混合检索不可用。请使用 /init?force_rebuild=true 重建知识库")
        else:
            try:
                docs = load_documents(materials_path)
                if docs:
                    texts = split_documents(docs)
                    vectorstore = create_vectorstore(texts)
                    # 构建后同时加载 BM25
                    bm25_path = os.path.join(kb_path, "bm25_index.pkl")
                    if os.path.exists(bm25_path):
                        with open(bm25_path, "rb") as f:
                            bm25_data = pickle.load(f)
                        bm25_index = BM25Okapi(bm25_data["tokenized_corpus"])
                        bm25_docs = bm25_data["documents"]
                    print("从文档创建了新的向量库和 BM25 索引")
                else:
                    print("未找到文档，向量库未初始化")
                    return False
            except Exception as e:
                print(f"创建向量库失败: {e}")
                return False

        retriever = vectorstore.as_retriever(search_kwargs={"k": HYBRID_CANDIDATE_K})
        return True
    except Exception as e:
        print(f"初始化向量库失败: {e}")
        return False

# ============================================================
# 预检索优化：查询改写
# ============================================================

def rewrite_query(question: str, session_id: Optional[str] = None) -> str:
    """用 LLM 将口语化问题改写为规范化检索词"""
    history_text = format_history(session_id) if session_id else ""
    if not history_text:
        history_text = "（无对话历史）\n"

    prompt = QUERY_REWRITE_PROMPT.format(history=history_text, question=question)
    try:
        answer = llm.invoke(prompt)
        rewritten = answer.content if hasattr(answer, 'content') else str(answer)
        rewritten = rewritten.strip().strip('"').strip("'")
        print(f"查询改写: '{question}' -> '{rewritten}'")
        return rewritten if rewritten else question
    except Exception as e:
        print(f"查询改写失败: {e}，使用原始问题")
        return question

# ============================================================
# 预检索优化：HyDE
# ============================================================

def hyde_generate(question: str) -> str:
    """让 LLM 生成假设性答案，用于检索"""
    prompt = HYDE_PROMPT.format(question=question)
    try:
        answer = llm.invoke(prompt)
        hypothetical = answer.content if hasattr(answer, 'content') else str(answer)
        print(f"HyDE 假设答案生成成功 (长度: {len(hypothetical)})")
        return hypothetical
    except Exception as e:
        print(f"HyDE 生成失败: {e}，使用原始问题")
        return question

# ============================================================
# 混合检索策略（向量 + BM25 + RRF 融合）
# ============================================================

def hybrid_retrieve(query: str, k: int = DEFAULT_TOP_K) -> list:
    """混合检索：FAISS 向量检索 + BM25 关键词检索，倒数排名融合"""
    candidate_k = HYBRID_CANDIDATE_K

    # 1. 向量检索
    vector_docs = vectorstore.similarity_search(query, k=candidate_k)

    # 2. BM25 检索
    bm25_results = []
    if bm25_index is not None and bm25_docs is not None:
        query_tokens = tokenize(query)
        scores = bm25_index.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:candidate_k]
        bm25_results = [bm25_docs[i] for i in top_indices if scores[i] > 0]

    # 3. 倒数排名融合（Reciprocal Rank Fusion）
    doc_scores = {}
    rrf_k = 60  # RRF 常数

    for rank, doc in enumerate(vector_docs):
        key = doc.page_content[:100]  # 用前100字符作为去重key
        doc_scores[key] = doc_scores.get(key, {"doc": doc, "score": 0})
        doc_scores[key]["score"] += VECTOR_WEIGHT / (rank + rrf_k)

    for rank, doc in enumerate(bm25_results):
        key = doc.page_content[:100]
        doc_scores[key] = doc_scores.get(key, {"doc": doc, "score": 0})
        doc_scores[key]["score"] += BM25_WEIGHT / (rank + rrf_k)

    # 按融合分数排序
    sorted_items = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
    result = []
    for item in sorted_items[:k]:
        doc = item["doc"]
        if hasattr(doc, 'metadata'):
            doc.metadata['rrf_score'] = round(item["score"], 6)
        result.append(doc)
    print(f"混合检索: 向量={len(vector_docs)}篇, BM25={len(bm25_results)}篇, 融合后={len(result)}篇")
    return result

# ============================================================
# RAG 查询函数（带检索优化 + 会话历史）
# ============================================================

def format_history(session_id: str) -> str:
    if not session_id:
        return ""
    messages = get_recent_messages(session_id, limit=MAX_HISTORY_EXCHANGES * 2)
    if not messages:
        return ""
    lines = []
    for msg in messages:
        role_label = "用户" if msg["role"] == "user" else "助手"
        lines.append(f"{role_label}：{msg['content']}")
    return "### 对话历史：\n" + "\n\n".join(lines) + "\n\n"

def extract_sources(docs, top_k=DEFAULT_TOP_K):
    """从检索文档中提取来源信息"""
    sources = []
    for i, doc in enumerate(docs[:top_k]):
        meta = doc.metadata if hasattr(doc, 'metadata') else {}
        source = {
            "rank": i + 1,
            "content": doc.page_content[:500] if hasattr(doc, 'page_content') else str(doc)[:500],
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


def rag_query(question: str, session_id: Optional[str] = None,
              retrieval_strategy: str = "default",
              pre_retrieval: str = "none",
              post_retrieval: str = "none") -> tuple:
    # 1. 预检索优化
    search_query = question
    if pre_retrieval == "rewrite":
        search_query = rewrite_query(question, session_id)
    elif pre_retrieval == "hyde":
        search_query = hyde_generate(question)

    # 2. 检索
    if pre_retrieval == "hyde":
        # HyDE：用假设答案的 embedding 向量检索（带分数）
        hyde_embedding = embeddings.embed_query(search_query)
        scored = vectorstore.similarity_search_with_score_by_vector(hyde_embedding, k=HYBRID_CANDIDATE_K)
        docs = []
        for doc, score in scored:
            if hasattr(doc, 'metadata'):
                doc.metadata['vector_score'] = round(float(score), 4)
            docs.append(doc)
    elif retrieval_strategy == "hybrid":
        docs = hybrid_retrieve(search_query, k=HYBRID_CANDIDATE_K)
    else:
        # 默认向量检索（带分数）
        scored = vectorstore.similarity_search_with_score(search_query, k=HYBRID_CANDIDATE_K)
        docs = []
        for doc, score in scored:
            if hasattr(doc, 'metadata'):
                doc.metadata['vector_score'] = round(float(score), 4)
            docs.append(doc)

    # 3. 后检索优化：重排序（用原始问题，不用改写后的查询词）
    if post_retrieval == "rerank" and reranker:
        docs = reranker.rerank(question, docs, top_n=DEFAULT_TOP_K)

    # 4. 取 top-k 组装 context
    top_docs = docs[:DEFAULT_TOP_K]
    context = "\n\n".join([doc.page_content for doc in top_docs])

    # 5. 提取来源
    sources = extract_sources(docs)

    # 6. 加载历史
    history_text = format_history(session_id)

    # 7. Prompt + LLM
    prompt_text = PROMPT_WITH_HISTORY.format(
        context=context,
        history=history_text,
        question=question
    )
    answer = llm.invoke(prompt_text)
    answer_text = answer.content if hasattr(answer, 'content') else str(answer)
    return answer_text, sources

def rag_query_stateless(question: str,
                        retrieval_strategy: str = "default",
                        pre_retrieval: str = "none",
                        post_retrieval: str = "none") -> tuple:
    """无状态 RAG 查询"""
    search_query = question
    if pre_retrieval == "rewrite":
        search_query = rewrite_query(question)
    elif pre_retrieval == "hyde":
        search_query = hyde_generate(question)

    if pre_retrieval == "hyde":
        hyde_embedding = embeddings.embed_query(search_query)
        scored = vectorstore.similarity_search_with_score_by_vector(hyde_embedding, k=HYBRID_CANDIDATE_K)
        docs = []
        for doc, score in scored:
            if hasattr(doc, 'metadata'):
                doc.metadata['vector_score'] = round(float(score), 4)
            docs.append(doc)
    elif retrieval_strategy == "hybrid":
        docs = hybrid_retrieve(search_query, k=HYBRID_CANDIDATE_K)
    else:
        scored = vectorstore.similarity_search_with_score(search_query, k=HYBRID_CANDIDATE_K)
        docs = []
        for doc, score in scored:
            if hasattr(doc, 'metadata'):
                doc.metadata['vector_score'] = round(float(score), 4)
            docs.append(doc)

    if post_retrieval == "rerank" and reranker:
        docs = reranker.rerank(question, docs, top_n=DEFAULT_TOP_K)

    top_docs = docs[:DEFAULT_TOP_K]
    context = "\n\n".join([doc.page_content for doc in top_docs])
    sources = extract_sources(docs)
    prompt_text = PROMPT.format(context=context, question=question)
    answer = llm.invoke(prompt_text)
    answer_text = answer.content if hasattr(answer, 'content') else str(answer)
    return answer_text, sources

def _retrieve_and_build_prompt(question, session_id=None, retrieval_strategy="default",
                               pre_retrieval="none", post_retrieval="none", use_history=True):
    """公共检索逻辑，返回 (prompt_text, sources)"""
    search_query = question
    if pre_retrieval == "rewrite":
        search_query = rewrite_query(question, session_id) if session_id else rewrite_query(question)
    elif pre_retrieval == "hyde":
        search_query = hyde_generate(question)

    if pre_retrieval == "hyde":
        hyde_embedding = embeddings.embed_query(search_query)
        scored = vectorstore.similarity_search_with_score_by_vector(hyde_embedding, k=HYBRID_CANDIDATE_K)
        docs = []
        for doc, score in scored:
            if hasattr(doc, 'metadata'):
                doc.metadata['vector_score'] = round(float(score), 4)
            docs.append(doc)
    elif retrieval_strategy == "hybrid":
        docs = hybrid_retrieve(search_query, k=HYBRID_CANDIDATE_K)
    else:
        scored = vectorstore.similarity_search_with_score(search_query, k=HYBRID_CANDIDATE_K)
        docs = []
        for doc, score in scored:
            if hasattr(doc, 'metadata'):
                doc.metadata['vector_score'] = round(float(score), 4)
            docs.append(doc)

    if post_retrieval == "rerank" and reranker:
        docs = reranker.rerank(question, docs, top_n=DEFAULT_TOP_K)

    top_docs = docs[:DEFAULT_TOP_K]
    context = "\n\n".join([doc.page_content for doc in top_docs])
    sources = extract_sources(docs)

    if use_history and session_id:
        history_text = format_history(session_id)
        prompt_text = PROMPT_WITH_HISTORY.format(context=context, history=history_text, question=question)
    else:
        prompt_text = PROMPT.format(context=context, question=question)

    return prompt_text, sources

def rag_query_stream(question, session_id=None, retrieval_strategy="default",
                     pre_retrieval="none", post_retrieval="none"):
    """流式 RAG 查询（同步生成器）：先 yield sources JSON，再逐 token yield"""
    prompt_text, sources = _retrieve_and_build_prompt(
        question, session_id, retrieval_strategy, pre_retrieval, post_retrieval, use_history=True
    )
    yield json.dumps({"type": "sources", "data": sources}, ensure_ascii=False)
    for chunk in llm.stream(prompt_text):
        content = chunk.content if hasattr(chunk, 'content') else str(chunk)
        if content:
            yield json.dumps({"type": "token", "data": content}, ensure_ascii=False)

def rag_query_stateless_stream(question, retrieval_strategy="default",
                               pre_retrieval="none", post_retrieval="none"):
    """无状态流式 RAG 查询（同步生成器）"""
    prompt_text, sources = _retrieve_and_build_prompt(
        question, None, retrieval_strategy, pre_retrieval, post_retrieval, use_history=False
    )
    yield json.dumps({"type": "sources", "data": sources}, ensure_ascii=False)
    for chunk in llm.stream(prompt_text):
        content = chunk.content if hasattr(chunk, 'content') else str(chunk)
        if content:
            yield json.dumps({"type": "token", "data": content}, ensure_ascii=False)

def generate_summary_task(session_id: str):
    """后台任务：生成会话摘要"""
    try:
        messages = get_recent_messages(session_id, limit=999)
        if len(messages) < 2:
            return
        conversation = "\n".join(
            f"{'用户' if m['role'] == 'user' else '助手'}：{m['content']}"
            for m in messages
        )
        prompt = SUMMARY_PROMPT_TEMPLATE.format(conversation=conversation)
        answer = llm.invoke(prompt)
        summary = answer.content if hasattr(answer, 'content') else str(answer)
        update_session_summary(session_id, summary)
        print(f"会话 {session_id} 摘要已生成")
    except Exception as e:
        print(f"生成摘要失败: {e}")

# ============================================================
# API 端点
# ============================================================

@app.get("/")
async def root():
    return {"message": "知识库API已启动（检索优化版），支持查询改写/HyDE/混合检索/重排序"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}


def mask_key(key):
    """Mask API key, showing only last 4 characters"""
    if not key or len(key) < 8:
        return "***" if key else ""
    return "***" + key[-4:]


def save_env_file():
    """Save current config to .env file, preserving comments and formatting"""
    env_path = os.path.join(base_dir, ".env")
    updates = {
        "LLM_API_KEY": llm_api_key or "",
        "LLM_BASE_URL": llm_base_url,
        "LLM_MODEL": llm_model,
        "EMBEDDING_API_KEY": embedding_api_key or "",
        "EMBEDDING_BASE_URL": embedding_base_url,
        "EMBEDDING_MODEL": embedding_model_name,
        "RERANKER_MODEL": RERANKER_MODEL,
        "RERANKER_API_KEY": reranker_api_key or "",
        "RERANKER_BASE_URL": reranker_base_url,
    }
    updated_keys = set()

    lines = []
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and "=" in stripped:
                    key = stripped.split("=", 1)[0].strip()
                    if key in updates:
                        lines.append(f"{key}={updates[key]}\n")
                        updated_keys.add(key)
                        continue
                lines.append(line)

    # Append any vars not found in the original file
    for key, val in updates.items():
        if key not in updated_keys:
            lines.append(f"{key}={val}\n")

    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


@app.get("/config")
async def get_config():
    return {
        "llm": {
            "model": llm_model,
            "base_url": llm_base_url,
            "api_key": mask_key(llm_api_key),
        },
        "embedding": {
            "model": embedding_model_name,
            "base_url": embedding_base_url,
            "api_key": mask_key(embedding_api_key),
        },
        "reranker": {
            "model": RERANKER_MODEL,
            "base_url": reranker_base_url,
            "api_key": mask_key(reranker_api_key),
        },
    }


@app.post("/config")
async def update_config(request: dict):
    global llm_model, llm_base_url, llm_api_key
    global embedding_base_url, embedding_api_key, embedding_model_name
    global llm, embeddings, reranker, RERANKER_MODEL
    global reranker_base_url, reranker_api_key

    llm_cfg = request.get("llm", {})
    emb_cfg = request.get("embedding", {})
    rer_cfg = request.get("reranker", {})

    # Update LLM config
    if "model" in llm_cfg and llm_cfg["model"]:
        llm_model = llm_cfg["model"]
    if "base_url" in llm_cfg and llm_cfg["base_url"]:
        llm_base_url = llm_cfg["base_url"]
    if "api_key" in llm_cfg and llm_cfg["api_key"] and not llm_cfg["api_key"].startswith("***"):
        llm_api_key = llm_cfg["api_key"]

    # Update embedding config
    if "base_url" in emb_cfg and emb_cfg["base_url"]:
        embedding_base_url = emb_cfg["base_url"]
    if "api_key" in emb_cfg and emb_cfg["api_key"] and not emb_cfg["api_key"].startswith("***"):
        embedding_api_key = emb_cfg["api_key"]
    if "model" in emb_cfg and emb_cfg["model"]:
        embedding_model_name = emb_cfg["model"]

    # Update reranker config
    if "model" in rer_cfg and rer_cfg["model"]:
        RERANKER_MODEL = rer_cfg["model"]
    if "base_url" in rer_cfg and rer_cfg["base_url"]:
        reranker_base_url = rer_cfg["base_url"]
    if "api_key" in rer_cfg and rer_cfg["api_key"] and not rer_cfg["api_key"].startswith("***"):
        reranker_api_key = rer_cfg["api_key"]

    # Reinitialize components
    errors = []
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            openai_api_key=llm_api_key,
            openai_api_base=llm_base_url,
            model_name=llm_model,
            temperature=0.3,
            request_timeout=120,
            max_retries=2,
        )
    except Exception as e:
        errors.append(f"LLM 初始化失败: {e}")

    try:
        embeddings = OpenAICompatibleEmbeddings()
    except Exception as e:
        errors.append(f"Embedding 初始化失败: {e}")

    try:
        reranker = OpenAICompatibleReranker(model=RERANKER_MODEL)
    except Exception as e:
        errors.append(f"Reranker 初始化失败: {e}")

    # Persist to .env
    try:
        save_env_file()
    except Exception as e:
        errors.append(f"保存 .env 失败: {e}")

    if errors:
        return {"status": "partial", "message": "; ".join(errors)}

    return {"status": "success", "message": "配置已更新并保存"}

@app.post("/init")
async def init_knowledge_base(force_rebuild: bool = False):
    try:
        global embedding_api_key
        if not embedding_api_key:
            return {"status": "error", "message": "未设置EMBEDDING_API_KEY环境变量"}

        success = init_vectorstore(force_rebuild=force_rebuild)
        if success:
            message = "知识库重建成功（含BM25索引）" if force_rebuild else "知识库初始化成功"
            return {"status": "success", "message": message}
        else:
            return {"status": "error", "message": "知识库初始化失败，请确保course_materials文件夹中有支持的文件（PDF/PPTX/DOCX/MD）"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

SUPPORTED_EXTENSIONS = {".pdf", ".pptx", ".docx", ".md"}

@app.get("/materials")
async def list_materials():
    """列出 course_materials 目录中的文档文件。"""
    try:
        materials_path = os.path.join(base_dir, "course_materials")
        if not os.path.isdir(materials_path):
            return {"status": "success", "files": []}

        files = []
        for name in os.listdir(materials_path):
            ext = os.path.splitext(name)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue
            fp = os.path.join(materials_path, name)
            stat = os.stat(fp)
            files.append({
                "name": name,
                "size": stat.st_size,
                "modified": stat.st_mtime,
            })
        files.sort(key=lambda f: f["modified"], reverse=True)
        return {"status": "success", "files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/materials/upload")
async def upload_material(files: List[UploadFile] = File(...)):
    """上传文档到 course_materials 目录。"""
    try:
        materials_path = os.path.join(base_dir, "course_materials")
        os.makedirs(materials_path, exist_ok=True)

        uploaded = []
        errors = []
        for upload_file in files:
            safe_name = os.path.basename(upload_file.filename)
            if not safe_name:
                errors.append(f"{upload_file.filename}: 无效文件名")
                continue
            ext = os.path.splitext(safe_name)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                errors.append(f"{safe_name}: 不支持的文件格式（仅支持 PDF/PPTX/DOCX/MD）")
                continue
            dest = os.path.join(materials_path, safe_name)
            content = await upload_file.read()
            with open(dest, "wb") as f:
                f.write(content)
            uploaded.append(safe_name)

        if errors:
            return {"status": "partial" if uploaded else "error", "uploaded": uploaded, "errors": errors}
        return {"status": "success", "uploaded": uploaded}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/materials/{filename}")
async def delete_material(filename: str):
    """删除 course_materials 目录中的指定文件。"""
    try:
        filename = os.path.normpath(filename)
        if os.path.dirname(filename) != '':
            raise HTTPException(status_code=400, detail="非法文件名")

        materials_path = os.path.join(base_dir, "course_materials")
        fp = os.path.join(materials_path, filename)
        if not os.path.realpath(fp).startswith(os.path.realpath(materials_path) + os.sep):
            raise HTTPException(status_code=400, detail="非法文件名")
        if not os.path.isfile(fp):
            raise HTTPException(status_code=404, detail=f"文件不存在: {filename}")

        os.remove(fp)
        return {"status": "success", "message": f"已删除: {filename}"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/stream")
async def ask_stream(query: Query, background_tasks: BackgroundTasks):
    """流式问答接口，返回 SSE 事件流"""
    global embedding_api_key, llm_api_key
    if not llm_api_key:
        raise HTTPException(status_code=503, detail="未设置LLM_API_KEY环境变量")
    if not embedding_api_key:
        raise HTTPException(status_code=503, detail="未设置EMBEDDING_API_KEY环境变量")

    if not retriever:
        if not init_vectorstore():
            raise HTTPException(status_code=503, detail="向量库未初始化，请先上传文档")

    session_id = query.session_id
    if session_id:
        session = get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")

    def event_generator():
        full_answer = []
        try:
            gen = rag_query_stream if session_id else rag_query_stateless_stream
            kwargs = {
                "question": query.question,
                "retrieval_strategy": query.retrieval_strategy,
                "pre_retrieval": query.pre_retrieval,
                "post_retrieval": query.post_retrieval,
            }
            if session_id:
                kwargs["session_id"] = session_id

            for event_str in gen(**kwargs):
                event = json.loads(event_str)
                if event["type"] == "token":
                    full_answer.append(event["data"])
                yield f"data: {event_str}\n\n"

            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)}, ensure_ascii=False)}\n\n"

        # 流式完成后保存消息
        answer_text = "".join(full_answer)
        if session_id and answer_text:
            add_message(session_id, "user", query.question)
            add_message(session_id, "assistant", answer_text)
            msg_count = get_message_count(session_id)
            if msg_count == 2:
                title = query.question[:30] + ("..." if len(query.question) > 30 else "")
                update_session_title(session_id, title)
            if msg_count > 0 and msg_count % SUMMARY_TRIGGER_COUNT == 0:
                background_tasks.add_task(generate_summary_task, session_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/ask", response_model=Response)
async def ask_question(query: Query, background_tasks: BackgroundTasks):
    try:
        global embedding_api_key, llm_api_key
        if not llm_api_key:
            raise HTTPException(status_code=503, detail="未设置LLM_API_KEY环境变量")
        if not embedding_api_key:
            raise HTTPException(status_code=503, detail="未设置EMBEDDING_API_KEY环境变量")

        if not retriever:
            print("retriever未初始化，尝试初始化...")
            if not init_vectorstore():
                raise HTTPException(status_code=503, detail="向量库未初始化，请先上传文档")

        session_id = query.session_id
        if session_id:
            session = get_session(session_id)
            if not session:
                raise HTTPException(status_code=404, detail="会话不存在")

        strategy_info = f"retrieval={query.retrieval_strategy}, pre={query.pre_retrieval}, post={query.post_retrieval}"
        print(f"收到问题: {query.question} (session_id={session_id}, {strategy_info})")

        try:
            if session_id:
                answer, sources = await asyncio.wait_for(
                    asyncio.to_thread(
                        rag_query, query.question, session_id,
                        query.retrieval_strategy, query.pre_retrieval, query.post_retrieval
                    ),
                    timeout=180
                )
            else:
                answer, sources = await asyncio.wait_for(
                    asyncio.to_thread(
                        rag_query_stateless, query.question,
                        query.retrieval_strategy, query.pre_retrieval, query.post_retrieval
                    ),
                    timeout=180
                )
        except asyncio.TimeoutError:
            print("LLM调用超时(180秒)")
            raise HTTPException(status_code=504, detail="LLM调用超时，请检查模型配置或网络连接")

        print(f"LLM返回结果，长度: {len(answer)}")

        if session_id:
            add_message(session_id, "user", query.question)
            add_message(session_id, "assistant", answer)

            msg_count = get_message_count(session_id)
            if msg_count == 2:
                title = query.question[:30] + ("..." if len(query.question) > 30 else "")
                update_session_title(session_id, title)

            if msg_count > 0 and msg_count % SUMMARY_TRIGGER_COUNT == 0:
                background_tasks.add_task(generate_summary_task, session_id)

        return Response(answer=answer, sources=sources, session_id=session_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    return OpenAIModelsResponse(
        data=[OpenAIModel(id="knowledge-base")]
    )

@app.post("/v1/chat/completions")
async def chat_completions(request: OpenAIChatRequest, background_tasks: BackgroundTasks):
    global embedding_api_key, llm_api_key
    if not llm_api_key:
        raise HTTPException(status_code=503, detail="未设置LLM_API_KEY环境变量")
    if not embedding_api_key:
        raise HTTPException(status_code=503, detail="未设置EMBEDDING_API_KEY环境变量")

    if not retriever:
        if not init_vectorstore():
            raise HTTPException(status_code=503, detail="向量库未初始化，请先上传文档")

    session_id = request.session_id

    user_message = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_message = msg.content
            break

    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found")

    if session_id:
        session = get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")

    print(f"[chat] 收到问题: {user_message} (session_id={session_id}, stream={request.stream})")

    # 流式输出
    if request.stream:
        import time as _time
        chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        def openai_stream_generator():
            full_answer = []
            try:
                gen = rag_query_stream if session_id else rag_query_stateless_stream
                kwargs = {
                    "question": user_message,
                    "retrieval_strategy": request.retrieval_strategy,
                    "pre_retrieval": request.pre_retrieval,
                    "post_retrieval": request.post_retrieval,
                }
                if session_id:
                    kwargs["session_id"] = session_id

                for event_str in gen(**kwargs):
                    event = json.loads(event_str)
                    if event["type"] == "token":
                        full_answer.append(event["data"])
                        chunk = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": int(_time.time()),
                            "model": request.model,
                            "choices": [{"index": 0, "delta": {"content": event["data"]}, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

                # 结束 chunk
                chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": int(_time.time()),
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                error_chunk = {"error": {"message": str(e), "type": "server_error"}}
                yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"

            # 保存消息
            answer_text = "".join(full_answer)
            if session_id and answer_text:
                add_message(session_id, "user", user_message)
                add_message(session_id, "assistant", answer_text)
                msg_count = get_message_count(session_id)
                if msg_count == 2:
                    title = user_message[:30] + ("..." if len(user_message) > 30 else "")
                    update_session_title(session_id, title)
                if msg_count > 0 and msg_count % SUMMARY_TRIGGER_COUNT == 0:
                    background_tasks.add_task(generate_summary_task, session_id)

        return StreamingResponse(
            openai_stream_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # 非流式输出
    try:
        if session_id:
            answer, _sources = await asyncio.wait_for(
                asyncio.to_thread(
                    rag_query, user_message, session_id,
                    request.retrieval_strategy, request.pre_retrieval, request.post_retrieval
                ),
                timeout=180
            )
        else:
            if len(request.messages) > 1:
                history_lines = []
                for msg in request.messages[:-1]:
                    role_label = "用户" if msg.role == "user" else "助手"
                    history_lines.append(f"{role_label}：{msg.content}")
                history_text = "### 对话历史：\n" + "\n\n".join(history_lines) + "\n\n"

                search_query = user_message
                if request.pre_retrieval == "rewrite":
                    search_query = rewrite_query(user_message)
                elif request.pre_retrieval == "hyde":
                    search_query = hyde_generate(user_message)

                if request.pre_retrieval == "hyde":
                    hyde_embedding = embeddings.embed_query(search_query)
                    docs = vectorstore.similarity_search_by_vector(hyde_embedding, k=HYBRID_CANDIDATE_K)
                elif request.retrieval_strategy == "hybrid":
                    docs = hybrid_retrieve(search_query, k=HYBRID_CANDIDATE_K)
                else:
                    docs = retriever.invoke(search_query)

                if request.post_retrieval == "rerank" and reranker:
                    docs = reranker.rerank(user_message, docs, top_n=DEFAULT_TOP_K)

                context = "\n\n".join([doc.page_content for doc in docs[:DEFAULT_TOP_K]])
                prompt_text = PROMPT_WITH_HISTORY.format(
                    context=context, history=history_text, question=user_message
                )
                answer_obj = await asyncio.wait_for(
                    asyncio.to_thread(llm.invoke, prompt_text),
                    timeout=180
                )
                answer = answer_obj.content if hasattr(answer_obj, 'content') else str(answer_obj)
            else:
                answer, _sources = await asyncio.wait_for(
                    asyncio.to_thread(
                        rag_query_stateless, user_message,
                        request.retrieval_strategy, request.pre_retrieval, request.post_retrieval
                    ),
                    timeout=180
                )
    except asyncio.TimeoutError:
        print("[chat] LLM调用超时(180秒)")
        raise HTTPException(status_code=504, detail="LLM调用超时，请检查模型配置或网络连接")

    if session_id:
        add_message(session_id, "user", user_message)
        add_message(session_id, "assistant", answer)
        msg_count = get_message_count(session_id)
        if msg_count == 2:
            title = user_message[:30] + ("..." if len(user_message) > 30 else "")
            update_session_title(session_id, title)
        if msg_count > 0 and msg_count % SUMMARY_TRIGGER_COUNT == 0:
            background_tasks.add_task(generate_summary_task, session_id)

    return OpenAIChatResponse(
        model=request.model,
        choices=[
            OpenAIChatChoice(
                message=OpenAIMessage(role="assistant", content=answer)
            )
        ]
    )

# ============================================================
# 会话管理端点
# ============================================================

@app.post("/sessions", response_model=SessionInfo)
async def create_new_session(body: SessionCreate = None):
    title = body.title if body and body.title else "新对话"
    session_id = str(uuid.uuid4())
    session = create_session(session_id, title)
    return SessionInfo(
        session_id=session["session_id"],
        title=session["title"],
        created_at=session["created_at"],
        updated_at=session["updated_at"],
        message_count=0
    )

@app.get("/sessions", response_model=SessionListResponse)
async def get_all_sessions():
    sessions = list_sessions()
    return SessionListResponse(
        sessions=[
            SessionInfo(
                session_id=s["session_id"],
                title=s["title"],
                created_at=s["created_at"],
                updated_at=s["updated_at"],
                summary=s.get("summary"),
                message_count=s.get("message_count", 0)
            )
            for s in sessions
        ]
    )

@app.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session_info(session_id: str):
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    msg_count = get_message_count(session_id)
    return SessionInfo(
        session_id=session["session_id"],
        title=session["title"],
        created_at=session["created_at"],
        updated_at=session["updated_at"],
        summary=session.get("summary"),
        message_count=msg_count
    )

@app.delete("/sessions/{session_id}")
async def delete_session_endpoint(session_id: str):
    deleted = delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"status": "deleted", "session_id": session_id}

@app.get("/sessions/{session_id}/messages", response_model=SessionHistoryResponse)
async def get_session_messages(session_id: str):
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    messages = get_recent_messages(session_id, limit=999)
    return SessionHistoryResponse(
        session_id=session_id,
        messages=[
            MessageInfo(role=m["role"], content=m["content"], created_at=m["created_at"])
            for m in messages
        ]
    )

@app.post("/sessions/cleanup")
async def cleanup_sessions_endpoint(max_age_days: int = SESSION_MAX_AGE_DAYS):
    deleted = cleanup_old_sessions(max_age_days)
    return {"deleted_sessions": deleted, "max_age_days": max_age_days}

# ============================================================
# 启动
# ============================================================

init_db()
init_vectorstore()

if __name__ == "__main__":
    print("=" * 50)
    print("启动知识库API（检索优化版）...")
    print(f"LLM_API_KEY: {'已设置' if llm_api_key else '未设置'}")
    print(f"LLM_BASE_URL: {llm_base_url}")
    print(f"LLM_MODEL: {llm_model}")
    print(f"EMBEDDING_API_KEY: {'已设置' if embedding_api_key else '未设置'}")
    print(f"EMBEDDING_BASE_URL: {embedding_base_url}")
    print(f"EMBEDDING_MODEL: {embedding_model_name}")
    print(f"RERANKER_MODEL: {RERANKER_MODEL}")
    print(f"RERANKER_API_KEY: {'已设置' if reranker_api_key else '未设置'}")
    print(f"RERANKER_BASE_URL: {reranker_base_url}")
    print(f"数据库路径: {DB_PATH}")
    print(f"最大历史轮数: {MAX_HISTORY_EXCHANGES}")
    print(f"会话过期天数: {SESSION_MAX_AGE_DAYS}")
    print(f"混合检索权重: 向量={VECTOR_WEIGHT}, BM25={BM25_WEIGHT}")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
