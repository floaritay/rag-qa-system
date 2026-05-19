from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, timedelta
import uuid
import time
import asyncio
import sqlite3
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
import os
import uvicorn
import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

app = FastAPI(title="课程助手API（会话记忆版）")

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
MAX_HISTORY_EXCHANGES = 10      # 发给 LLM 的最近对话轮数
SESSION_MAX_AGE_DAYS = 7        # 会话过期天数
SUMMARY_TRIGGER_COUNT = 10      # 每 N 条消息触发自动摘要

# ============================================================
# Pydantic 模型
# ============================================================

class Query(BaseModel):
    question: str
    session_id: Optional[str] = None

class Response(BaseModel):
    answer: str
    sources: list = []
    session_id: Optional[str] = None

class OpenAIModel(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "course-assistant"

class OpenAIModelsResponse(BaseModel):
    object: str = "list"
    data: List[OpenAIModel]

class OpenAIMessage(BaseModel):
    role: str
    content: str

class OpenAIChatRequest(BaseModel):
    model: str = "course-assistant"
    messages: List[OpenAIMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    session_id: Optional[str] = None

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
    model: str = "course-assistant"
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

prompt_template = """你是一个专业的课程助教。你的核心职责是基于提供的参考资料回答学生问题。

### 回答原则：
1. 严禁编造课程资料中的信息。
2. 无论问题是否与参考资料相关，你都必须给出有益的回复，但同时必须严格区分并标注信息的来源。

### 来源标注规范（必须严格执行）：
- **来自知识库**：陈述参考资料中的内容时，必须在对应句子或段落末尾标注，如 [来源:xx文件xx节]。
- **超出知识库**：如果问题无法从参考资料中找到答案，你必须先明确声明“该问题未在课程知识库中找到相关资料”，然后可以调用你的通用知识进行补充解答，并在补充内容后标注 [来源:通用知识]。
- **混合情况**：如果回答中既有参考资料的内容，又有你补充的通用知识，必须分别标注，绝不能混淆。

### 回答格式要求：
1. 语言简洁，逻辑清晰，使用列表或分段提升可读性。
2. 如果问题完全与课程无关，在提供通用解答后，可礼貌提醒该问题偏离了当前课程。

---
参考资料：
{context}
---
学生问题：
{question}
---
### 回答"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

prompt_template_with_history = """你是一个专业的课程助教。你的核心职责是基于提供的参考资料回答学生问题，并结合对话上下文理解意图。

### 回答原则：
1. 严禁编造课程资料中的信息。
2. 无论问题是否与参考资料相关，你都必须给出有益的回复，但同时必须严格区分并标注信息的来源。

### 来源标注规范（必须严格执行）：
- **来自知识库**：陈述参考资料中的内容时，必须在对应句子或段落末尾标注，如 [来源:xx文件xx节]。
- **超出知识库**：如果问题无法从参考资料中找到答案，你必须先明确声明“该问题未在课程知识库中找到相关资料”，然后可以调用你的通用知识进行补充解答，并在补充内容后标注 [来源:通用知识]。
- **混合情况**：如果回答中既有参考资料的内容，又有你补充的通用知识，必须分别标注，绝不能混淆。

### 回答格式要求：
1. 语言简洁，逻辑清晰，使用列表或分段提升可读性。
2. 结合对话上下文，准确理解代词和省略的指代对象（如"它"、"这个方法"、"请详细解释"等）。
3. 如果问题完全与课程无关，在提供通用解答后，可礼貌提醒该问题偏离了当前课程。

---
参考资料：
{context}
---
历史对话：
{history}
---
学生问题：
{question}
---
### 回答："""

PROMPT_WITH_HISTORY = PromptTemplate(
    template=prompt_template_with_history,
    input_variables=["context", "history", "question"]
)

SUMMARY_PROMPT_TEMPLATE = "请用2-3句话总结以下对话的主要内容：\n\n{conversation}\n\n总结："

# ============================================================
# 全局变量
# ============================================================
vectorstore = None
retriever = None
embeddings = None
llm = None

# 硅基流动配置
siliconflow_base_url = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
siliconflow_api_key = os.getenv("SILICONFLOW_API_KEY")
siliconflow_embedding_model = "BAAI/bge-m3"

# 百炼平台配置（仅用于LLM）
bailian_base_url = os.getenv("BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
bailian_api_key = os.getenv("BAILIAN_API_KEY")
bailian_model = "qwen3.5-122b-a10b"

# 数据库路径
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(base_dir, "backend", "sessions.db")

# ============================================================
# SiliconFlowEmbeddings 类
# ============================================================

class SiliconFlowEmbeddings(Embeddings):
    """硅基流动嵌入模型，使用BAAI/bge-m3"""
    def __init__(self, api_key=None, base_url=None, model=None):
        self.api_key = api_key or siliconflow_api_key
        self.base_url = base_url or siliconflow_base_url
        self.model = model or siliconflow_embedding_model

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
            print(f"调用硅基流动Embedding API失败: {e}")
            print(f"响应内容: {response.text if 'response' in locals() else '无响应'}")
            return None

    def embed_documents(self, texts):
        result = self._get_embeddings(texts)
        return result if result else []

    def embed_query(self, text):
        result = self._get_embeddings([text])
        return result[0] if result else []

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

def load_documents(directory_path):
    loader = DirectoryLoader(
        directory_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    print(f"加载了 {len(documents)} 个PDF文件")
    return documents

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
# 向量库创建与初始化
# ============================================================

def create_vectorstore(texts):
    global siliconflow_api_key

    if not siliconflow_api_key:
        print("错误：未设置SILICONFLOW_API_KEY环境变量")
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

    emb = SiliconFlowEmbeddings()
    batch_size = 32
    vectorstore = None

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
            if vectorstore is None:
                vectorstore = FAISS.from_documents(valid_docs, emb)
            else:
                temp_vectorstore = FAISS.from_documents(valid_docs, emb)
                vectorstore.merge_from(temp_vectorstore)

    if vectorstore:
        kb_path = os.path.join(base_dir, "course_knowledge_base")
        vectorstore.save_local(kb_path)
        print(f"知识库已保存到 {kb_path} 文件夹")
        return vectorstore
    else:
        print("错误：向量库创建失败")
        return None

def init_vectorstore(force_rebuild=False):
    global vectorstore, retriever, embeddings, llm
    global siliconflow_api_key, bailian_api_key, bailian_base_url, bailian_model
    try:
        if not siliconflow_api_key:
            print("未设置SILICONFLOW_API_KEY环境变量")
            return False

        from langchain_openai import ChatOpenAI

        embeddings = SiliconFlowEmbeddings()
        llm = ChatOpenAI(
            openai_api_key=bailian_api_key,
            openai_api_base=bailian_base_url,
            model_name=bailian_model,
            temperature=0.3,
            request_timeout=120,
            max_retries=2
        )

        kb_path = os.path.join(base_dir, "course_knowledge_base")
        materials_path = os.path.join(base_dir, "course_materials")

        if force_rebuild and os.path.exists(kb_path):
            import shutil
            shutil.rmtree(kb_path)
            print("已删除旧的向量库，准备重新创建")

        if not force_rebuild and os.path.exists(kb_path):
            vectorstore = FAISS.load_local(kb_path, embeddings, allow_dangerous_deserialization=True)
            print("成功加载现有向量库")
        else:
            try:
                docs = load_documents(materials_path)
                if docs:
                    texts = split_documents(docs)
                    vectorstore = create_vectorstore(texts)
                    print("从课程资料创建了新的向量库")
                else:
                    print("未找到课程资料，向量库未初始化")
                    return False
            except Exception as e:
                print(f"创建向量库失败: {e}")
                return False

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        return True
    except Exception as e:
        print(f"初始化向量库失败: {e}")
        return False

# ============================================================
# RAG 查询函数（带会话历史）
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

def rag_query(question: str, session_id: Optional[str] = None) -> str:
    # 1. 检索相关文档
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    # 2. 构建历史文本
    history_text = format_history(session_id)

    # 3. 格式化 prompt 并调用 LLM
    prompt_text = PROMPT_WITH_HISTORY.format(
        context=context,
        history=history_text,
        question=question
    )
    answer = llm.invoke(prompt_text)
    return answer.content if hasattr(answer, 'content') else str(answer)

def rag_query_stateless(question: str) -> str:
    """无状态 RAG 查询，兼容不带 session_id 的调用"""
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt_text = PROMPT.format(context=context, question=question)
    answer = llm.invoke(prompt_text)
    return answer.content if hasattr(answer, 'content') else str(answer)

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
    return {"message": "课程助手API已启动（会话记忆版，硅基流动bge-m3嵌入模型），访问 /ask 接口进行问答，/init 接口初始化知识库"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/init")
async def init_knowledge_base(force_rebuild: bool = False):
    try:
        global siliconflow_api_key
        if not siliconflow_api_key:
            return {"status": "error", "message": "未设置SILICONFLOW_API_KEY环境变量"}

        success = init_vectorstore(force_rebuild=force_rebuild)
        if success:
            message = "知识库重建成功" if force_rebuild else "知识库初始化成功"
            return {"status": "success", "message": message}
        else:
            return {"status": "error", "message": "知识库初始化失败，请确保course_materials文件夹中有PDF文件"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=Response)
async def ask_question(query: Query, background_tasks: BackgroundTasks):
    try:
        global siliconflow_api_key
        if not siliconflow_api_key:
            raise HTTPException(status_code=503, detail="未设置SILICONFLOW_API_KEY环境变量")

        if not retriever:
            print("retriever未初始化，尝试初始化...")
            if not init_vectorstore():
                raise HTTPException(status_code=503, detail="向量库未初始化，请先上传课程资料")

        session_id = query.session_id

        # 验证会话存在
        if session_id:
            session = get_session(session_id)
            if not session:
                raise HTTPException(status_code=404, detail="会话不存在")

        print(f"收到问题: {query.question} (session_id={session_id})")

        # 调用 RAG 查询
        try:
            if session_id:
                answer = await asyncio.wait_for(
                    asyncio.to_thread(rag_query, query.question, session_id),
                    timeout=120
                )
            else:
                answer = await asyncio.wait_for(
                    asyncio.to_thread(rag_query_stateless, query.question),
                    timeout=120
                )
        except asyncio.TimeoutError:
            print("LLM调用超时(120秒)")
            raise HTTPException(status_code=504, detail="LLM调用超时，请检查模型配置或网络连接")

        print(f"LLM返回结果，长度: {len(answer)}")

        # 持久化消息
        if session_id:
            add_message(session_id, "user", query.question)
            add_message(session_id, "assistant", answer)

            # 自动标题：第一条消息后用前30字
            msg_count = get_message_count(session_id)
            if msg_count == 2:
                title = query.question[:30] + ("..." if len(query.question) > 30 else "")
                update_session_title(session_id, title)

            # 自动摘要：每 SUMMARY_TRIGGER_COUNT 条消息触发
            if msg_count > 0 and msg_count % SUMMARY_TRIGGER_COUNT == 0:
                background_tasks.add_task(generate_summary_task, session_id)

        return Response(answer=answer, sources=[], session_id=session_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    return OpenAIModelsResponse(
        data=[OpenAIModel(id="course-assistant")]
    )

@app.post("/v1/chat/completions")
async def chat_completions(request: OpenAIChatRequest, background_tasks: BackgroundTasks):
    try:
        global siliconflow_api_key
        if not siliconflow_api_key:
            raise HTTPException(status_code=503, detail="未设置SILICONFLOW_API_KEY环境变量")

        if not retriever:
            if not init_vectorstore():
                raise HTTPException(status_code=503, detail="向量库未初始化，请先上传课程资料")

        session_id = request.session_id

        # 提取最新用户消息
        user_message = ""
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_message = msg.content
                break

        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")

        # 验证会话存在
        if session_id:
            session = get_session(session_id)
            if not session:
                raise HTTPException(status_code=404, detail="会话不存在")

        print(f"[chat] 收到问题: {user_message} (session_id={session_id})")

        # 调用 RAG 查询
        try:
            if session_id:
                answer = await asyncio.wait_for(
                    asyncio.to_thread(rag_query, user_message, session_id),
                    timeout=120
                )
            else:
                # 无 session_id 但有历史消息时，用消息列表构建历史
                if len(request.messages) > 1:
                    history_lines = []
                    for msg in request.messages[:-1]:
                        role_label = "用户" if msg.role == "user" else "助手"
                        history_lines.append(f"{role_label}：{msg.content}")
                    history_text = "### 对话历史：\n" + "\n\n".join(history_lines) + "\n\n"
                    docs = retriever.invoke(user_message)
                    context = "\n\n".join([doc.page_content for doc in docs])
                    prompt_text = PROMPT_WITH_HISTORY.format(
                        context=context, history=history_text, question=user_message
                    )
                    answer_obj = await asyncio.wait_for(
                        asyncio.to_thread(llm.invoke, prompt_text),
                        timeout=120
                    )
                    answer = answer_obj.content if hasattr(answer_obj, 'content') else str(answer_obj)
                else:
                    answer = await asyncio.wait_for(
                        asyncio.to_thread(rag_query_stateless, user_message),
                        timeout=120
                    )
        except asyncio.TimeoutError:
            print("[chat] LLM调用超时(120秒)")
            raise HTTPException(status_code=504, detail="LLM调用超时，请检查模型配置或网络连接")

        # 持久化消息
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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
    print("启动课程助手API（会话记忆版）...")
    print(f"SILICONFLOW_API_KEY: {'已设置' if siliconflow_api_key else '未设置'}")
    print(f"BAILIAN_API_KEY: {'已设置' if bailian_api_key else '未设置'}")
    print(f"BAILIAN_BASE_URL: {bailian_base_url}")
    print(f"BAILIAN_MODEL: {bailian_model}")
    print(f"SILICONFLOW_BASE_URL: {siliconflow_base_url}")
    print(f"SILICONFLOW_MODEL: {siliconflow_embedding_model}")
    print(f"数据库路径: {DB_PATH}")
    print(f"最大历史轮数: {MAX_HISTORY_EXCHANGES}")
    print(f"会话过期天数: {SESSION_MAX_AGE_DAYS}")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
