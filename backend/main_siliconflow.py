from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import uuid
import time
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

app = FastAPI(title="课程助手API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

class Response(BaseModel):
    answer: str
    sources: list = []

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

prompt_template = """你是一个专业的课程助教，请基于以下参考资料回答学生的问题。禁止编造任何信息。
如果参考资料中包含与问题相关的内容，哪怕只有部分相关，也要详细回答；只有参考资料完全无相关内容时，才能直接回答"在提供的课程资料中找不到相关信息"。

参考资料：
{context}

学生问题：
{question}

### 回答要求：
1. 回答必须基于参考资料，标注关键信息来源；
2. 语言简洁，逻辑清晰"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# 全局变量
vectorstore = None
retriever = None
rag_chain = None
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
            response = requests.post(url, headers=headers, json=data, timeout=60)
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
    '''
    语义分割：按段落、句子、标点符号等进行分割，避免截断句子或词语。

    不直接把文本按500字符切断，而是按照 separators 列表的顺序，从大到小尝试分割：
    1. 尝试用第一个分隔符 "\n\n"（双换行，即段落分隔）把文本切开。
    2. 判断：如果切出来的某一段长度依然 > 500，说明这一段太长了。
    3. 针对那段超长的文本，降级使用第二个分隔符 "\n"（单换行，即行分隔）再次切开。
    4. 判断：如果切出来还是 > 500，就继续降级，用 "。"（句号）切……
    最终兜底：一路降级，直到用完所有标点。如果还有超长文本，最后用 ""（空字符串）强制按单字符切断，保证绝对不超过 chunk_size。
'''
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
    )
    texts = text_splitter.split_documents(documents)
    print(f"分割为 {len(texts)} 个文本块")
    return texts

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
    if texts:
        print(f"第一个文本块类型: {type(texts[0])}")
        if hasattr(texts[0], 'page_content'):
            print(f"第一个文本块长度: {len(texts[0].page_content)}")
            sample = texts[0].page_content[:50] + "..." if len(texts[0].page_content) > 50 else texts[0].page_content
            print(f"第一个文本块示例: {sample}")

    emb = SiliconFlowEmbeddings()

    # bge-m3支持较大batch，使用32作为安全批次大小
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

        print(f"有效Document数量: {len(valid_docs)}")

        if valid_docs:
            if vectorstore is None:
                vectorstore = FAISS.from_documents(valid_docs, emb)
            else:
                temp_vectorstore = FAISS.from_documents(valid_docs, emb)
                vectorstore.merge_from(temp_vectorstore)

    if vectorstore:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        kb_path = os.path.join(base_dir, "course_knowledge_base")
        vectorstore.save_local(kb_path)
        print(f"知识库已保存到 {kb_path} 文件夹")
        return vectorstore
    else:
        print("错误：向量库创建失败")
        return None

def init_vectorstore(force_rebuild=False):
    global vectorstore, retriever, rag_chain, embeddings, llm
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
            temperature=0.3
        )

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | PROMPT
            | llm
            | StrOutputParser()
        )
        return True
    except Exception as e:
        print(f"初始化向量库失败: {e}")
        return False

init_vectorstore()

@app.post("/ask", response_model=Response)
async def ask_question(query: Query):
    try:
        global siliconflow_api_key
        if not siliconflow_api_key:
            raise HTTPException(status_code=503, detail="未设置SILICONFLOW_API_KEY环境变量")

        if not rag_chain:
            if not init_vectorstore():
                raise HTTPException(status_code=503, detail="向量库未初始化，请先上传课程资料")

        answer = rag_chain.invoke(query.question)
        sources = []
        return Response(answer=answer, sources=sources)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "课程助手API已启动（硅基流动bge-m3嵌入模型），访问 /ask 接口进行问答，/init 接口初始化知识库"}

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

@app.get("/v1/models")
async def list_models():
    return OpenAIModelsResponse(
        data=[OpenAIModel(id="course-assistant")]
    )

@app.post("/v1/chat/completions")
async def chat_completions(request: OpenAIChatRequest):
    try:
        global siliconflow_api_key
        if not siliconflow_api_key:
            raise HTTPException(status_code=503, detail="未设置SILICONFLOW_API_KEY环境变量")

        if not rag_chain:
            if not init_vectorstore():
                raise HTTPException(status_code=503, detail="向量库未初始化，请先上传课程资料")

        user_message = ""
        for msg in request.messages:
            if msg.role == "user":
                user_message = msg.content
                break

        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")

        answer = rag_chain.invoke(user_message)

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
