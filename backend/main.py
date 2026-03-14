from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
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

app = FastAPI(title="课程助手API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求体模型
class Query(BaseModel):
    question: str

# 响应体模型
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

# 自定义Prompt模板
prompt_template = """你是一个专业的课程助教，请基于以下参考资料回答学生的问题。禁止编造任何信息。
如果参考资料中包含与问题相关的内容，哪怕只有部分相关，也要详细回答；只有参考资料完全无相关内容时，才能直接回答“在提供的课程资料中找不到相关信息”。

参考资料：
{context}

学生问题：
{question}

### 回答要求：
1. 回答必须基于参考资料，标注关键信息来源（如“华东理工大学硕士学位论文”）；
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
# 百炼平台配置

# 百炼平台配置
bailian_base_url = os.getenv("BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
bailian_api_key = os.getenv("BAILIAN_API_KEY")
bailian_model = "qwen-plus-2025-07-28"
bailian_embedding_model = "text-embedding-v2"

# 自定义Embeddings类，使用直接API调用阿里云百炼
class BailianEmbeddings(Embeddings):
    def __init__(self, api_key=None, base_url=None, model=None):
        self.api_key = api_key or bailian_api_key
        self.base_url = base_url or bailian_base_url
        self.model = model or bailian_embedding_model
    
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
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            embeddings = [item['embedding'] for item in result['data']]
            return embeddings
        except Exception as e:
            print(f"调用Embedding API失败: {e}")
            print(f"响应内容: {response.text if 'response' in locals() else '无响应'}")
            return None
    
    def embed_documents(self, texts):
        result = self._get_embeddings(texts)
        return result if result else []
    
    def embed_query(self, text):
        result = self._get_embeddings([text])
        return result[0] if result else []

# 加载文档（批量处理整个文件夹）
def load_documents(directory_path):
    loader = DirectoryLoader(
        directory_path, 
        glob="**/*.pdf",  # 加载所有PDF
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    print(f"加载了 {len(documents)} 个PDF文件")
    return documents

# 文本分割（关键参数：块大小和重叠）
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # 每个文本块大小
        chunk_overlap=50,    # 块间重叠，避免信息截断
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
    )
    texts = text_splitter.split_documents(documents)
    print(f"分割为 {len(texts)} 个文本块")
    return texts

# 生成向量并存储
def create_vectorstore(texts):
    global bailian_api_key, bailian_base_url, bailian_embedding_model
    
    # 检查API密钥
    if not bailian_api_key:
        print("错误：未设置BAILIAN_API_KEY环境变量")
        return None
    
    from langchain_core.documents import Document
    
    # 确保texts是Document对象列表，并清理文本内容
    if texts and not isinstance(texts[0], Document):
        print("警告：texts不是Document对象列表，尝试转换")
        # 转换为Document对象
        new_texts = []
        for i, text in enumerate(texts):
            if isinstance(text, str):
                # 清理文本内容，确保是纯字符串
                cleaned_text = str(text).strip()
                # 移除不可见字符和控制字符
                cleaned_text = ''.join(c for c in cleaned_text if ord(c) >= 32 or c in '\n\t')
                if cleaned_text:
                    new_texts.append(Document(page_content=cleaned_text, metadata={}))
                else:
                    print(f"跳过空文本元素: {i}")
            else:
                print(f"跳过非字符串元素: {type(text)}")
        texts = new_texts
    else:
        # 清理现有Document对象的文本内容
        cleaned_texts = []
        for doc in texts:
            if hasattr(doc, 'page_content'):
                # 清理文本内容，确保是纯字符串
                cleaned_content = str(doc.page_content).strip()
                # 移除不可见字符和控制字符
                cleaned_content = ''.join(c for c in cleaned_content if ord(c) >= 32 or c in '\n\t')
                if cleaned_content:
                    # 创建新的Document对象，保留原有metadata
                    cleaned_texts.append(Document(page_content=cleaned_content, metadata=doc.metadata))
                else:
                    print("跳过空文本Document")
            else:
                print(f"跳过非Document对象: {type(doc)}")
        texts = cleaned_texts
    
    # 检查是否有有效文本
    if not texts:
        print("错误：没有有效文本可处理")
        return None
    
    # 调试：检查texts的类型和内容
    print(f"清理后文本块数量: {len(texts)}")
    if texts:
        print(f"第一个文本块类型: {type(texts[0])}")
        if hasattr(texts[0], 'page_content'):
            print(f"第一个文本块长度: {len(texts[0].page_content)}")
            # 打印前50个字符作为示例
            sample = texts[0].page_content[:50] + "..." if len(texts[0].page_content) > 50 else texts[0].page_content
            print(f"第一个文本块示例: {sample}")
    
    # 使用全局的BailianEmbeddings类
    embeddings = BailianEmbeddings()
    
    # 批量处理文本块，每次不超过25个（阿里云百炼限制）
    batch_size = 25
    vectorstore = None
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        print(f"处理第 {i//batch_size + 1} 批文本块，共 {len(batch_texts)} 个")
        
        # 再次检查batch_texts中的每个元素
        valid_docs = []
        for j, doc in enumerate(batch_texts):
            if hasattr(doc, 'page_content') and isinstance(doc.page_content, str) and doc.page_content.strip():
                valid_docs.append(doc)
            else:
                print(f"跳过无效的Document对象: {j}")
        
        print(f"有效Document数量: {len(valid_docs)}")
        
        if valid_docs:
            if vectorstore is None:
                # 创建第一个向量库
                vectorstore = FAISS.from_documents(valid_docs, embeddings)
            else:
                # 合并到现有向量库
                temp_vectorstore = FAISS.from_documents(valid_docs, embeddings)
                vectorstore.merge_from(temp_vectorstore)
    
    # 保存到本地
    if vectorstore:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        kb_path = os.path.join(base_dir, "course_knowledge_base")
        vectorstore.save_local(kb_path)
        print(f"知识库已保存到 {kb_path} 文件夹")
        return vectorstore
    else:
        print("错误：向量库创建失败")
        return None

# 初始化向量库
def init_vectorstore(force_rebuild=False):
    global vectorstore, retriever, rag_chain, embeddings, llm, bailian_api_key, bailian_base_url, bailian_model, bailian_embedding_model
    try:
        # 检查API密钥
        if not bailian_api_key:
            print("未设置BAILIAN_API_KEY环境变量")
            return False
        
        # 导入百炼平台相关模块
        from langchain_openai import ChatOpenAI
        
        # 初始化embeddings和llm
        # 使用自定义的BailianEmbeddings类
        embeddings = BailianEmbeddings()
        llm = ChatOpenAI(
            openai_api_key=bailian_api_key,
            openai_api_base=bailian_base_url,
            model_name=bailian_model,
            temperature=0.3
        )
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        kb_path = os.path.join(base_dir, "course_knowledge_base")
        materials_path = os.path.join(base_dir, "course_materials")
        
        # 如果强制重建，删除旧的向量库
        if force_rebuild and os.path.exists(kb_path):
            import shutil
            shutil.rmtree(kb_path)
            print("已删除旧的向量库，准备重新创建")
        
        # 尝试加载现有向量库（非强制重建时）
        if not force_rebuild and os.path.exists(kb_path):
            vectorstore = FAISS.load_local(kb_path, embeddings, allow_dangerous_deserialization=True)
            print("成功加载现有向量库")
        else:
            # 尝试从course_materials文件夹创建向量库
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
        
        # 初始化检索器和RAG链
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

# 初始化向量库（启动时尝试）
init_vectorstore()

@app.post("/ask", response_model=Response)
async def ask_question(query: Query):
    try:
        # 检查API密钥
        global bailian_api_key
        if not bailian_api_key:
            raise HTTPException(status_code=503, detail="未设置BAILIAN_API_KEY环境变量")
        
        # 确保向量库已初始化
        if not rag_chain:
            if not init_vectorstore():
                raise HTTPException(status_code=503, detail="向量库未初始化，请先上传课程资料")
        
        answer = rag_chain.invoke(query.question)
        # 简单实现，实际项目中可以从retriever获取来源
        sources = []
        return Response(
            answer=answer,
            sources=sources
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "课程助手API已启动，访问 /ask 接口进行问答，/init 接口初始化知识库"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/init")
async def init_knowledge_base(force_rebuild: bool = False):
    try:
        # 检查API密钥
        global bailian_api_key
        if not bailian_api_key:
            return {"status": "error", "message": "未设置BAILIAN_API_KEY环境变量"}
        
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
        data=[
            OpenAIModel(id="course-assistant")
        ]
    )

@app.post("/v1/chat/completions")
async def chat_completions(request: OpenAIChatRequest):
    try:
        global bailian_api_key
        if not bailian_api_key:
            raise HTTPException(status_code=503, detail="未设置BAILIAN_API_KEY环境变量")
        
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
                    message=OpenAIMessage(
                        role="assistant",
                        content=answer
                    )
                )
            ]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)