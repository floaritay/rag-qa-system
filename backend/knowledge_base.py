from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os

# 1. 加载文档（批量处理整个文件夹）
def load_documents(directory_path):
    loader = DirectoryLoader(
        directory_path, 
        glob="**/*.pdf",  # 加载所有PDF
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    print(f"加载了 {len(documents)} 个PDF文件")
    return documents

# 2. 文本分割（关键参数：块大小和重叠）
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # 每个文本块大小
        chunk_overlap=50,    # 块间重叠，避免信息截断
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
    )
    texts = text_splitter.split_documents(documents)
    print(f"分割为 {len(texts)} 个文本块")
    return texts

# 3. 生成向量并存储
def create_vectorstore(texts):
    # 使用OpenAI的嵌入模型
    embeddings = OpenAIEmbeddings()
    
    # 创建FAISS向量数据库（本地存储，无需服务器）
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    # 保存到本地
    vectorstore.save_local("course_knowledge_base")
    print("知识库已保存到 course_knowledge_base 文件夹")
    return vectorstore

# 执行
if __name__ == "__main__":
    docs = load_documents("./course_materials")
    texts = split_documents(docs)
    vectorstore = create_vectorstore(texts)