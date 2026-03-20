from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader # 加载PDF文件
from langchain_text_splitters import RecursiveCharacterTextSplitter # 文本分割器
from langchain_openai import OpenAIEmbeddings # OpenAI嵌入模型，用于将文本转换为向量表示，以便后续在向量空间中进行相似性搜索。
from langchain_community.vectorstores import FAISS # 向量数据库
import os

# 1. 加载文档（批量处理整个文件夹）
def load_documents(directory_path):
    loader = DirectoryLoader(
        directory_path, 
        glob="**/*.pdf",  # 加载所有PDF
        loader_cls=PyPDFLoader # 使用PyPDFLoader加载PDF文件
    )
    documents = loader.load() # 加载所有PDF文件
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
    
    # 创建一个 FAISS 向量数据库（本地存储，无需服务器） ，它将文本块 texts 转换为向量，并使用 embeddings 模型进行嵌入计算。
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    # 将向量数据库保存到本地的 course_knowledge_base 文件夹中。
    vectorstore.save_local("course_knowledge_base")
    print("知识库已保存到 course_knowledge_base 文件夹")
    return vectorstore

# 执行
if __name__ == "__main__":
    docs = load_documents("./course_materials")
    texts = split_documents(docs)
    vectorstore = create_vectorstore(texts)