from langchain.chains import RetrievalQA # 构建检索式问答链的类。这个链结合了文档检索和语言模型，以基于给定的文档源回答问题。
from langchain.llms import OpenAI # OpenAI的LLM模型
from langchain.prompts import PromptTemplate # 用于创建可定制的提示模板。提示模板允许你定义传递给语言模型的输入格式，使其更符合特定的任务需求。
from langchain.embeddings import OpenAIEmbeddings # OpenAI嵌入模型，用于将文本转换为向量表示，以便后续在向量空间中进行相似性搜索。
from langchain.vectorstores import FAISS # 向量数据库

# 自定义Prompt模板（让回答更贴合学生场景）
prompt_template = """
你是一个专业的课程助教，请基于以下参考资料回答学生的问题。
如果参考资料中包含与问题相关的内容，哪怕只有部分相关，也要详细回答；只有参考资料完全无相关内容时，才能直接回答“在提供的课程资料中找不到相关信息”。
参考资料：
{context}

学生问题：{question}
助教回答：
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# 加载之前保存的向量库"course_knowledge_base"
# embeddings 用于指定加载向量数据库时所需的嵌入模型。加载后的向量数据库 vectorstore 将用于后续的检索操作，以查找与问题相关的文本片段。
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local("course_knowledge_base", embeddings)

# 创建检索QA链
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0.3),  # 温度低一点，更严谨
    chain_type="stuff",  # 使用的链类型为 “stuff”：把所有检索结果塞给LLM
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),  # 指定检索器，它从之前加载的向量数据库 vectorstore 中获取数据，返回top3相关段落
    chain_type_kwargs={"prompt": PROMPT}  # 传递自定义的提示模板
)

# 测试
question = "机器学习课程的成绩构成是怎样的？"
answer = qa_chain.run(question)
print(answer)