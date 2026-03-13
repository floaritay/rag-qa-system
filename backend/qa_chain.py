from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# 自定义Prompt模板（让回答更贴合学生场景）
prompt_template = """你是一个专业的课程助教，请基于以下参考资料回答学生的问题。
如果你不知道答案，请直接说"在提供的课程资料中找不到相关信息"。
参考资料：
{context}

学生问题：{question}
助教回答："""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# 加载之前保存的向量库
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local("course_knowledge_base", embeddings)

# 创建检索QA链
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0.3),  # 温度低一点，更严谨
    chain_type="stuff",  # 把所有检索结果塞给LLM
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),  # 返回top3相关段落
    chain_type_kwargs={"prompt": PROMPT}
)

# 测试
question = "机器学习课程的成绩构成是怎样的？"
answer = qa_chain.run(question)
print(answer)