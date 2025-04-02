# 1. 加载文档
import os
from dotenv import load_dotenv
# 加载环境变量
load_dotenv()

from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(
    web_paths=("https://zh.wikipedia.org/wiki/黑神话：悟空",)
)
docs = loader.load()

# 2. 分割文档
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# 3. 设置嵌入模型
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 4. 创建向量存储
from langchain_core.vectorstores import InMemoryVectorStore

vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(all_splits)

# 5. 创建检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 6. 创建提示模板
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
基于以下上下文，回答问题。如果上下文中没有相关信息，
请说"我无法从提供的上下文中找到相关信息"。
上下文: {context} 
问题: {question} 
回答:""")

# 7. 设置语言模型和输出解析器
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = ChatOllama(model=os.getenv("OLLAMA_MODEL"))

# 8. 构建 LCEL 链
# 管道式数据流像使用 Unix 命令管道 (|) 一样，将不同的处理逻辑串联在一起
chain = (
    {
        # 检索器输入:问题字符串, 输出:Document列表
        # lambda函数输入:Document列表, 输出:合并后的文本字符串
        "context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
        # RunnablePassthrough输入:问题字符串, 输出:原样传递问题字符串
        "question": RunnablePassthrough()
    }
    # prompt输入:字典{"context":文本,"question":问题}, 输出:格式化后的提示模板字符串
    | prompt
    # llm输入:提示模板字符串, 输出:ChatMessage对象
    | llm
    # StrOutputParser输入:ChatMessage对象, 输出:回答文本字符串
    | StrOutputParser()
)

# 查看每个阶段的输入输出
question = "测试问题"

# 1. 检索器阶段
retriever_output = retriever.invoke(question)
print("检索器输出:", retriever_output)

# 2. 合并文档阶段
context = "\n\n".join(doc.page_content for doc in retriever_output)
print("合并文档输出:", context)

# 3. 提示模板阶段
prompt_output = prompt.invoke({"context": context, "question": question})
print("提示模板输出:", prompt_output)

# 4. LLM阶段
llm_output = llm.invoke(prompt_output)
print("LLM输出:", llm_output)

# 5. 解析器阶段
final_output = StrOutputParser().invoke(llm_output)
print("最终输出:", final_output)

# 9. 执行查询
question = "黑悟空有哪些游戏场景？"
response = chain.invoke(question) # 同步，可以换成异步执行
