from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

# 读取pdf
pdf = PdfReader('./理解收益率曲线.pdf')
text = ""
for page in pdf.pages:
    text += page.extract_text()

text_spliter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

chunks = text_spliter.split_text(text)

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# 实例化一个Embeddings模型
embeddings = OpenAIEmbeddings(
    model='text-embedding-3-large',
    api_key=keys.freeoopenai,
    base_url='https://api.chatanywhere.tech/v1'
)

# 用来创建和管理向量数据库。它会把文本转换成向量，并存储起来，方便之后快速查找相似的内容
vector_store = FAISS.from_texts(chunks, embeddings)


# 查找和最相似的文本内容，并返回最相似的前3个结果及其相似度分数L2距离，L2越小越相似
docs = vector_store.similarity_search_with_score("到期收益率曲线", k=3)

# 保存
vector_store.save_local('my_vectorDB')

# 读取
vector_store_read = FAISS.load_local('my_vectorDB', embeddings,allow_dangerous_deserialization=True)
vector_store_read.similarity_search_with_score("到期收益率曲线", k=3)


from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader


def pdf_to_vector(pdf_path):
    pdf = PdfReader(pdf_path)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()

    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    chunks = text_spliter.split_text(text)

    # 实例化一个Embeddings模型
    embeddings = OpenAIEmbeddings(
        model='text-embedding-3-large',
        api_key=keys.freeoopenai,
        base_url='https://api.chatanywhere.tech/v1'
    )

    # 用来创建和管理向量数据库。它会把文本转换成向量，并存储起来，方便之后快速查找相似的内容
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

    db1 = pdf_to_vector('./理解收益率曲线.pdf')
db2 = pdf_to_vector('./海龟交易法则.pdf')

len(db1.docstore._dict)
db1.merge_from(db2)

from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import keys
import warnings
warnings.filterwarnings("ignore")

model = ChatOpenAI(model='deepseek-chat', api_key=keys.test, base_url='https://api.deepseek.com')
# model_openai = ChatOpenAI(model='gpt-4o', api_key=keys.freeoopenai, base_url='https://api.chatanywhere.tech/v1')
# model_silicon = ChatOpenAI(model='deepseek-ai/DeepSeek-R1-Distill-Llama-8B', api_key=keys.silicon, base_url='https://api.siliconflow.cn/v1')

question = "到期收益率曲线"
docs = db1.similarity_search(question, k=1)

chain = load_qa_chain(llm=model, chain_type='stuff')
# chain = load_qa_chain(llm=model_openai, chain_type='stuff')
# chain = load_qa_chain(llm=model_silicon, chain_type='stuff')

# with get_openai_callback() as cb:
responses = chain.run(input_documents=docs, question=question)