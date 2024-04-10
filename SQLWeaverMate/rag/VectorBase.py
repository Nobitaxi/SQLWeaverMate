from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def persist(docs):
    # 对文本进行分块
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
    split_docs = text_splitter.split_documents(docs)

    # 加载开源词向量模型
    embeddings = HuggingFaceEmbeddings(model_name="./model/paraphrase-MiniLM-L6-v2")

    # 构建向量数据库
    # 定义持久化路径
    persist_directory = "./data/faiss_index"
    # 加载数据库
    vectordb = FAISS.from_documents(split_docs, embeddings)
    # 将加载的向量数据库持久化到磁盘上
    vectordb.save_local(persist_directory)

def load():
    # 加载开源词向量模型
    embeddings = HuggingFaceEmbeddings(model_name="./model/paraphrase-MiniLM-L6-v2")
    # 加载向量数据库
    vector_db = FAISS.load_local("./data/faiss_index", embeddings, allow_dangerous_deserialization=True)
    # print(vector_db.docstore._dict)
    # print(vector_db.index.ntotal)
    return vector_db