from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from agent import connection


def get_retrieval_qa_chain(llm, db):
    # 加载开源词向量模型
    embeddings = HuggingFaceEmbeddings(model_name="./model/paraphrase-MiniLM-L6-v2")
    # 加载向量数据库
    vector_db = FAISS.load_local("./data/faiss_index", embeddings, allow_dangerous_deserialization=True)

    # 构造Prompt模板
    template = """If you are an SQLite expert, please generate only SQLite queries based on my question and combining database schema and contextual information, without requiring any further explanation.
    question: {question}
    schema: {schema}
    in_context：
    ···
    {context}
    ···
    answer:"""
    schema = connection.getSchema(db)
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["question", "schema", "context"], template=template)
    QA_CHAIN_PROMPT = QA_CHAIN_PROMPT.partial(schema=schema)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vector_db.as_retriever(), return_source_documents=True,
                                           chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    return qa_chain