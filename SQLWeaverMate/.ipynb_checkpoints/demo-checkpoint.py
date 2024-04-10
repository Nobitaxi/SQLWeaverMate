import ast
import os
import pandas as pd
import streamlit as st
from agent import connection
import rag.dataprocesses as data_processes
from rag import VectorBase
from llm import LLM
from rag import retrieval_qa
from agent import sqlqa



if __name__ == '__main__':
    st.title('SQLWeaverMate')
    option = st.sidebar.selectbox(
        '模型选择',
        ('GPT-3.5-Turbo', 'InternLM2-chat-7B-SQL')
    )
    on = st.sidebar.toggle('RAG')

    sqlite_save_path = "./restore/database"
    knowledge_save_path = "./restore/knowledge"
    db = None
    table_names = []

    uploaded_db = st.sidebar.file_uploader("选择上传数据库文件", type='sqlite')
    if uploaded_db is not None:
        # 获取文件名
        file_name = uploaded_db.name
        # 完整的文件路径
        file_path = os.path.join(sqlite_save_path, file_name)

        # 保存文件到本地
        with open(file_path, 'wb') as f:
            f.write(uploaded_db.getvalue())

        # 连接数据库

        if os.path.exists(file_path):
            db = connection.getDB(file_path)
            table_names = db.get_usable_table_names()
            # print("db.get_usable_table_names()：", db.get_usable_table_names())
        # st.success('成功识别到数据库内容!', icon="✅")

        @st.cache_data
        def fetch_and_create_dataframe(table_name):
            query_content = f"SELECT * FROM {table_name};"
            result_content = db.run(query_content)
            content_list = ast.literal_eval(result_content)
            query_columns_info = f"PRAGMA table_info({table_name});"
            result_columns_info = db.run(query_columns_info)
            result_columns_info_list = ast.literal_eval(result_columns_info)
            columns = [column_info[1] for column_info in result_columns_info_list]
            return pd.DataFrame(content_list, columns=columns)

        # 显示数据库内容
        tabs = st.tabs(table_names)
        with st.spinner('正在识别数据库内容...'):
            for i in range(len(tabs)):
                with tabs[i]:
                    df = fetch_and_create_dataframe(table_names[i])
                    st.dataframe(df)


    uploaded_knowledge = st.sidebar.file_uploader("选择上传知识库文件", type=['txt', 'md', 'pdf'])
    if uploaded_knowledge is not None:
        # 获取文件名
        file_name = uploaded_knowledge.name
        # 完整的文件路径
        file_path = os.path.join(knowledge_save_path, file_name)
        with st.spinner('正在构建知识库...'):
            # 保存文件到本地
            with open(file_path, 'wb') as f:
                f.write(uploaded_knowledge.getvalue())

            if os.path.exists(file_path):
                # 获得所有知识文本内容
                docs = data_processes.ReadFiles(knowledge_save_path).get_text()

                # 持久化向量数据库
                VectorBase.persist(docs)

                # 加载向量数据库
                vector_db = VectorBase.load()
                # print(vector_db.docstore._dict)
        # st.success('知识库构建成功!', icon="✅")



    if uploaded_db is not None:
        if option == 'GPT-3.5-Turbo':
            sql = None
            result = None
            input = st.sidebar.text_input("请输入您的查询")
            if input != "":
                with st.chat_message("user"):
                    st.write(input)
                llm = LLM.get_ChatGPT()
                if on:  # RAG
                    qa_chain = retrieval_qa.get_retrieval_qa_chain(llm, db)
                    sql, result = sqlqa.rag_get_sql_result(db, qa_chain, input)
                else:
                    schema = connection.getSchema(db)
                    sql, result = sqlqa.get_sql_result(llm, db, schema, input)
                with st.chat_message("assistant"):
                    st.write("SQL: ", sql)
                    st.write("Result: ", result)
        elif option == 'InternLM2-chat-7B-SQL':
            sql = None
            result = None
            input = st.sidebar.text_input("请输入您的查询")
            if input != "":
                with st.chat_message("user"):
                    st.write(input)
                llm = LLM.InternLM_SQL("../modelscope/InternLM2-chat-7B-SQL") # 替换自己模型地址
                if on:  # RAG
                    qa_chain = retrieval_qa.get_retrieval_qa_chain(llm, db)
                    sql, result = sqlqa.rag_get_sql_result(db, qa_chain, input)
                else:
                    schema = connection.getSchema(db)
                    sql, result = sqlqa.get_sql_result(llm, db, schema, input)
                with st.chat_message("assistant"):
                    st.write("SQL: ", sql)
                    st.write("Result: ", result)
        else:
            pass
    else:
        pass








