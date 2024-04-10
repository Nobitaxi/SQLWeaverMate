from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.chains import create_sql_query_chain


sql = None


def extract_result(qa_output):
    global sql
    sql = qa_output['result']
    return qa_output['result']

def extract_result2(wq_output):
    global sql
    sql = wq_output
    return wq_output


def rag_get_sql_result(db, qa_chain, question):
    execute_query = QuerySQLDataBaseTool(db=db)
    chain = qa_chain | extract_result | execute_query
    result = chain.invoke({'query': question})
    return sql, result


def get_sql_result(llm, db, schema, question):
    write_query = create_sql_query_chain(llm, db)
    execute_query = QuerySQLDataBaseTool(db=db)
    chain = write_query | extract_result2 | execute_query
    prompt_with_context = chain.get_prompts()[0].partial(table_info=schema)
    # print(prompt_with_context.pretty_repr())
    result = chain.invoke({"question": question})
    return sql, result
