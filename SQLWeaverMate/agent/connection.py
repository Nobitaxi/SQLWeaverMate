import re

from langchain_community.utilities import SQLDatabase

def getDB(path):
    db = SQLDatabase.from_uri("sqlite:///{}".format(path))
    return db

def getSchema(db):
    context = db.get_context()
    table_info = context["table_info"]
    create_table_patterns = re.findall(r'CREATE TABLE.*?[/][*]', table_info, re.DOTALL)
    schema = ""
    for create_table in create_table_patterns:
        create_table = create_table.replace("\n/*", "")
        schema += create_table
    return schema
