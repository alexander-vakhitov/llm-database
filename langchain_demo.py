from langchain_community.utilities import SQLDatabase
from typing_extensions import TypedDict
import os

class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str
    

if __name__ == '__main__':
    # db = SQLDatabase.from_uri("sqlite:///Chinook.db")
    db = SQLDatabase.from_uri("sqlite:///aware_data.db")
    print(db.dialect)
    print(db.get_usable_table_names())
    # db.run("SELECT * FROM Artist LIMIT 10;")
    
    # AIzaSyBISAXX0l6wJDcBzJdVYD8cQxVseUbctuw
    
    if not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = "AIzaSyBISAXX0l6wJDcBzJdVYD8cQxVseUbctuw"

    from langchain.chat_models import init_chat_model

    llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
    llm.invoke("Hello, world!")
    
    from langchain_core.prompts import ChatPromptTemplate

    system_message = """
    Given an input question, create a syntactically correct {dialect} query to
    run to help find the answer. Unless the user specifies in his question a
    specific number of examples they wish to obtain, always limit your query to
    at most {top_k} results. You can order the results by a relevant column to
    return the most interesting examples in the database.

    Never query for all the columns from a specific table, only ask for a the
    few relevant columns given the question.

    Pay attention to use only the column names that you can see in the schema
    description. Be careful to not query for columns that do not exist. Also,
    pay attention to which column is in which table.

    Only use the following tables:
    {table_info}
    """

    user_prompt = "Question: {input}"

    query_prompt_template = ChatPromptTemplate(
        [("system", system_message), ("user", user_prompt)]
    )

    for message in query_prompt_template.messages:
        message.pretty_print()
        
    from typing_extensions import Annotated


    class QueryOutput(TypedDict):
        """Generated SQL query."""

        query: Annotated[str, ..., "Syntactically valid SQL query."]


    def write_query(state: State):
        """Generate SQL query to fetch information."""
        prompt = query_prompt_template.invoke(
            {
                "dialect": db.dialect,
                "top_k": 10,
                "table_info": db.get_table_info(),
                "input": state["question"],
            }
        )
        structured_llm = llm.with_structured_output(QueryOutput)
        result = structured_llm.invoke(prompt)
        return {"query": result["query"]}
    
    # print(write_query({"question": "How many Employees are there?"}))
    print(write_query({"question": "How many forklifts are there?"}))

