from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
import streamlit as st
import os

from prompts import prompt

api_key=st.secrets["OPENAI_API_KEY"]

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=api_key)
db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "./2023_pdf"))
db_2023 = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0
)

@tool
def search_yoram_2023(search_word):
    """Retrieve information from 2023 sogang university Catalog return context from catalog"""
    retrieved_docs = db_2023.similarity_search(search_word)
    return {"context": retrieved_docs}


def get_answer_from_agent(question):
    tools=[search_yoram_2023]
    llm_with_tools=llm.bind_tools(tools)
    agent_executor=create_react_agent(llm_with_tools, tools)
    for chunk in agent_executor.stream(
        {"messages": [HumanMessage(content=question)]}
    ):
        final_message=chunk
    return final_message["agent"]["messages"][0].content

def main():
    st.title("유니피")
    st.write("안녕하세요 저는 유니피에요. 원하는 질문을 입력해주세요.")
    user_input = st.text_input("질문을 입력하세요")
    if st.button("Send"):
        response = get_answer_from_agent(user_input)
        st.write(response)
        

if __name__ == "__main__":
    main()

