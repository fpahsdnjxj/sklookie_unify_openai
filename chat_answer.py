from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import streamlit as st

from prompts import prompt

load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=api_key)
db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "./2023_pdf"))
db_2023 = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)


llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0,
    openai_api_key=api_key
)

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    if not state["question"] or not isinstance(state["question"], str):
        raise ValueError("The question must be a non-empty string.")
    retrieved_docs = db_2023.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

def main():
    st.title("유니피")
    st.write("안녕하세요 저는 유니피에요. 원하는 질문을 입력해주세요.")
    user_input = st.text_input("질문을 입력하세요")
    if st.button("Send"):
        response = graph.invoke({"question": user_input})
        st.write(response["answer"])
        

if __name__ == "__main__":
    main()

