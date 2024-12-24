from langchain_community.document_loaders import PyPDFLoader
import os
import getpass
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

pdfloader=PyPDFLoader("./sogang2023.pdf",)

pdf_2023=pdfloader.load()
print(pdf_2023[0].metadata)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True,)
all_splits = text_splitter.split_documents(pdf_2023)


embeddings = OpenAIEmbeddings( model="text-embedding-3-large",)

vectorstore = FAISS.from_documents(all_splits, embedding = embeddings,distance_strategy = DistanceStrategy.COSINE)
vectorstore.save_local("./db/2023_pdf")

print("문서 임베딩 및 벡터 스토어 추가 완료!")






