from dotenv import load_dotenv

load_dotenv()

import os
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

index_name = "tax-markdown-index"

# 1) 인덱스 생성 (없을 때만)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=3072,  # text-embedding-3-large 차원
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print("인덱스 생성 완료")

# 2) 문서 불러오기 + 나누기
loader = Docx2txtLoader("./tax_with_markdown.docx")  # 실제 파일명으로 변경
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
documents = loader.load_and_split(text_splitter=text_splitter)
print(f"문서 조각 수: {len(documents)}")

# 3) 임베딩 후 업로드
embedding = OpenAIEmbeddings(model="text-embedding-3-large")
PineconeVectorStore.from_documents(documents, embedding, index_name=index_name)
print("업로드 완료")
