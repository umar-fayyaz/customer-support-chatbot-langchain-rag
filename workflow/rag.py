# rag_pipeline.py

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()  # Load environment variables from .env file



# Load and split documents
def load_pdf_file(data_path: str):
    """Load all PDFs from a directory."""
    loader = DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


def text_split(extracted_data, chunk_size=800, chunk_overlap=200):
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(extracted_data)


# 2. Setup RAG pipeline
def build_rag_pipeline(text_chunks):
    """
    Build a hybrid RAG pipeline with Pinecone (dense) + BM25 (sparse).
    Assumes Pinecone index already has embeddings ingested.
    """
        # Config
    index_name = "remote-lock"

    # Embeddings + LLM
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = ChatOpenAI(model="gpt-4.1-mini")

    # Load Pinecone index (no ingestion, just connect)
    dense_vector = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    dense_retriever = dense_vector.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # BM25 retriever
    sparse_retriever = BM25Retriever.from_documents(text_chunks)
    sparse_retriever.k = 3

    # Hybrid retriever
    hybrid_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[0.7, 0.3]
    )

    # Prompt
    prompt = PromptTemplate.from_template("""
    Answer the question based on the context below.
    If user asks any question that is not in context, handle it politely.
                                          
    Context:
    {context}

    Question: {input}
    """)

    # Document chain
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

    # RAG chain
    rag_chain = create_retrieval_chain(
        retriever=hybrid_retriever,
        combine_docs_chain=document_chain
    )

    return rag_chain


# Example usage (if run directly)
if __name__ == "__main__":
    # If you still want local BM25 support, load text_chunks
    extracted_data = load_pdf_file("Data/")
    text_chunks = text_split(extracted_data)

    rag_chain = build_rag_pipeline(text_chunks)

    # Test query
    result = rag_chain.invoke({"input": "What is remote lock?"})
    print(result["answer"])
