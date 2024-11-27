import streamlit as st
import tempfile
from pathlib import Path
import os
from typing import List, Optional
import torch
from my_rag.components.pdf_loader import PDFLoader
from my_rag.components.pipeline.document_processor import DocumentProcessor
from my_rag.components.pipeline.embedder import DocumentEmbedder, QueryEmbedder
from my_rag.components.pipeline.retriever import Retriever
from my_rag.components.pipeline.generator import Generator
from my_rag.components.pipeline.rag_pipeline import RAGPipeline
from my_rag.components.embeddings.huggingface_embedding import HuggingFaceEmbedding
from my_rag.components.llms.aws_llm import AWSBedrockLLM
from my_rag.components.vectorstores.chroma_store import (
    ChromaVectorStore,
    CollectionMode,
)
import configparser


class RAGApp:
    def __init__(self, params):
        self.pdf_loader = PDFLoader()
        self.params = params
        self.setup_embedding_model()
        self.setup_llm()
        self.setup_vector_store()
        self.setup_pipeline()

    def setup_embedding_model(self):
        self.embedding_model = HuggingFaceEmbedding(
            model_name="mixedbread-ai/mxbai-embed-large-v1",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    def setup_llm(self):
        config_file = "/home/ubuntu/Multi-Agent-LLM-System-with-LangGraph-RAG-and-LangChain/config/config.ini"
        config_name = "BedRock_LLM_API"
        config = configparser.ConfigParser()
        config.read(config_file)
        aws_access_key = config[config_name]["aws_access_key_id"]
        aws_secret_key = config[config_name]["aws_secret_access_key"]
        aws_session_token = config[config_name]["aws_session_token"]
        region = config[config_name]["region"]
        self.llm = AWSBedrockLLM(
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            aws_session_token=aws_session_token,
            region_name=region,
        )

    def setup_vector_store(self):
        self.vector_store = ChromaVectorStore(
            collection_name="rag_app", mode=CollectionMode.DROP_IF_EXISTS
        )

    def setup_pipeline(self):
        self.pipeline = RAGPipeline(
            [
                DocumentProcessor(
                    chunk_size=self.params["chunk_size"],
                    chunk_overlap=self.params["chunk_overlap"],
                ),
                DocumentEmbedder(embedding_model=self.embedding_model),
                QueryEmbedder(embedding_model=self.embedding_model),
                Retriever(vector_store=self.vector_store, k=self.params["k"]),
                Generator(
                    llm=self.llm,
                    generation_config={
                        "max_tokens": self.params["max_tokens"],
                        "temperature": self.params["temperature"],
                        "top_k": self.params["top_k"],
                        "top_p": self.params["top_p"],
                        "anthropic_version": "bedrock-2023-05-31",
                    },
                ),
            ]
        )

    def process_file(self, file) -> str:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(file.name).suffix
        ) as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name

        try:
            if file.name.endswith(".pdf"):
                content = self.pdf_loader.load_single_pdf(tmp_path)
            elif file.name.endswith((".txt", ".md")):
                with open(tmp_path, "r", encoding="utf-8") as f:
                    content = f.read()
            else:
                content = ""
        finally:
            os.unlink(tmp_path)

        return content

    def process_documents(self, files) -> List[str]:
        documents = []
        document_ids = []

        for file in files:
            content = self.process_file(file)
            if content:
                documents.append(content)
                document_ids.append(file.name)

        self.pipeline.run(
            documents=documents,
            document_ids=document_ids,
        )

        return documents

    def query_documents(self, query: str) -> Optional[str]:
        pipeline_data = self.pipeline.run(
            queries=[query],
        )

        return (
            pipeline_data.generated_responses[0]
            if pipeline_data.generated_responses
            else None
        )


def main():
    st.set_page_config(page_title="RAG App", layout="wide")
    st.title("RAG App")

    st.sidebar.header("Pipeline Parameters")

    st.sidebar.subheader("Document Processing")
    chunk_size = st.sidebar.slider("Chunk Size", 500, 8000, 4000, 100)
    chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 1000, 200, 50)

    st.sidebar.subheader("Retrieval")
    k = st.sidebar.slider("Number of Retrieved Documents (k)", 1, 10, 5)

    st.sidebar.subheader("Generation")
    max_tokens = st.sidebar.slider("Max Tokens", 64, 1024, 512, 64)
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    top_k = st.sidebar.slider("Top-k", 1, 100, 50, 1)
    top_p = st.sidebar.slider("Top-p", 0.1, 1.0, 0.95, 0.05)

    params = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "k": k,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
    }

    if "rag_app" not in st.session_state or st.sidebar.button("Update Parameters"):
        st.session_state.rag_app = RAGApp(params)
        st.sidebar.success("Parameters updated!")

    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or MD files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                documents = st.session_state.rag_app.process_documents(uploaded_files)
                st.success(f"Successfully processed {len(documents)} documents")

    st.header("Ask Questions")
    query = st.text_input("Enter your question:")

    if query:
        if st.button("Get Answer"):
            with st.spinner("Generating answer..."):
                answer = st.session_state.rag_app.query_documents(query)
                st.write("Answer:")
                st.write(answer)


if __name__ == "__main__":
    main()
