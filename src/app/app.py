import streamlit as st
import tempfile
from pathlib import Path
import os
from typing import List, Optional
import torch
import gc
from my_rag.components.pdf_loader import PDFLoader
from my_rag.components.pipeline.document_processor import DocumentProcessor
from my_rag.components.pipeline.embedder import DocumentEmbedder, QueryEmbedder
from my_rag.components.pipeline.retriever import Retriever
from my_rag.components.pipeline.generator import Generator
from my_rag.components.pipeline.rag_pipeline import RAGPipeline
from my_rag.components.embeddings.huggingface_embedding import HuggingFaceEmbedding
from my_rag.components.embeddings.aws_embedding import AWSBedrockEmbedding
from my_rag.components.llms.huggingface_llm import HuggingFaceLLM
from my_rag.components.llms.aws_llm import AWSBedrockLLM
from my_rag.components.vectorstores.chroma_store import (
    ChromaVectorStore,
    CollectionMode,
)
import configparser


class SimpleRAGApp:
    EMBEDDING_MODELS = {
        "MiniLM-L6": "sentence-transformers/all-MiniLM-L6-v2",
        "Amazon Titan Embed text v2": "amazon.titan-embed-text-v2:0",
        "MixedBread Embed Large v1": "mixedbread-ai/mxbai-embed-large-v1",
        "Stella 1.5B": "dunzhang/stella_en_1.5B_v5",
        "NVIDIA NV-Embed-v2": "nvidia/NV-Embed-v2",
    }

    LLM_MODELS = {
        "Claude 3.5 Sonnet v1": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "Llama 3 8B Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    }

    def __init__(self, params):
        self.pdf_loader = PDFLoader()
        self.params = params
        self.setup_models()
        self.setup_vector_store()
        self.setup_pipeline()

    def clean_up_resources(self):
        if hasattr(self, "embedding_model"):
            self.embedding_model.clean_up()
        if hasattr(self, "llm"):
            self.llm.clean_up()
        if hasattr(self, "vector_store"):
            self.vector_store.clean_up()
        torch.cuda.empty_cache()
        gc.collect()

    def get_model_config(self, model_name):
        """Returns appropriate configuration for each model"""
        if model_name == "nvidia/NV-Embed-v2":
            return {
                "trust_remote_code": True,
                "load_in_8bit": True,
                "max_length": 32768,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            }
        elif model_name == "meta-llama/Meta-Llama-3-8B-Instruct":
            return {
                "trust_remote_code": True,
                "load_in_8bit": True,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "device_map": "auto",
            }
        else:
            return {
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "trust_remote_code": True,
            }

    def setup_aws_credentials(self):
        config_file = "/home/ubuntu/Multi-Agent-LLM-System-with-LangGraph-RAG-and-LangChain/config/config.ini"
        config_name = "BedRock_LLM_API"
        config = configparser.ConfigParser()
        config.read(config_file)
        return {
            "aws_access_key_id": config[config_name]["aws_access_key_id"],
            "aws_secret_access_key": config[config_name]["aws_secret_access_key"],
            "aws_session_token": config[config_name]["aws_session_token"],
            "region_name": config[config_name]["region"],
        }

    def setup_models(self):
        self.clean_up_resources()

        # Setup embedding model
        model_name = self.EMBEDDING_MODELS[self.params["embedding_model"]]
        if model_name.startswith("amazon."):
            aws_creds = self.setup_aws_credentials()
            self.embedding_model = AWSBedrockEmbedding(model_id=model_name, **aws_creds)
        else:
            model_config = self.get_model_config(model_name)
            self.embedding_model = HuggingFaceEmbedding(
                model_name=model_name, **model_config
            )

        # Setup LLM
        model_name = self.LLM_MODELS[self.params["llm_model"]]
        if model_name.startswith("anthropic."):
            aws_creds = self.setup_aws_credentials()
            self.llm = AWSBedrockLLM(model_id=model_name, **aws_creds)
        else:
            model_config = self.get_model_config(model_name)
            self.llm = HuggingFaceLLM(model_name=model_name, **model_config)

    def setup_vector_store(self):
        self.vector_store = ChromaVectorStore(
            collection_name=f"rag_{self.params['embedding_model'].replace('/', '_').replace(' ', '_')}",
            mode=CollectionMode.DROP_IF_EXISTS,
        )

    def setup_pipeline(self):
        # Set batch size based on embedding model
        batch_size = 3
        self.pipeline = RAGPipeline(
            [
                DocumentProcessor(
                    chunk_size=self.params["chunk_size"],
                    chunk_overlap=self.params["chunk_overlap"],
                ),
                DocumentEmbedder(
                    embedding_model=self.embedding_model, batch_size=batch_size
                ),
                QueryEmbedder(
                    embedding_model=self.embedding_model, batch_size=batch_size
                ),
                Retriever(vector_store=self.vector_store, k=self.params["k"]),
                Generator(
                    llm=self.llm,
                    system_message=self.params["system_message"],
                    generation_config={
                        "max_tokens": self.params["max_tokens"],
                        "temperature": self.params["temperature"],
                        "top_k": self.params["top_k"],
                        "top_p": self.params["top_p"],
                        "anthropic_version": (
                            "bedrock-2023-05-31"
                            if "claude" in self.params["llm_model"].lower()
                            else None
                        ),
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

    def query_documents(self, query: str) -> dict:
        pipeline_data = self.pipeline.run(queries=[query])

        return {
            "answer": (
                pipeline_data.generated_responses[0]
                if pipeline_data.generated_responses
                else None
            ),
            "retrieved_documents": (
                pipeline_data.retrieved_documents[0]
                if pipeline_data.retrieved_documents
                else []
            ),
            "retrieved_metadata": (
                pipeline_data.retrieved_metadata[0]
                if pipeline_data.retrieved_metadata
                else []
            ),
            "query": query,
        }


def main():
    st.set_page_config(page_title="Simple RAG App", layout="wide")
    st.title("Simple RAG App")

    # Model Selection
    st.sidebar.header("Model Selection")

    embedding_model = st.sidebar.selectbox(
        "Embedding Model", options=list(SimpleRAGApp.EMBEDDING_MODELS.keys())
    )

    llm_model = st.sidebar.selectbox(
        "Language Model", options=list(SimpleRAGApp.LLM_MODELS.keys())
    )

    # System Message Template
    st.sidebar.subheader("System Message")
    default_system_message = """You are an AI assistant that provides accurate and helpful answers
    based on the given context. Your responses should be:
    1. Focused on the provided context
    2. Clear and concise
    3. Accurate and relevant to the question
    4. Based only on the information given"""

    system_message = st.sidebar.text_area(
        "Edit System Message", value=default_system_message, height=150
    )

    # Pipeline Parameters
    with st.sidebar.expander("Pipeline Parameters"):
        chunk_size = st.slider("Chunk Size", 500, 8000, 4000, 100)
        chunk_overlap = st.slider("Chunk Overlap", 0, 1000, 200, 50)
        k = st.slider("Number of Retrieved Documents (k)", 1, 10, 5)
        max_tokens = st.slider("Max Tokens", 64, 1024, 512, 64)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        top_k = st.slider("Top-k", 1, 100, 50, 1)
        top_p = st.slider("Top-p", 0.1, 1.0, 0.95, 0.05)

    params = {
        "embedding_model": embedding_model,
        "llm_model": llm_model,
        "system_message": system_message,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "k": k,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
    }

    # Update RAG app when parameters change
    if "rag_app" not in st.session_state or st.sidebar.button("Update Configuration"):
        with st.spinner("Updating configuration and loading models..."):
            if "rag_app" in st.session_state:
                st.session_state.rag_app.clean_up_resources()
            st.session_state.rag_app = SimpleRAGApp(params)
            st.sidebar.success("Configuration updated!")

    # Main interface with tabs
    tab1, tab2 = st.tabs(["Document Processing", "Question Answering"])

    with tab1:
        st.header("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF, TXT, or MD files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            if st.button("Process Documents"):
                with st.spinner("Processing documents..."):
                    documents = st.session_state.rag_app.process_documents(
                        uploaded_files
                    )
                    st.success(f"Successfully processed {len(documents)} documents")

    with tab2:
        st.header("Ask Questions")
        query = st.text_input("Enter your question:")

        if query:
            if st.button("Get Answer"):
                with st.spinner("Generating answer..."):
                    result = st.session_state.rag_app.query_documents(query)

                    # Display answer
                    st.markdown("### Answer")
                    st.write(result["answer"])

                    # Display retrieved documents in an expander
                    with st.expander("View Retrieved Context"):
                        for i, (doc, metadata) in enumerate(
                            zip(
                                result["retrieved_documents"],
                                result["retrieved_metadata"],
                            )
                        ):
                            st.markdown(
                                f"**Document {i+1}** (Source: {metadata.get('doc_id', 'Unknown')})"
                            )
                            st.text(doc)

                    # Display full context in an expander
                    with st.expander("View Full Query Details"):
                        st.markdown("**Query:**")
                        st.text(result["query"])
                        st.markdown("**System Message:**")
                        st.text(system_message)
                        st.markdown("**Model Configuration:**")
                        st.json(params)


if __name__ == "__main__":
    main()
