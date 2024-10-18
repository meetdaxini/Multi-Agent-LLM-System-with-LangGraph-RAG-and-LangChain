import os
import pandas as pd
from langchain_community.vectorstores.chroma import Chroma
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from get_embedding_function import (
    get_msmarco_embeddings,
    get_biobert_embeddings,
    get_bert_base_uncased_embeddings,
    get_roberta_base_embeddings,
    get_instructor_xl_embeddings,
    get_roberta_large_embeddings,
    get_bert_large_nli_embeddings,
    get_mxbai_embed_large_embeddings,
)

# PYTHONPATH=src python3 src/tests/my_rag_ollama/Tim_test.py

CHROMA_PATH_TEMPLATE = "chroma_{embedding_name}"
DEFAULT_DATA_PATH = "data_test"
# DEFAULT_DATA_PATH = "https://drive.google.com/drive/folders/1CbvEKSzt_JOtq_rps8T0U4YESaSWVnOZ?usp=drive_link"


def load_documents(data_path=DEFAULT_DATA_PATH):
    document_loader = PyPDFDirectoryLoader(data_path)
    return document_loader.load()


# The more chunk we have the bigger the accuracy
# The smaller the chunk the lower is the accuracy
def split_into_chunks(doc):
    text_splitter = RecursiveCharacterTextSplitter(
        # Chunk Size:
        # Increasing the chunk size means each chunk contains more text, providing more context for the model to
        # understand the content. However, larger chunks may also include irrelevant information, which can reduce
        # the precision of the retrieval. Smaller chunks focus on more specific content, but they may miss the
        # broader context needed for accurate retrieval.
        chunk_size=800,
        # Chunk Overlap:
        # Increasing chunk overlap allows for key information that falls between chunk boundaries to be captured in
        # multiple chunks, improving recall. However, too much overlap can create redundancy, leading to slower
        # processing and increased storage without significantly improving accuracy. Less overlap reduces redundancy
        # but may miss important details at the chunk edges.
        chunk_overlap=115,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_text(doc.page_content)
    chunk_docs = []
    for i, chunk in enumerate(chunks):
        chunk_doc = Document(
            page_content=chunk,
            metadata={
                **doc.metadata,
                'chunk_index': i
            }
        )
        chunk_docs.append(chunk_doc)
    return chunk_docs


def split_documents(documents, folder_path=DEFAULT_DATA_PATH):
    chunks = []
    pdf_files = sorted(os.listdir(folder_path))  # List and sort all PDF files in the folder
    for doc_index, doc in enumerate(documents):
        # Get the corresponding PDF file name based on document index
        doc_id = pdf_files[doc_index] if doc_index < len(pdf_files) else f"Document_{doc_index + 1}"

        doc.metadata['document_id'] = doc_id
        doc_chunks = split_into_chunks(doc)
        for chunk in doc_chunks:
            chunk.metadata['document_id'] = doc_id
            chunks.append(chunk)
    return chunks


def add_to_chroma(chunks, chroma_path, embedding_function):
    db = Chroma(
        persist_directory=chroma_path,
        embedding_function=embedding_function
    )
    db.add_documents(chunks)
    db.persist()
    print(f"Documents added to Chroma and saved at: {chroma_path}")


# Define multiple queries
queries = [
    "Is RANKL secreted from the cells?",
    "Is the monoclonal antibody Trastuzumab (Herceptin) of potential use in the treatment of prostate cancer?",
    "How do Yamanaka factors regulate developmental signaling in ES cells, and what unique role does c-Myc play?",
    "Is Hirschsprung disease a mendelian or a multifactorial disorder?",
    "Which are the Yamanaka factors?",
    "List signaling molecules (ligands) that interact with the receptor EGFR?",
    "Are long non coding RNAs spliced?",
    "Which miRNAs could be used as potential biomarkers for epithelial ovarian cancer?"
]


# Ground truth based on the document ID, to check weather the retrieved document is correct or not
# ground_truth = {
#     "Is RANKL secreted from the cells?"
#     : ["10.3892_mmr.2013.1572.pdf"],
#
#     "Is the monoclonal antibody Trastuzumab (Herceptin) of potential use in the treatment of prostate cancer?"
#     : ["10.1158_0008-5472.CAN-06-2731.pdf"],
#
#     # "How do Yamanaka factors regulate developmental signaling in ES cells, and what unique role does c-Myc play?": ["10.1038_leu.2013.304.pdf", "10.1038_cr.2008.309.pdf"],
#     "Is Hirschsprung disease a mendelian or a multifactorial disorder?"
#     : ["10.1186_1471-2350-12-138.pdf"],
#
#     "Which are the Yamanaka factors?"
#     : ["10.1038_cr.2008.309.pdf", "10.1038_leu.2013.304.pdf"],
#
#     "List signaling molecules (ligands) that interact with the receptor EGFR?"
#     : ["10.1371_journal.pone.0075907.pdf"],
#
#     "Are long non coding RNAs spliced?"
#     : ["10.1101_gr.132159.111.pdf"],
#
#     "Which miRNAs could be used as potential biomarkers for epithelial ovarian cancer?"
#     : ["10.3892_or.2012.1625.pdf"]
#
# }


# Define the ground truth for each query
ground_truth = {
    "Is RANKL secreted from the cells?": ["RANKL is secreted", "osteoblasts", "cytokine"],
    "Is the monoclonal antibody Trastuzumab (Herceptin) of potential use in the treatment of prostate cancer?": ["Trastuzumab", "HER2", "prostate cancer"],
    "Is Hirschsprung disease a mendelian or a multifactorial disorder?": ["Hirschsprung", "mendelian", "multifactorial"],
    "Which are the Yamanaka factors?": ["OCT4", "SOX2", "MYC", "KLF4"],
    "List signaling molecules (ligands) that interact with the receptor EGFR?": ["EGFR ligands", "EGF", "TGF-", "HB-EGF"],
    "Are long non coding RNAs spliced?": ["non coding RNAs", "spliced"],
    "Which miRNAs could be used as potential biomarkers for epithelial ovarian cancer?": ["miR-200a", "miR-100", "biomarkers", "epithelial ovarian cancer"]
}


def main():
    # Set the folder path to the folder containing PDFs (data_test)
    folder_path = DEFAULT_DATA_PATH

    # Load documents and split into chunks, using the PDF file names as document IDs
    documents = load_documents(folder_path)
    if not documents:
        print("No documents found.")
        return

    chunks = split_documents(documents, folder_path)

    # Define different embedding functions
    embedding_functions = {
        "roberta-base": get_roberta_base_embeddings(),
        "instructor-xl": get_instructor_xl_embeddings(),
        "roberta-large": get_roberta_large_embeddings(),
        "bert-large-nli": get_bert_large_nli_embeddings(),
        "mxbai-embed-large": get_mxbai_embed_large_embeddings(),
    }

    # Table to store results
    results_table = []

    # Define the values of k to check
    k_values = [5, 10, 15]

    # For each embedding function, create a separate Chroma vector store and query it
    for embedding_name, embedding_function in embedding_functions.items():
        print(f"Processing with embedding: {embedding_name}")
        chroma_path = CHROMA_PATH_TEMPLATE.format(embedding_name=embedding_name)

        # Add chunks to Chroma vector store
        add_to_chroma(chunks, chroma_path, embedding_function)

        # Create a Chroma vector store
        db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)

        # For each query, gather results
        for query in queries:
            for k in k_values:
                results = db.similarity_search_with_score(query, k=k)

                # Get the ground truth key phrases for the query
                expected_phrases = ground_truth.get(query, [])

                # Check if any of the retrieved chunks contain the expected phrases
                is_correct = any(
                    any(phrase.lower() in result[0].page_content.lower() for phrase in expected_phrases)
                    for result in results
                )

                # Gather the top result for analysis
                top_result = results[0] if results else None
                if top_result:
                    top_document_content, top_score = top_result
                    result_data = {
                        "Embedding": embedding_name,
                        "Query": query[:10],  # Shorten the query for display purposes
                        "Top k": k,  # Store the value of k
                        "Top Score": top_score,
                        "Top Result (Snippet)": top_document_content.page_content[:10],  # Adjusted snippet length
                        "Correct Retrieved": is_correct,
                    }
                else:
                    result_data = {
                        "Embedding": embedding_name,
                        "Query": query,
                        "Top k": k,  # Store the value of k
                        "Top Score": "N/A",
                        "Top Result (Snippet)": "No results",
                        "Correct Retrieved": is_correct,
                    }

                results_table.append(result_data)

        # Optionally clean up the vector store
        db.delete_collection()

    # Convert results into a pandas DataFrame
    df = pd.DataFrame(results_table)

    # Calculate accuracy for each embedding and k as before
    accuracy_df = df.groupby(['Embedding', 'Top k'])['Correct Retrieved'].mean().reset_index()
    accuracy_df.rename(columns={'Correct Retrieved': 'Accuracy'}, inplace=True)

    # Merge the accuracy data back into the original DataFrame `df`
    df = pd.merge(df, accuracy_df, on=['Embedding', 'Top k'], how='left')

    # Display the updated DataFrame with accuracy included
    print("\n", df.to_string(index=False))

    # Find the average accuracy for each embedding model
    average_accuracy_df = df.groupby('Embedding')['Accuracy'].mean().reset_index()

    # Find the highest accuracy
    best_accuracy = average_accuracy_df['Accuracy'].max()

    # Find the model(s) with the highest accuracy
    best_models = average_accuracy_df[average_accuracy_df['Accuracy'] == best_accuracy]

    # Display the accuracy of each model for each k
    print("\nAccuracy of each model for each Top k:")
    print(accuracy_df.to_string(index=False))

    # Output the best model(s)
    print("\nConclusion:")
    if len(best_models) == 1:
        print(
            f"The best performing model is: {best_models.iloc[0]['Embedding']} with an average accuracy of {best_accuracy:.4f}")
    else:
        print(
            f"The best performing models are: {', '.join(best_models['Embedding'])} with an average accuracy of {best_accuracy:.4f}")


if __name__ == "__main__":
    main()
