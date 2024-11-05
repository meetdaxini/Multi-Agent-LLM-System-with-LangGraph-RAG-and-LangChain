from mar_pipeline import ImprovedMemoryAugmentedRAG
from my_rag.components.memory_db.improved_mar import ChromaMemoryDB
from mar_pipeline import ChromaDBHandler
from my_rag.components.llms.huggingface_llm import HuggingFaceLLM
from my_rag.components.embeddings.huggingface_embedding import HuggingFaceEmbedding

class MemLongMARPipeline(ImprovedMemoryAugmentedRAG):
    def __init__(self, embedding_model, llm, chromadb_handler, memory_db, main_db_path="data_test"):
        super().__init__(embedding_model=embedding_model, embedding_dim=512, llm=llm,
                         chromadb_handler=chromadb_handler, memory_db=memory_db)
        self.main_db_path = main_db_path  # Path to main DB with PDF data

    def populate_memory_db(self, prepopulated_documents):
        """
        Populates the memory database (MemoryDB) with prepopulated documents.
        """
        for doc in prepopulated_documents:
            self.add_to_memory([doc])  # Using `add_to_memory` from ImprovedMemoryAugmentedRAG
        print("MemoryDB populated with preloaded documents.")

    def search_memory_first(self, question_text):
        """
        Searches MemoryDB for similar questions. If found, retrieves the answer.
        If no relevant information, falls back to main database.
        """
        result = self.process_query(question_text, k=5)
        if result['retrieved_documents']:
            print("Answer found in MemoryDB.")
            return result['answer']
        print("No answer found in MemoryDB. Fallback to MainDB.")
        return result['answer']

# Example usage
embedding_model = HuggingFaceEmbedding(model_name="dunzhang/stella_en_1.5B_v5")
llm = HuggingFaceLLM(model_name="meta-llama/Meta-Llama-3-8B-Instruct")
chromadb_handler = ChromaDBHandler("main_collection")
memory_db = ChromaMemoryDB(collection_name="interaction_memory")

pipeline = MemLongMARPipeline(embedding_model, llm, chromadb_handler, memory_db)

# Prepopulate the memory
prepopulated_documents = [
        "Question: Is Hirschsprung disease a mendelian or a multifactorial disorder? Answer: Coding sequence mutations in RET, GDNF, EDNRB, EDN3, and SOX10 are involved in the development of Hirschsprung disease. The majority of these genes was shown to be related to Mendelian syndromic forms of Hirschsprung's disease, whereas the non-Mendelian inheritance of sporadic non-syndromic Hirschsprung disease proved to be complex; involvement of multiple loci was demonstrated in a multiplicative model.",
        "Question: List EGFR receptor signaling molecules. Answer: The 7 known EGFR ligands are: epidermal growth factor (EGF), betacellulin (BTC), epiregulin (EPR), heparin-binding EGF (HB-EGF), transforming growth factor-α [TGF-α], amphiregulin (AREG), and epigen (EPG).",
        "Question: Do long non-coding RNAs get spliced? Answer: Long non coding RNAs appear to be spliced through the same pathway as the mRNAs."
    ]
pipeline.populate_memory_db(prepopulated_documents)

# Search for a question
question_text = "Is Hirschsprung disease a mendelian or a multifactorial disorder?"
answer = pipeline.search_memory_first(question_text)
print("Generated Answer:", answer)
