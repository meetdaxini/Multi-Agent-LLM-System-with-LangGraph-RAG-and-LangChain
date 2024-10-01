# evaluation.py

from datasets import load_dataset
from my_rag.components.embeddings.huggingface_embedding import HuggingFaceEmbedding
from my_rag.components.vectorstores.deeplake_store import DeepLakeVectorStore
from my_rag.components.llms.huggingface_llm import HuggingFaceLLM
import numpy as np
import torch
from tqdm import tqdm
import re

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punctuation(text):
        return re.sub(r'[^\w\s]', '', text)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punctuation(lower(s))))

def compute_exact(a_gold, a_pred):
    print(a_gold)
    print(a_pred)
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_tokens = normalize_answer(a_gold).split()
    pred_tokens = normalize_answer(a_pred).split()
    common = set(gold_tokens) & set(pred_tokens)
    num_same = len(common)
    if len(gold_tokens) == 0 or len(pred_tokens) == 0:
        # If either is empty, F1 is 0
        return 0
    if num_same == 0:
        return 0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def main():
    # Load a subset of the SQuAD v2.0 dataset
    dataset = load_dataset('squad_v2', split='validation[:100]')

    # Initialize components
    embedding_model = HuggingFaceEmbedding(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        device='cpu'
    )
    vector_store = DeepLakeVectorStore(dataset_path='evaluation_vector_store', overwrite=True)
    llm = HuggingFaceLLM(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        device='cuda' if torch.cuda.is_available() else 'cpu',
        torch_dtype=torch.float16,
        load_in_8bit=True,
        device_map='auto',
        trust_remote_code=True,
    )

    # Build the vector store with contexts from the dataset
    contexts = [item['context'] for item in dataset]
    context_embeddings = embedding_model.embed(contexts)
    context_embeddings = np.array(context_embeddings)
    metadata = [{'text': ctx} for ctx in contexts]
    vector_store.add_embeddings(context_embeddings, metadata)

    # Evaluation metrics
    total = 0
    exact_matches = 0
    f1_sum = 0

    # Evaluate
    for item in tqdm(dataset):
        question = item['question']
        answers = item['answers']['text']
        # Skip unanswerable questions
        if len(answers) == 0:
            continue
        gold_answer = answers[0]

        # Generate query embedding
        query_embedding = embedding_model.embed([question])
        query_embedding = np.array(query_embedding)

        # Retrieve relevant contexts
        results = vector_store.search(query_embedding=query_embedding, k=3)
        context = " ".join([res['metadata']['text'] for res in results])

        # Generate answer using LLM with context
        generated_answer = llm.generate_response_with_context(
            context=context,
            prompt=question,
            max_length=1500,
            temperature=0.7,
            top_p=0.9,
        )

        # Compute evaluation metrics
        exact_matches += compute_exact(gold_answer, generated_answer)
        f1_sum += compute_f1(gold_answer, generated_answer)
        total += 1

    # Calculate averages
    exact_match_score = exact_matches / total * 100
    f1_score = f1_sum / total * 100

    print(f"Exact Match: {exact_match_score:.2f}%")
    print(f"F1 Score: {f1_score:.2f}%")

    # Clean up resources
    embedding_model.clean_up()
    vector_store.clean_up()
    llm.clean_up()

if __name__ == '__main__':
    main()
