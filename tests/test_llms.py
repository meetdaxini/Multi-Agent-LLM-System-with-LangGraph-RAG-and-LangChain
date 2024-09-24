from my_rag.llms.huggingface_llm import HuggingFaceLLM
import torch

def main():
    # Parameters
    # model_name = "meta-llama/Llama-2-7b-chat-hf"
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    prompt = "Explain the theory of relativity in simple terms."
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize the LLM
    llm = HuggingFaceLLM(
        model_name=model_name,
        device=device,
        torch_dtype=torch.float16,
        load_in_8bit=True,
        device_map='auto',
        trust_remote_code=True,
    )

    # Generate text
    generated_text = llm.generate_response(
        prompt=prompt,
        max_length=256,
        no_repeat_ngram_size=2,
        early_stopping=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )

    print("Generated Text:")
    print(generated_text)

    # Clean up resources
    llm.clean_up()

if __name__ == '__main__':
    main()
