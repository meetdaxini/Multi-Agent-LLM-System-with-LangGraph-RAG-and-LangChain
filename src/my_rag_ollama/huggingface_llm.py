import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .base import BaseLLM
import gc
from typing import Optional, Dict, Any

class HuggingFaceLLM(BaseLLM):
    """
    LLM implementation using Hugging Face transformers.
    """

    def __init__(
        self,
        model_name: str, # TODO: add validations based on supported models
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = torch.float32,
        load_in_8bit: bool = False,
        device_map: Optional[str] = None,
        max_memory: Optional[Dict[str, str]] = None,
        trust_remote_code: bool = False,
        **kwargs
    ):
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.torch_dtype = torch_dtype
        self.load_in_8bit = load_in_8bit
        self.device_map = device_map
        self.max_memory = max_memory
        self.trust_remote_code = trust_remote_code

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                **kwargs
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                load_in_8bit=self.load_in_8bit,
                device_map=self.device_map,
                max_memory=self.max_memory,
                trust_remote_code=self.trust_remote_code,
                **kwargs
            )

            if self.device_map is None:
                self.model.to(self.device)

        except Exception as e:
            raise ValueError(f"Failed to load model '{self.model_name}': {str(e)}")

    def generate(
        self,
        prompt: str,
        max_length: int = 256,
        num_return_sequences: int = 1,
        no_repeat_ngram_size: Optional[int] = None,
        early_stopping: bool = True,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        **kwargs
    ) -> str:
        self.model.eval()
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=early_stopping,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                **kwargs
            )

        output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output

    def generate_summary(self, text: str, max_length: int = 150, max_new_tokens: int = 150, **kwargs) -> str:
        """
        Generates a summary of the input text, splitting it if necessary.

        Args:
            text (str): The text to summarize.
            max_length (int): Maximum length of the summary.
            max_new_tokens (int): Maximum number of new tokens to generate.
            **kwargs: Additional keyword arguments for the generation method.

        Returns:
            str: The generated summary.
        """
        self.model.eval()

        # Split text if input length exceeds a set threshold (e.g., 1024 tokens)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        text_chunks = text_splitter.split_text(text) if len(text) > 1024 else [text]

        summaries = []
        for chunk in text_chunks:
            input_ids = self.tokenizer.encode("Summarize: " + chunk, return_tensors='pt').to(self.device)
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
            summaries.append(self.tokenizer.decode(output_ids[0], skip_special_tokens=True))

        # Combine chunk summaries into a final summary
        final_summary = " ".join(summaries)
        return final_summary

    def generate_response(self,
        prompt: str,
        max_length: int = 256,
        num_return_sequences: int = 1,
        no_repeat_ngram_size: Optional[int] = None,
        early_stopping: bool = True,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        **kwargs
    ):
        messages = [
            {"role": "system", "content": "You are an AI assistant that provides helpful answers."},
            {"role": "user", "content": f"Question:\n{prompt}"},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
        outputs = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )
        response_ids = outputs[0][input_ids.shape[-1]:]
        answer = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        return answer

    def generate_response_with_context(self, context, prompt, max_new_tokens=512, **kwargs):
        """Generate a response with context using max_new_tokens instead of max_length."""
        # Format the input
        formatted_input = f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:"

        # Tokenize input
        inputs = self.tokenizer(formatted_input, return_tensors="pt", truncation=True)
        inputs = inputs.to(self.device)

        # Generate response
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,  # Use max_new_tokens instead of max_length
            pad_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )

        # Decode and return the response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the input prompt from the response
        response = response[len(formatted_input):].strip()

        return response

    def clean_up(self):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        gc.collect()
