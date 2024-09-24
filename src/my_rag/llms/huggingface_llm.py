import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base import BaseLLM
import gc
from typing import Optional, Dict, Any

class HuggingFaceLLM(BaseLLM):
    """
    LLM implementation using Hugging Face transformers.
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = torch.float32,
        load_in_8bit: bool = False,
        device_map: Optional[str] = None,
        max_memory: Optional[Dict[str, str]] = None,
        trust_remote_code: bool = False,
        **kwargs
    ):
        """
        Initializes the Hugging Face LLM.

        Args:
            model_name (str): The name or path of the pre-trained model.
            device (Optional[str]): Device to run the model on ('cpu', 'cuda', etc.).
            torch_dtype (Optional[torch.dtype]): Data type for model weights (e.g., torch.float32).
            load_in_8bit (bool): Whether to load the model in 8-bit precision.
            device_map (Optional[str]): Device map for model parallelism ('auto', None, or custom mapping).
            max_memory (Optional[Dict[str, str]]): Maximum memory allocation per device.
            trust_remote_code (bool): Whether to trust custom code from the model repository.
            **kwargs: Additional keyword arguments for model/tokenizer initialization.

        Raises:
            ValueError: If the model fails to load.
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.torch_dtype = torch_dtype
        self.load_in_8bit = load_in_8bit
        self.device_map = device_map
        self.max_memory = max_memory
        self.trust_remote_code = trust_remote_code

        try:
            # Initialize the tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                **kwargs
            )

            # Initialize the model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                load_in_8bit=self.load_in_8bit,
                device_map=self.device_map,
                max_memory=self.max_memory,
                trust_remote_code=self.trust_remote_code,
                **kwargs
            )

            # Move the model to the specified device
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
        """
        Generates text based on the input prompt.

        Args:
            prompt (str): The input prompt for the language model.
            max_length (int): Maximum length of the generated text.
            num_return_sequences (int): Number of sequences to return.
            no_repeat_ngram_size (Optional[int]): Size of ngrams to avoid repeating.
            early_stopping (bool): Whether to stop early when the end-of-sequence token is generated.
            temperature (float): Sampling temperature.
            top_k (int): The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (float): Cumulative probability for nucleus sampling.
            **kwargs: Additional keyword arguments for the generation method.

        Returns:
            str: The generated text.
        """
        self.model.eval()  # Set model to evaluation mode

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


    def generate_response(        self,
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

    def clean_up(self):
        """
        Cleans up resources to free memory.
        """
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        gc.collect()
