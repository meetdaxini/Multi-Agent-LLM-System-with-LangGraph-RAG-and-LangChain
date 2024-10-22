import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import gc
from typing import List, Optional, Dict


class HuggingFaceEmbedding:
    """
    Embedding model using Hugging Face transformers.
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        trust_remote_code: bool = False,
        load_in_8bit: bool = False,
        device_map: Optional[str] = None,
        max_memory: Optional[Dict] = None,
        **kwargs,
    ):
        """
        Initializes the Hugging Face embedding model.

        Args:
            model_name (str): The name of the Hugging Face model to use.
            device (Optional[str]): The device to load the model on (e.g., 'cpu', 'cuda').
            trust_remote_code (bool): Whether to trust remote code from the model repository.
            load_in_8bit (bool): Whether to load the model in 8-bit precision.
            device_map (Optional[str]): Device map for model parallelism.
            max_memory (Optional[Dict]): Maximum memory allocation for devices.
            **kwargs: Additional keyword arguments for model/tokenizer initialization.
        """
        self.supported_models = [
            "nvidia/NV-Embed-v2",
            "sentence-transformers/all-MiniLM-L6-v2",
            "dunzhang/stella_en_1.5B_v5",
            "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
            "dunzhang/stella_en_400M_v5",
            # TODO: Add more supported models here
        ]

        if model_name not in self.supported_models:
            raise ValueError(
                f"Model '{model_name}' is not supported. Supported models are: {self.supported_models}"
            )

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.trust_remote_code = trust_remote_code
        self.load_in_8bit = load_in_8bit
        self.device_map = device_map
        self.max_memory = max_memory

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=self.trust_remote_code, **kwargs
            )

            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                load_in_8bit=self.load_in_8bit,
                device_map=self.device_map,
                max_memory=self.max_memory,
                **kwargs,
            )
            if not self.load_in_8bit:
                self.model = self.model.to(self.device)

        except Exception as e:
            raise ValueError(f"Failed to load model '{self.model_name}': {str(e)}")

    def embed(
        self,
        texts: List[str],
        instruction: Optional[str] = None,
        max_length: Optional[int] = None,
        batch_size: int = 32,
        **kwargs,
    ) -> torch.Tensor:
        """
        Embeds a list of texts.

        Args:
            texts (List[str]): The texts to embed.
            instruction (Optional[str]): Instruction text for models that support it.
            max_length (Optional[int]): Maximum sequence length for tokenization.
            batch_size (int): Batch size for processing texts.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Normalized embeddings.
        """
        all_embeddings = []

        if not hasattr(self.model, "encode"):
            # Define default embedding extraction for models without 'encode'
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use the last hidden state as embeddings
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                    all_embeddings.append(embeddings.cpu())
                del embeddings
                torch.cuda.empty_cache()
                gc.collect()
        else:
            # Use the model's 'encode' method
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                embeddings = self.model.encode(
                    batch_texts,
                    instruction=instruction,
                    max_length=max_length,
                    batch_size=batch_size,
                    **kwargs,
                )
                embeddings = F.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu())
                del embeddings
                torch.cuda.empty_cache()
                gc.collect()

        return torch.cat(all_embeddings, dim=0)

    def clean_up(self):
        """
        Cleans up resources to free memory.
        """
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        gc.collect()
