import torch, re, json, os, logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, Optional
from .model import LLMBackend

logger = logging.getLogger(__name__)

class LocalHFBackend(LLMBackend):
    def __init__(self, hyperparams: Dict[str, Any]):
        # Accept hyperparams dict with model_name, temperature, max_new_tokens, device
        model_name = hyperparams.get("model_name", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.temperature = hyperparams.get("temperature", 0.7)
        self.max_tokens = hyperparams.get("max_new_tokens", 64)
        self.device = hyperparams.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading HuggingFace model: {model_name} on device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt: str) -> str:
        # For TinyLlama, we need to format the prompt as a chat conversation
        # TinyLlama expects: <|system|>...<|user|>...<|assistant|>

        # logger.info(f"Prompt: {prompt}")
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        response = response.strip()
        # logger.info(f"Raw LLM Response: {response}")
        
        json_match = re.search(r'\{.*\}', response)
        if json_match:
            json_str = json_match.group(0)
            json.loads(json_str)
            logger.info(f"LLM Response: {json_str}")
            return json_str
        else:
            logger.error(f"No valid JSON found in response {response}")
            return '{"action": "Monitor", "reason": "No valid JSON found in response"}'
