"""Ollama LLM client."""
import requests
from typing import Optional


class OllamaLLM:
    """Ollama LLM client."""
    
    def __init__(
        self,
        model: str = "qwen3.5:cloud",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.2,
        max_tokens: int = 512
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def generate(self, prompt: str, system: str = None) -> str:
        """Generate response from LLM."""
        messages = []
        
        if system:
            messages.append({"role": "system", "content": system})
        
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }
        
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        
        return result["message"]["content"]
    
    def generate_with_context(self, question: str, context: str) -> str:
        """Generate answer using retrieved context."""
        prompt = f"""Context:
{context}

Question: {question}
Answer:"""
        
        return self.generate(prompt)