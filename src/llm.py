"""LLM integration via Ollama or OpenRouter"""
import requests
import json
from src.config import get_config


class LLMError(Exception):
    pass


class OllamaLLM:
    """Wrapper for Ollama API, falls back to OpenRouter if configured"""
    
    def __init__(
        self,
        model: str = None,
        base_url: str = None,
        temperature: float = None,
        max_tokens: int = None,
        api_key: str = None
    ):
        config = get_config()
        self.model = model or config.llm_model
        self.base_url = base_url or config.llm_base_url
        self.temperature = temperature if temperature is not None else config.llm_temperature
        self.max_tokens = max_tokens or config.llm_max_tokens
        self.api_key = api_key or config.llm_api_key
        
        # Determine if we're using OpenRouter based on API key
        self.is_openrouter = bool(self.api_key)
        if self.is_openrouter:
            self.base_url = "https://openrouter.ai/api/v1"
    
    def check_connection(self) -> bool:
        """Check if LLM backend is available"""
        try:
            if self.is_openrouter:
                # Basic check for openrouter without incurring cost
                headers = {"Authorization": f"Bearer {self.api_key}"}
                resp = requests.get(f"{self.base_url}/auth/key", headers=headers, timeout=5)
                return resp.status_code == 200
            else:
                # Check Ollama tags
                resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
                if resp.status_code == 200:
                    models = [m["name"] for m in resp.json().get("models", [])]
                    if self.model not in models:
                        print(f"Warning: Model {self.model} not pulled in Ollama. Run: ollama pull {self.model}")
                return resp.status_code == 200
        except requests.RequestException:
            return False
    
    def generate(self, prompt: str, system: str = None) -> str:
        """Generate response from LLM"""
        messages = []
        
        if system:
            messages.append({"role": "system", "content": system})
        
        messages.append({"role": "user", "content": prompt})
        
        headers = {}
        if self.is_openrouter:
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["HTTP-Referer"] = "https://github.com/rag-practice"
            url = f"{self.base_url}/chat/completions"
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        else:
            url = f"{self.base_url}/api/chat"
            payload = {
                "model": self.model,
                "messages": messages,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
                "stream": False
            }
            
        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            
            if self.is_openrouter:
                return result["choices"][0]["message"]["content"]
            else:
                return result["message"]["content"]
                
        except requests.RequestException as e:
            msg = str(e)
            if hasattr(e, 'response') and e.response is not None:
                msg += f" - Response: {e.response.text}"
            raise LLMError(f"LLM API Error: {msg}")
    
    def generate_with_context(self, question: str, context: str) -> str:
        """Generate answer given retrieved context"""
        system_prompt = """You are a helpful assistant that answers questions about Databricks Delta Lake and Lakeflow Declarative Pipeline using ONLY the provided documentation excerpts. If the answer cannot be found in the excerpts, say you don't know."""
        
        prompt = f"""Context:
{context}

Question: {question}
Answer:"""
        
        return self.generate(prompt, system=system_prompt)


def get_llm() -> OllamaLLM:
    """Get LLM instance"""
    return OllamaLLM()
