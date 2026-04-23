"""
Step 6: LLM GENERATION
----------------------
The final part of RAG is Generation. We take the user's original question, AND the 
relevant document chunks we found in Step 5 (Retrieval), and package them together 
into a "Prompt".

We send this prompt to a Large Language Model (like GPT-4, LLaMA, or GLM-5.1) with 
very specific instructions: "Answer the question, but ONLY use the information I just 
provided in the context chunks."

This solves the "hallucination" problem in AI. The AI doesn't have to rely on its 
unreliable memory; it acts like an open-book test taker reading the documents we provided.
"""
import requests
import json
from src.config import get_config


class LLMError(Exception):
    pass


class OllamaLLM:
    """
    Connects to an AI model to generate answers.
    Supports local models (via Ollama) or cloud models (via OpenRouter API).
    """
    
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
        
        # Temperature controls creativity. In RAG, we want facts, not creativity.
        # So we usually set temperature very low (e.g. 0.0 or 0.2).
        self.temperature = temperature if temperature is not None else config.llm_temperature
        self.max_tokens = max_tokens or config.llm_max_tokens
        self.api_key = api_key or config.llm_api_key
        
        self.is_openrouter = bool(self.api_key)
        if self.is_openrouter:
            self.base_url = "https://openrouter.ai/api/v1"
    
    def check_connection(self) -> bool:
        """Check if the AI service is running and accessible."""
        try:
            if self.is_openrouter:
                headers = {"Authorization": f"Bearer {self.api_key}"}
                resp = requests.get(f"{self.base_url}/auth/key", headers=headers, timeout=5)
                return resp.status_code == 200
            else:
                resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
                if resp.status_code == 200:
                    models = [m["name"] for m in resp.json().get("models", [])]
                    if self.model not in models:
                        print(f"Warning: Model {self.model} not pulled in Ollama. Run: ollama pull {self.model}")
                return resp.status_code == 200
        except requests.RequestException:
            return False
    
    def generate(self, prompt: str, system: str = None) -> str:
        """Send a raw prompt to the AI and return its response."""
        messages = []
        
        # The system prompt tells the AI its overarching persona/rules
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
            response = requests.post(url, json=payload, headers=headers, timeout=120)
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
        """
        The core of RAG. We create a prompt that contains BOTH the retrieved documents (context)
        AND the user's question, and force the AI to answer using ONLY the context.
        """
        # Strict instructions prevent the AI from making things up (hallucinating)
        system_prompt = """You are a helpful assistant that answers questions about Databricks Delta Lake and Lakeflow Declarative Pipeline using ONLY the provided documentation excerpts. If the answer cannot be found in the excerpts, say you don't know."""
        
        # The prompt structure: Context first, then the question
        prompt = f"""Context:
{context}

Question: {question}
Answer:"""
        
        return self.generate(prompt, system=system_prompt)


def get_llm() -> OllamaLLM:
    """Helper to get a configured LLM instance."""
    return OllamaLLM()
