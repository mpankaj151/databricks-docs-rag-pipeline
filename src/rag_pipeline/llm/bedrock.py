"""AWS Bedrock LLM client.

Uses AWS boto3 + Bedrock Converse API.
Reads AWS credentials from environment or boto3 default credential chain.
"""
from rag_pipeline.llm._base import LLMClient
from rag_pipeline.config import DEFAULT_STRICT_PROMPT

try:
    import boto3
    from botocore.config import Config as BotoConfig
except ImportError:
    boto3 = None
    BotoConfig = None


class BedrockLLM(LLMClient):
    """AWS Bedrock LLM client.

    Supports Anthropic models on Bedrock (claude-3-5-sonnet, etc.)
    and other Bedrock models that implement the Converse API.

    Required env vars (or boto3 default chain):
      AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
    """

    def __init__(
        self,
        model: str = "anthropic.claude-3-5-sonnet-latest",
        region: str = "us-east-1",
        temperature: float = 0.2,
        max_tokens: int = 512,
        api_key: str = "",
        strict_prompt: str = "",
    ):
        self.model = model
        self.region = region
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._strict_prompt = strict_prompt or DEFAULT_STRICT_PROMPT

    @property
    def strict_prompt(self) -> str:
        return self._strict_prompt

    def generate(self, prompt: str, system: str = None) -> str:
        if boto3 is None:
            raise ImportError(
                "boto3 is required for Bedrock. Install: pip install boto3"
            )

        boto_config = BotoConfig(region_name=self.region)
        bedrock = boto3.client("bedrock-runtime", config=boto_config)

        messages = [{"role": "user", "content": [{"text": prompt}]}]
        if system:
            messages.insert(0, {"role": "user", "content": [{"text": system}]})

        payload = {
            "modelId": self.model,
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": self.max_tokens,
                "temperature": self.temperature,
            },
        }

        response = bedrock.converse(**payload)
        return response["output"]["message"]["content"][0]["text"]

    def generate_with_context(
        self, question: str, context: str, strict_prompt: str = None
    ) -> str:
        template = strict_prompt or self._strict_prompt
        prompt = template.format(context=context, question=question)
        return self.generate(prompt)