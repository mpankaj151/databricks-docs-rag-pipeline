"""AWS Bedrock LLM client.

Connects to AWS Bedrock's API to generate responses without
managing API keys — AWS credentials are used instead.

Setup:
    1. Configure AWS credentials (aws configure or environment variables).
    2. Request access to the model in Bedrock console.
    3. Set the region (base_url) and model name.

Provider:
    base_url is the AWS region (e.g. "us-east-1", "us-west-2").
    The model should be an Anthropic model hosted on Bedrock,
    e.g. "anthropic.claude-3-5-sonnet-latest".

Key difference from direct Anthropic:
    Bedrock authenticates via AWS credentials (IAM), not API keys.
    The request format is slightly different — uses the Bedrock agent API.

Important:
    This client is for Claude models on Bedrock. For other models,
    the API format may differ — check AWS Bedrock docs.

Optional dependency:
    boto3 is required only when using the Bedrock provider.
    Install with: pip install databricks-docs-rag-pipeline[bedrock]
"""

import json

from rag_pipeline.config import DEFAULT_STRICT_PROMPT

# Optional — loaded lazily so the package installs without boto3.
# Wrapped in a try so the module loads even if boto3 is not installed.
# The actual call will fail at __init__ time with a helpful error.
try:
    import boto3
except ImportError:
    boto3 = None


class BedrockLLM:
    """AWS Bedrock LLM client implementing the LLMClient Protocol.

    Uses AWS credentials (from aws configure or env vars) instead of API keys.
    Supports Claude models on Bedrock (recommended: anthropic.claude-3-5-sonnet-latest).

    Attributes:
        model: Bedrock model ID (e.g. "anthropic.claude-3-5-sonnet-latest").
        base_url: AWS region (e.g. "us-east-1").
        temperature: Sampling temperature.
        max_tokens: Max response tokens.
        api_key: Not used — AWS credentials authenticate instead.
        _strict_prompt: Anti-hallucination prompt template.

    Environment:
        AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
        (or ~/.aws/credentials from `aws configure`).
    """

    def __init__(
        self,
        model: str = "anthropic.claude-3-5-sonnet-latest",
        base_url: str = "us-east-1",
        temperature: float = 0.2,
        max_tokens: int = 512,
        api_key: str = "",
        strict_prompt: str = "",
    ):
        self.region = base_url  # base_url doubles as AWS region here
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._strict_prompt = strict_prompt or DEFAULT_STRICT_PROMPT

        if boto3 is None:
            raise ImportError(
                "boto3 is required for Bedrock. Install with: "
                "pip install databricks-docs-rag-pipeline[bedrock]"
            )

        # boto3 client — uses AWS credentials from environment or ~/.aws/.
        self._bedrock = boto3.client(
            service_name="bedrock-agent-runtime",
            region_name=self.region,
        )

    @property
    def strict_prompt(self) -> str:
        return self._strict_prompt

    def generate(self, prompt: str, system: str = None) -> str:
        """Send a prompt to Bedrock and return the response.

        Args:
            prompt: User's message text.
            system: Optional system instruction.

        Returns:
            Bedrock's text response.
        """
        messages = [
            {
                "role": "user",
                "content": [{"text": prompt}],
            }
        ]

        if system:
            messages.insert(0, {"role": "user", "content": [{"text": system}]})

        response = self._bedrock.invoke_model(
            modelId=self.model,
            contentBody={"messages": messages},
        )

        result = json.loads(response["body"].read())
        return result["completion"]

    def generate_with_context(
        self, question: str, context: str, strict_prompt: str = None
    ) -> str:
        """Generate a RAG-grounded answer using retrieved context.

        Args:
            question: The user's question.
            context: Concatenated retrieved chunks from FAISS.
            strict_prompt: Override (uses instance default if None).

        Returns:
            Bedrock's grounded answer.
        """
        template = strict_prompt or self._strict_prompt
        prompt = template.format(context=context, question=question)
        return self.generate(prompt)