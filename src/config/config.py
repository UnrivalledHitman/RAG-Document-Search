"""Configuration module for Agentic RAG system"""

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Load environment variables at module level
load_dotenv()


class Config:
    """Configuration class for RAG system"""

    # Model Configuration
    LLM_MODEL = "openai/gpt-oss-20b"

    # Document Processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    # Default URLs
    DEFAULT_URLS = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
    ]

    @classmethod
    def get_llm(cls):
        """Initialize and return the LLM model"""
        return init_chat_model(cls.LLM_MODEL, model_provider="groq")
