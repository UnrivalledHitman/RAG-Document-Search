"""Configuration module for Agentic RAG system"""

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pathlib import Path
from typing import List

# Load environment variables at module level
load_dotenv()


class Config:
    """Configuration class for RAG system"""

    # Model Configuration
    LLM_MODEL = "openai/gpt-oss-20b"

    # Document Processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    # Data paths
    DATA_FOLDER = Path("data") if Path("data").exists() else None

    # Web sources
    WEB_URLS = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
    ]

    # Default sources (fallback if no choice is made)
    DEFAULT_URLS = [
        str(DATA_FOLDER),
    ]

    @classmethod
    def get_llm(cls):
        """Initialize and return the LLM model"""
        return init_chat_model(cls.LLM_MODEL, model_provider="groq")

    @classmethod
    def get_sources(cls, source_type: str = "both") -> List[str]:
        """
        Get document sources based on user selection

        Args:
            source_type: One of "local", "web", or "both"

        Returns:
            List of source paths/URLs
        """
        sources = []

        if source_type in ["local", "both"] and cls.DATA_FOLDER:
            sources.append(str(cls.DATA_FOLDER))

        if source_type in ["web", "both"]:
            sources.extend(cls.WEB_URLS)

        # Fallback to default if nothing selected
        if not sources:
            sources = cls.DEFAULT_URLS

        return sources
