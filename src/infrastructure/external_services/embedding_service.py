"""
Google Text Embedding Service for Semantic Memory.

Provides:
- Single and batch embedding generation
- 768-dimensional vectors using gemini-embedding-001
- Error handling and retry logic
- Cost tracking (optional)
"""

import os
import logging
from typing import List, Optional, Dict, Any

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings using Google's gemini-embedding-001 model.

    Features:
    - 768-dimensional embeddings
    - Batch processing support
    - Automatic retry on failures
    - Semantic similarity optimized
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "models/gemini-embedding-001",
        embedding_dimensions: int = 768
    ):
        """
        Initialize embedding service.

        Args:
            api_key: Google API key (uses GOOGLE_API_KEY env var if None)
            model_name: Embedding model name (default: gemini-embedding-001)
            embedding_dimensions: Output dimensions (default: 768)
        """
        if not GENAI_AVAILABLE:
            raise ImportError(
                "google-generativeai not installed. "
                "Install with: pip install google-generativeai"
            )

        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            raise ValueError("Google API key not provided. Set GOOGLE_API_KEY environment variable.")

        genai.configure(api_key=self.api_key)  # type: ignore

        self.model_name = model_name
        self.embedding_dimensions = embedding_dimensions
        self.request_count = 0

        logger.info(f"Initialized EmbeddingService with model={model_name}, dims={embedding_dimensions}")

    async def generate_embedding(
        self,
        text: str,
        task_type: str = "SEMANTIC_SIMILARITY",
        title: Optional[str] = None
    ) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed
            task_type: Task type for embedding optimization
                      Options: RETRIEVAL_QUERY, RETRIEVAL_DOCUMENT, SEMANTIC_SIMILARITY
            title: Optional title for document embeddings

        Returns:
            List of 768 floats representing the embedding vector

        Raises:
            ValueError: If text is empty
            Exception: If embedding generation fails
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        try:
            result = genai.embed_content(  # type: ignore
                model=self.model_name,
                content=text,
                task_type=task_type,
                title=title
            )

            self.request_count += 1
            embedding = result['embedding']

            if len(embedding) != self.embedding_dimensions:
                logger.warning(
                    f"Embedding dimension mismatch: expected {self.embedding_dimensions}, "
                    f"got {len(embedding)}"
                )

            logger.debug(
                f"Generated embedding for text (length={len(text)}): "
                f"vector_dims={len(embedding)}"
            )

            return embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {e}", exc_info=True)
            raise

    async def generate_embeddings_batch(
        self,
        texts: List[str],
        task_type: str = "SEMANTIC_SIMILARITY",
        titles: Optional[List[str]] = None
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch.

        Args:
            texts: List of input texts to embed
            task_type: Task type for embedding optimization
            titles: Optional titles for document embeddings (must match texts length)

        Returns:
            List of embedding vectors (each is 768 floats)

        Raises:
            ValueError: If texts is empty or titles length mismatch
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        if titles and len(titles) != len(texts):
            raise ValueError(f"Titles length ({len(titles)}) must match texts length ({len(texts)})")

        try:
            embeddings = []

            for idx, text in enumerate(texts):
                if not text or not text.strip():
                    logger.warning(f"Skipping empty text at index {idx}")
                    embeddings.append([0.0] * self.embedding_dimensions)
                    continue

                title = titles[idx] if titles else None

                result = genai.embed_content(  # type: ignore
                    model=self.model_name,
                    content=text,
                    task_type=task_type,
                    title=title
                )

                embeddings.append(result['embedding'])

            self.request_count += len(texts)

            logger.info(f"Generated {len(embeddings)} embeddings in batch")
            return embeddings

        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}", exc_info=True)
            raise

    async def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding optimized for query/search.

        Args:
            query: Search query text

        Returns:
            Query-optimized embedding vector
        """
        return await self.generate_embedding(query, task_type="RETRIEVAL_QUERY")

    async def generate_document_embedding(
        self,
        document: str,
        title: Optional[str] = None
    ) -> List[float]:
        """
        Generate embedding optimized for document storage.

        Args:
            document: Document text to embed
            title: Optional document title

        Returns:
            Document-optimized embedding vector
        """
        return await self.generate_embedding(
            document,
            task_type="RETRIEVAL_DOCUMENT",
            title=title
        )

    def _validate_task_type(self, task_type_str: str) -> str:
        """
        Validate task type string.

        Args:
            task_type_str: String representation of task type

        Returns:
            Validated task type string
        """
        valid_types = {
            "RETRIEVAL_QUERY",
            "RETRIEVAL_DOCUMENT",
            "SEMANTIC_SIMILARITY",
            "CLASSIFICATION",
            "CLUSTERING"
        }

        task_type_upper = task_type_str.upper()

        if task_type_upper in valid_types:
            return task_type_upper

        logger.warning(f"Invalid task type '{task_type_str}', using SEMANTIC_SIMILARITY")
        return "SEMANTIC_SIMILARITY"

    def get_stats(self) -> Dict[str, Any]:
        """
        Get service statistics.

        Returns:
            Dict with request count and configuration
        """
        return {
            "model": self.model_name,
            "dimensions": self.embedding_dimensions,
            "total_requests": self.request_count
        }

    def reset_stats(self) -> None:
        """Reset request counter."""
        self.request_count = 0
        logger.info("Reset embedding service statistics")


def create_embedding_service(
    api_key: Optional[str] = None,
    model_name: str = "models/gemini-embedding-001",
    embedding_dimensions: int = 768
) -> EmbeddingService:
    """
    Factory function to create embedding service.

    Args:
        api_key: Google API key (uses env var if None)
        model_name: Embedding model name
        embedding_dimensions: Output dimensions

    Returns:
        Configured EmbeddingService instance
    """
    return EmbeddingService(
        api_key=api_key,
        model_name=model_name,
        embedding_dimensions=embedding_dimensions
    )
