"""
Qdrant Vector Database Client

Manages episodic memory storage using Qdrant vector database.
Stores conversation events, transactions, and user interactions with vector embeddings.
"""

import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

try:
    from qdrant_client import QdrantClient as QdrantClientSDK
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
        Range,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClientSDK = None


@dataclass
class EpisodicMemory:
    """Represents an episodic memory entry"""
    id: str
    user_id: str
    session_id: str
    event_type: str  # transaction|inquiry|complaint|conversation
    content: str
    timestamp: datetime
    metadata: Dict[str, Any]
    importance: float  # 0.0-1.0
    similarity_score: Optional[float] = None


class QdrantClient:
    """
    Qdrant vector database client for episodic memory storage

    Handles:
    - Connection management
    - Collection creation
    - Vector storage and retrieval
    - Metadata filtering
    - Memory cleanup
    """

    def __init__(self, host: str = "localhost", port: int = 6333, api_key: Optional[str] = None):
        self.host = host
        self.port = port
        self.api_key = api_key
        self.client: Optional[Any] = None  # Use Any to avoid type checking issues
        self.collection_name = "episodic_memories"
        self.vector_size = 768  # Text-Embedding-004

    def connect(self) -> bool:
        """
        Initialize connection to Qdrant

        Returns:
            bool: True if connection successful
        """
        if not QDRANT_AVAILABLE:
            raise ImportError("qdrant-client not installed. Run: pip install qdrant-client")

        try:
            if QdrantClientSDK is None:
                raise ImportError("QdrantClient not available")

            self.client = QdrantClientSDK(
                host=self.host,
                port=self.port,
                api_key=self.api_key,
                timeout=10
            )

            # Test connection
            if self.client:
                collections = self.client.get_collections()
            return True

        except Exception as e:
            raise ConnectionError(f"Failed to connect to Qdrant: {str(e)}")

    def create_collection(self, collection_name: Optional[str] = None, vector_size: int = 768) -> bool:
        """
        Create Qdrant collection for episodic memories

        Args:
            collection_name: Name of collection (default: episodic_memories)
            vector_size: Dimension of vectors (default: 768)

        Returns:
            bool: True if created or already exists
        """
        if not self.client:
            raise RuntimeError("Not connected to Qdrant. Call connect() first.")

        name = collection_name or self.collection_name

        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            if any(c.name == name for c in collections):
                return True

            # Create collection
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )

            return True

        except Exception as e:
            raise RuntimeError(f"Failed to create collection: {str(e)}")

    async def store_episodic_memory(
        self,
        user_id: str,
        session_id: str,
        content: str,
        embedding: List[float],
        event_type: str = "conversation",
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.7
    ) -> str:
        """
        Store episodic memory in Qdrant

        Args:
            user_id: User identifier
            session_id: Session identifier
            content: Memory content text
            embedding: 768-dim vector embedding
            event_type: Type of event (transaction|inquiry|complaint|conversation)
            metadata: Additional metadata
            importance: Importance score 0.0-1.0

        Returns:
            str: Memory ID
        """
        if not self.client:
            raise RuntimeError("Not connected to Qdrant. Call connect() first.")

        if len(embedding) != self.vector_size:
            raise ValueError(f"Embedding must be {self.vector_size} dimensions")

        memory_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        payload = {
            "user_id": user_id,
            "session_id": session_id,
            "event_type": event_type,
            "content": content,
            "timestamp": timestamp,
            "importance": importance,
            "metadata": metadata or {}
        }

        point = PointStruct(
            id=memory_id,
            vector=embedding,
            payload=payload
        )

        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )

        return memory_id

    async def search_episodic_memories(
        self,
        user_id: str,
        query_embedding: List[float],
        top_k: int = 5,
        event_type: Optional[str] = None,
        days_back: Optional[int] = None,
        min_importance: float = 0.0
    ) -> List[EpisodicMemory]:
        """
        Search episodic memories using vector similarity

        Args:
            user_id: User identifier
            query_embedding: Query vector
            top_k: Number of results to return
            event_type: Filter by event type
            days_back: Only return memories from last N days
            min_importance: Minimum importance threshold

        Returns:
            List of EpisodicMemory objects with similarity scores
        """
        if not self.client:
            raise RuntimeError("Not connected to Qdrant. Call connect() first.")

        if len(query_embedding) != self.vector_size:
            raise ValueError(f"Query embedding must be {self.vector_size} dimensions")

        # Build filter
        must_conditions: list = [
            FieldCondition(
                key="user_id",
                match=MatchValue(value=user_id)
            )
        ]

        if event_type:
            must_conditions.append(
                FieldCondition(
                    key="event_type",
                    match=MatchValue(value=event_type)
                )
            )

        if days_back:
            cutoff_timestamp = (datetime.utcnow() - timedelta(days=days_back)).timestamp()
            must_conditions.append(
                FieldCondition(
                    key="timestamp",
                    range=Range(gte=cutoff_timestamp)
                )
            )

        if min_importance > 0.0:
            must_conditions.append(
                FieldCondition(
                    key="importance",
                    range=Range(gte=min_importance)
                )
            )

        search_filter = Filter(must=must_conditions)

        # Search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter=search_filter,
            limit=top_k
        )

        # Convert to EpisodicMemory objects
        memories = []
        for result in results:
            payload = result.payload
            memory = EpisodicMemory(
                id=result.id,
                user_id=payload["user_id"],
                session_id=payload["session_id"],
                event_type=payload["event_type"],
                content=payload["content"],
                timestamp=datetime.fromisoformat(payload["timestamp"]),
                metadata=payload.get("metadata", {}),
                importance=payload["importance"],
                similarity_score=result.score
            )
            memories.append(memory)

        return memories

    async def get_recent_episodes(
        self,
        user_id: str,
        limit: int = 10,
        days: int = 30,
        event_type: Optional[str] = None
    ) -> List[EpisodicMemory]:
        """
        Get recent episodic memories (time-based, no vector search)

        Args:
            user_id: User identifier
            limit: Max number of memories
            days: Look back N days
            event_type: Filter by event type

        Returns:
            List of recent EpisodicMemory objects
        """
        if not self.client:
            raise RuntimeError("Not connected to Qdrant. Call connect() first.")

        # Build filter
        cutoff_timestamp = (datetime.utcnow() - timedelta(days=days)).timestamp()
        must_conditions: list = [
            FieldCondition(
                key="user_id",
                match=MatchValue(value=user_id)
            ),
            FieldCondition(
                key="timestamp",
                range=Range(gte=cutoff_timestamp)
            )
        ]

        if event_type:
            must_conditions.append(
                FieldCondition(
                    key="event_type",
                    match=MatchValue(value=event_type)
                )
            )

        search_filter = Filter(must=must_conditions)

        # Scroll through results
        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=search_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )

        # Convert to EpisodicMemory objects
        memories = []
        for result in results:
            payload = result.payload
            memory = EpisodicMemory(
                id=result.id,
                user_id=payload["user_id"],
                session_id=payload["session_id"],
                event_type=payload["event_type"],
                content=payload["content"],
                timestamp=datetime.fromisoformat(payload["timestamp"]),
                metadata=payload.get("metadata", {}),
                importance=payload["importance"]
            )
            memories.append(memory)

        # Sort by timestamp descending
        memories.sort(key=lambda m: m.timestamp, reverse=True)

        return memories

    async def delete_old_episodes(
        self,
        user_id: str,
        days: int = 90
    ) -> int:
        """
        Delete episodic memories older than N days

        Args:
            user_id: User identifier
            days: Delete memories older than this

        Returns:
            int: Number of memories deleted
        """
        if not self.client:
            raise RuntimeError("Not connected to Qdrant. Call connect() first.")

        cutoff_timestamp = (datetime.utcnow() - timedelta(days=days)).timestamp()

        # Build filter
        must_conditions: list = [
            FieldCondition(
                key="user_id",
                match=MatchValue(value=user_id)
            ),
            FieldCondition(
                key="timestamp",
                range=Range(lt=cutoff_timestamp)
            )
        ]

        search_filter = Filter(must=must_conditions)

        # Delete
        result = self.client.delete(
            collection_name=self.collection_name,
            points_selector=search_filter
        )

        return result.deleted if hasattr(result, 'deleted') else 0

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection

        Returns:
            Dict with collection statistics
        """
        if not self.client:
            raise RuntimeError("Not connected to Qdrant. Call connect() first.")

        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "vectors_count": collection_info.vectors_count,
                "points_count": collection_info.points_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "status": collection_info.status
            }
        except Exception as e:
            return {"error": str(e)}

    def close(self):
        """Close connection to Qdrant"""
        if self.client:
            self.client.close()
            self.client = None
