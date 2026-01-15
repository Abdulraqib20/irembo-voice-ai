"""
Hybrid Memory Service

Orchestrates retrieval from all three memory systems:
1. Procedural (Redis): Recent context, active workflows
2. Semantic (PostgreSQL): Preferences, goals, facts
3. Episodic (Qdrant): Past events, transactions, conversations

Combines and ranks results for optimal LLM context.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# Import memory services
from src.infrastructure.external_services.redis_cache_service import (
    RedisCacheService,
    ConversationContext
)
from src.infrastructure.external_services.qdrant_client import (
    QdrantClient,
    EpisodicMemory
)
from src.infrastructure.database.repositories.memory_repository import MemoryRepository
from src.infrastructure.database.models import UserMemory


@dataclass
class HybridMemoryResult:
    """Combined memory retrieval result"""
    procedural: Optional[ConversationContext]
    semantic: List[Tuple[UserMemory, float]]  # (memory, similarity_score)
    episodic: List[EpisodicMemory]
    ranked_results: List[Dict[str, Any]]
    total_memories: int


class HybridMemoryService:
    """
    Hybrid memory service orchestrating all three memory systems

    Memory Systems:
    - Procedural (Redis): Immediate context (last 5-10 turns, active slots)
    - Semantic (PostgreSQL): Long-term preferences, goals, facts
    - Episodic (Qdrant): Past events, transactions, conversation history

    Ranking Strategy:
    - Procedural: Always included (highest priority)
    - Semantic: Ranked by similarity * confidence
    - Episodic: Ranked by similarity * recency * importance
    """

    def __init__(
        self,
        redis_service: RedisCacheService,
        qdrant_client: Optional[QdrantClient],  # Made optional - episodic memory disabled if None
        memory_repository: MemoryRepository,
        embedding_service
    ):
        self.redis = redis_service
        self.qdrant = qdrant_client
        self.memory_repo = memory_repository
        self.embedding_service = embedding_service

        # Ranking weights
        self.weights = {
            "procedural": 1.0,  # Always included
            "semantic_similarity": 0.4,
            "semantic_confidence": 0.3,
            "episodic_similarity": 0.3,
            "episodic_recency": 0.2,
            "episodic_importance": 0.3
        }

    async def retrieve_all_memories(
        self,
        user_id: str,
        session_id: str,
        query: str,
        top_k_per_type: int = 5,
        include_procedural: bool = True,
        include_semantic: bool = True,
        include_episodic: bool = True
    ) -> HybridMemoryResult:
        """
        Retrieve memories from all three systems

        Args:
            user_id: User identifier
            session_id: Session identifier
            query: Search query
            top_k_per_type: Number of results per memory type
            include_procedural: Include Redis cache
            include_semantic: Include PostgreSQL semantic memories
            include_episodic: Include Qdrant episodic memories

        Returns:
            HybridMemoryResult with combined memories
        """

        # Generate query embedding (for semantic and episodic)
        query_embedding = None
        if include_semantic or include_episodic:
            query_embedding = await self.embedding_service.generate_query_embedding(query)

        # Retrieve from all systems in parallel
        procedural = None
        semantic = []
        episodic = []

        if include_procedural and self.redis:
            procedural = self._retrieve_procedural(user_id, session_id)

        if include_semantic and query_embedding:
            semantic = await self._retrieve_semantic(user_id, query_embedding, top_k_per_type)

        if include_episodic and query_embedding and self.qdrant:
            episodic = await self._retrieve_episodic(user_id, query_embedding, top_k_per_type)

        # Rank and merge results
        ranked_results = self.rank_and_merge(procedural, semantic, episodic)

        total_memories = len(ranked_results)
        if procedural:
            total_memories += 1  # Count procedural context as 1

        return HybridMemoryResult(
            procedural=procedural,
            semantic=semantic,
            episodic=episodic,
            ranked_results=ranked_results,
            total_memories=total_memories
        )

    def _retrieve_procedural(
        self,
        user_id: str,
        session_id: str
    ) -> Optional[ConversationContext]:
        """
        Retrieve procedural memory from Redis (synchronous)

        Returns:
            ConversationContext or None
        """
        try:
            if not self.redis:
                return None
            context = self.redis.get_conversation_context(user_id, session_id)
            return context
        except Exception as e:
            print(f"Error retrieving procedural memory: {e}")
            return None

    async def _retrieve_semantic(
        self,
        user_id: str,
        query_embedding: List[float],
        top_k: int
    ) -> List[Tuple[UserMemory, float]]:
        """
        Retrieve semantic memories from PostgreSQL

        Returns:
            List of (UserMemory, similarity_score) tuples
        """
        try:
            # semantic_search_memories returns List[Tuple[UserMemory, float]]
            memory_tuples = await self.memory_repo.semantic_search_memories(
                user_id=user_id,
                query_embedding=query_embedding,
                top_k=top_k,
                min_similarity=0.7
            )
            return memory_tuples
        except Exception as e:
            print(f"Error retrieving semantic memories: {e}")
            return []

    async def _retrieve_episodic(
        self,
        user_id: str,
        query_embedding: List[float],
        top_k: int
    ) -> List[EpisodicMemory]:
        """
        Retrieve episodic memories from Qdrant

        Returns:
            List of EpisodicMemory objects
        """
        if not self.qdrant:
            return []
        try:
            memories = await self.qdrant.search_episodic_memories(
                user_id=user_id,
                query_embedding=query_embedding,
                top_k=top_k,
                days_back=90,  # Last 3 months
                min_importance=0.5
            )
            return memories
        except Exception as e:
            print(f"Error retrieving episodic memories: {e}")
            return []

    def rank_and_merge(
        self,
        procedural: Optional[ConversationContext],
        semantic: List[Tuple[UserMemory, float]],
        episodic: List[EpisodicMemory]
    ) -> List[Dict[str, Any]]:
        """
        Rank and merge memories from all systems

        Ranking:
        - Procedural: Always first (most recent context)
        - Semantic: similarity * confidence
        - Episodic: similarity * recency_weight * importance

        Args:
            procedural: Redis context
            semantic: List of (memory, similarity) from PostgreSQL
            episodic: List of memories from Qdrant

        Returns:
            Sorted list of memory dicts with scores
        """
        ranked = []

        # Add procedural (always highest priority)
        if procedural and procedural.recent_messages:
            ranked.append({
                "source": "procedural",
                "type": "conversation_context",
                "content": {
                    "recent_messages": [
                        {"role": msg.role, "content": msg.content, "timestamp": msg.timestamp}
                        for msg in procedural.recent_messages[-5:]  # Last 5 messages
                    ],
                    "active_slots": procedural.active_slots,
                    "detected_intent": procedural.detected_intent
                },
                "score": 1.0,  # Highest priority
                "timestamp": procedural.last_updated
            })

        # Add semantic memories with scores
        for memory, similarity in semantic:
            score = (
                similarity * self.weights["semantic_similarity"] +
                memory.confidence_score * self.weights["semantic_confidence"]
            )

            ranked.append({
                "source": "semantic",
                "type": memory.memory_type,
                "content": memory.content,
                "score": score,
                "similarity": similarity,
                "confidence": memory.confidence_score,
                "timestamp": memory.created_at.isoformat() if hasattr(memory.created_at, 'isoformat') else str(memory.created_at)
            })

        # Add episodic memories with scores
        for memory in episodic:
            # Calculate recency weight (more recent = higher score)
            days_ago = (datetime.utcnow() - memory.timestamp).days
            recency_weight = max(0.1, 1.0 - (days_ago / 90))  # Decay over 90 days

            score = (
                (memory.similarity_score or 0.7) * self.weights["episodic_similarity"] +
                recency_weight * self.weights["episodic_recency"] +
                memory.importance * self.weights["episodic_importance"]
            )

            ranked.append({
                "source": "episodic",
                "type": memory.event_type,
                "content": memory.content,
                "score": score,
                "similarity": memory.similarity_score,
                "importance": memory.importance,
                "timestamp": memory.timestamp.isoformat(),
                "metadata": memory.metadata
            })

        # Sort by score descending (procedural always first due to score 1.0)
        ranked.sort(key=lambda x: x["score"], reverse=True)

        return ranked

    def format_context_for_llm(
        self,
        memories: HybridMemoryResult,
        max_context_length: int = 2000
    ) -> str:
        """
        Format hybrid memories into LLM-friendly context string

        Args:
            memories: HybridMemoryResult object
            max_context_length: Maximum characters for context

        Returns:
            Formatted context string
        """
        context_parts = []
        current_length = 0

        # Always include procedural context first
        if memories.procedural and memories.procedural.recent_messages:
            procedural_text = "## Recent Conversation:\n"
            for msg in memories.procedural.recent_messages[-5:]:
                procedural_text += f"{msg.role.upper()}: {msg.content}\n"

            if memories.procedural.active_slots:
                procedural_text += f"\nActive Context: {memories.procedural.active_slots}\n"

            context_parts.append(procedural_text)
            current_length += len(procedural_text)

        # Add top-ranked semantic and episodic memories
        if memories.ranked_results:
            semantic_memories = []
            episodic_memories = []

            for item in memories.ranked_results:
                if item["source"] == "procedural":
                    continue  # Already handled
                elif item["source"] == "semantic":
                    semantic_memories.append(item)
                elif item["source"] == "episodic":
                    episodic_memories.append(item)

            # Add semantic memories
            if semantic_memories:
                semantic_text = "\n## User Preferences & Information:\n"
                for mem in semantic_memories[:3]:  # Top 3
                    semantic_text += f"- {mem['content']} (confidence: {mem['confidence']:.2f})\n"

                if current_length + len(semantic_text) < max_context_length:
                    context_parts.append(semantic_text)
                    current_length += len(semantic_text)

            # Add episodic memories
            if episodic_memories:
                episodic_text = "\n## Relevant Past Events:\n"
                for mem in episodic_memories[:3]:  # Top 3
                    episodic_text += f"- {mem['content']} ({mem['type']})\n"

                if current_length + len(episodic_text) < max_context_length:
                    context_parts.append(episodic_text)
                    current_length += len(episodic_text)

        return "\n".join(context_parts)

    async def store_to_appropriate_memory(
        self,
        user_id: str,
        session_id: str,
        memory_type: str,
        content: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        confidence: float = 0.7,
        importance: float = 0.7
    ) -> str:
        """
        Route memory storage to appropriate system

        Routing Logic:
        - episodic → Qdrant
        - preference, goal, fact → PostgreSQL
        - conversation_turn → Redis (handled separately)

        Args:
            user_id: User identifier
            session_id: Session identifier
            memory_type: Type of memory
            content: Memory content
            embedding: Vector embedding (optional, will generate if needed)
            metadata: Additional metadata
            confidence: Confidence score (for semantic)
            importance: Importance score (for episodic)

        Returns:
            str: Memory ID or key
        """

        # Route to episodic (Qdrant)
        if memory_type == "episodic" or memory_type in ["transaction", "inquiry", "complaint"]:
            if not embedding:
                embedding = await self.embedding_service.generate_document_embedding(
                    content,
                    title=memory_type
                )

            # Type guard: embedding is guaranteed to be List[float] here
            if embedding:
                memory_id = await self.qdrant.store_episodic_memory(
                    user_id=user_id,
                    session_id=session_id,
                    content=content,
                    embedding=embedding,
                    event_type=memory_type,
                    metadata=metadata or {},
                    importance=importance
                )
                return memory_id

        # Route to semantic (PostgreSQL)
        elif memory_type in ["preference", "goal", "fact"]:
            if not embedding:
                embedding = await self.embedding_service.generate_document_embedding(
                    content,
                    title=memory_type
                )

            # Type guard: embedding is guaranteed to be List[float] here
            if embedding:
                memory = await self.memory_repo.store_memory_with_embedding(
                    user_id=user_id,
                    memory_type=memory_type,
                    content=content,
                    embedding=embedding,
                    confidence_score=confidence
                )
                return str(memory.memory_id)

        # Default to semantic
        else:
            if not embedding:
                embedding = await self.embedding_service.generate_document_embedding(
                    content,
                    title=memory_type
                )

            # Type guard: embedding is guaranteed to be List[float] here
            if embedding:
                memory = await self.memory_repo.store_memory_with_embedding(
                    user_id=user_id,
                    memory_type=memory_type,
                    content=content,
                    embedding=embedding,
                    confidence_score=confidence
                )
                return str(memory.memory_id)

        # Fallback: should not reach here if embedding generation succeeds
        raise ValueError("Failed to store memory: embedding generation failed")

    async def get_memory_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get statistics from all memory systems

        Args:
            user_id: User identifier

        Returns:
            Dict with stats from each system
        """
        stats = {
            "procedural": {},
            "semantic": {},
            "episodic": {}
        }

        # Procedural (Redis) stats
        try:
            stats["procedural"] = self.redis.get_cache_stats()
        except Exception as e:
            stats["procedural"]["error"] = str(e)

        # Semantic (PostgreSQL) stats
        try:
            count = await self.memory_repo.get_memories_with_embeddings_count(user_id)
            stats["semantic"]["total_memories"] = count
        except Exception as e:
            stats["semantic"]["error"] = str(e)

        # Episodic (Qdrant) stats
        try:
            stats["episodic"] = self.qdrant.get_collection_stats()
        except Exception as e:
            stats["episodic"]["error"] = str(e)

        return stats
