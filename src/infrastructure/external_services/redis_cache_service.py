"""
Redis Cache Service

Manages procedural memory (short-term context) using Redis.
Stores recent conversation turns, active slots, and workflow state.
"""

import json
import redis
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class ConversationMessage:
    """Represents a single conversation message"""
    role: str  # user|assistant
    content: str
    timestamp: str


@dataclass
class ConversationContext:
    """Represents full conversation context"""
    user_id: str
    session_id: str
    recent_messages: List[ConversationMessage]
    active_slots: Dict[str, Any]
    detected_intent: Optional[str]
    conversation_stage: Optional[str]
    last_updated: str


class RedisCacheService:
    """
    Redis cache service for procedural memory

    Handles:
    - Recent conversation context (last 5-10 turns)
    - Active slot-filling state
    - Workflow state tracking
    - TTL-based expiration
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        default_ttl: int = 3600  # 1 hour
    ):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.default_ttl = default_ttl
        self.client: Optional[redis.Redis] = None

    def connect(self) -> bool:
        """
        Initialize connection to Redis

        Returns:
            bool: True if connection successful
        """
        try:
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )

            # Test connection
            self.client.ping()
            return True

        except Exception as e:
            raise ConnectionError(f"Failed to connect to Redis: {str(e)}")

    def _get_key(self, user_id: str, session_id: str) -> str:
        """Generate Redis key for context"""
        return f"user:{user_id}:session:{session_id}:context"

    async def store_conversation_context(
        self,
        user_id: str,
        session_id: str,
        context: ConversationContext,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Store full conversation context

        Args:
            user_id: User identifier
            session_id: Session identifier
            context: ConversationContext object
            ttl: Time to live in seconds (default: self.default_ttl)

        Returns:
            bool: True if stored successfully
        """
        if not self.client:
            raise RuntimeError("Not connected to Redis. Call connect() first.")

        key = self._get_key(user_id, session_id)
        ttl = ttl or self.default_ttl

        # Convert context to dict
        context_dict = {
            "user_id": context.user_id,
            "session_id": context.session_id,
            "recent_messages": [asdict(msg) for msg in context.recent_messages],
            "active_slots": context.active_slots,
            "detected_intent": context.detected_intent,
            "conversation_stage": context.conversation_stage,
            "last_updated": context.last_updated
        }

        # Store as JSON
        self.client.setex(
            key,
            ttl,
            json.dumps(context_dict)
        )

        return True

    async def get_conversation_context(
        self,
        user_id: str,
        session_id: str
    ) -> Optional[ConversationContext]:
        """
        Retrieve conversation context from cache

        Args:
            user_id: User identifier
            session_id: Session identifier

        Returns:
            ConversationContext if exists, None otherwise
        """
        if not self.client:
            raise RuntimeError("Not connected to Redis. Call connect() first.")

        key = self._get_key(user_id, session_id)
        data = self.client.get(key)

        if not data:
            return None

        # Parse JSON
        context_dict = json.loads(data)

        # Reconstruct ConversationContext
        messages = [
            ConversationMessage(**msg)
            for msg in context_dict["recent_messages"]
        ]

        return ConversationContext(
            user_id=context_dict["user_id"],
            session_id=context_dict["session_id"],
            recent_messages=messages,
            active_slots=context_dict["active_slots"],
            detected_intent=context_dict.get("detected_intent"),
            conversation_stage=context_dict.get("conversation_stage"),
            last_updated=context_dict["last_updated"]
        )

    async def append_message(
        self,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
        max_messages: int = 10
    ) -> bool:
        """
        Append message to recent messages (maintaining limit)

        Args:
            user_id: User identifier
            session_id: Session identifier
            role: Message role (user|assistant)
            content: Message content
            max_messages: Maximum messages to keep

        Returns:
            bool: True if appended successfully
        """
        if not self.client:
            raise RuntimeError("Not connected to Redis. Call connect() first.")

        # Get existing context
        context = await self.get_conversation_context(user_id, session_id)

        # Create new message
        new_message = ConversationMessage(
            role=role,
            content=content,
            timestamp=datetime.utcnow().isoformat()
        )

        if context:
            # Append to existing
            context.recent_messages.append(new_message)
            # Keep only last N messages
            context.recent_messages = context.recent_messages[-max_messages:]
            context.last_updated = datetime.utcnow().isoformat()
        else:
            # Create new context
            context = ConversationContext(
                user_id=user_id,
                session_id=session_id,
                recent_messages=[new_message],
                active_slots={},
                detected_intent=None,
                conversation_stage=None,
                last_updated=datetime.utcnow().isoformat()
            )

        # Store updated context
        return await self.store_conversation_context(user_id, session_id, context)

    async def update_slot_filling(
        self,
        user_id: str,
        session_id: str,
        slots: Dict[str, Any],
        detected_intent: Optional[str] = None,
        conversation_stage: Optional[str] = None
    ) -> bool:
        """
        Update slot-filling state

        Args:
            user_id: User identifier
            session_id: Session identifier
            slots: Slot values to update
            detected_intent: Current intent
            conversation_stage: Current stage

        Returns:
            bool: True if updated successfully
        """
        if not self.client:
            raise RuntimeError("Not connected to Redis. Call connect() first.")

        # Get existing context
        context = await self.get_conversation_context(user_id, session_id)

        if not context:
            # Create new context
            context = ConversationContext(
                user_id=user_id,
                session_id=session_id,
                recent_messages=[],
                active_slots=slots,
                detected_intent=detected_intent,
                conversation_stage=conversation_stage,
                last_updated=datetime.utcnow().isoformat()
            )
        else:
            # Update existing
            context.active_slots.update(slots)
            if detected_intent:
                context.detected_intent = detected_intent
            if conversation_stage:
                context.conversation_stage = conversation_stage
            context.last_updated = datetime.utcnow().isoformat()

        # Store updated context
        return await self.store_conversation_context(user_id, session_id, context)

    async def get_recent_messages(
        self,
        user_id: str,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[ConversationMessage]:
        """
        Get recent messages from context

        Args:
            user_id: User identifier
            session_id: Session identifier
            limit: Maximum messages to return

        Returns:
            List of ConversationMessage objects
        """
        if not self.client:
            raise RuntimeError("Not connected to Redis. Call connect() first.")

        context = await self.get_conversation_context(user_id, session_id)

        if not context:
            return []

        messages = context.recent_messages

        if limit:
            messages = messages[-limit:]

        return messages

    async def get_active_slots(
        self,
        user_id: str,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Get active slot-filling state

        Args:
            user_id: User identifier
            session_id: Session identifier

        Returns:
            Dict of active slots
        """
        if not self.client:
            raise RuntimeError("Not connected to Redis. Call connect() first.")

        context = await self.get_conversation_context(user_id, session_id)

        if not context:
            return {}

        return context.active_slots

    async def clear_session(
        self,
        user_id: str,
        session_id: str
    ) -> bool:
        """
        Clear session context from cache

        Args:
            user_id: User identifier
            session_id: Session identifier

        Returns:
            bool: True if cleared
        """
        if not self.client:
            raise RuntimeError("Not connected to Redis. Call connect() first.")

        key = self._get_key(user_id, session_id)
        self.client.delete(key)
        return True

    async def extend_ttl(
        self,
        user_id: str,
        session_id: str,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Extend TTL for active session

        Args:
            user_id: User identifier
            session_id: Session identifier
            ttl: New TTL in seconds (default: self.default_ttl)

        Returns:
            bool: True if extended
        """
        if not self.client:
            raise RuntimeError("Not connected to Redis. Call connect() first.")

        key = self._get_key(user_id, session_id)
        ttl = ttl or self.default_ttl

        return self.client.expire(key, ttl)

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get Redis cache statistics

        Returns:
            Dict with cache stats
        """
        if not self.client:
            raise RuntimeError("Not connected to Redis. Call connect() first.")

        try:
            info = self.client.info()
            return {
                "used_memory": info.get("used_memory_human", "N/A"),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(info)
            }
        except Exception as e:
            return {"error": str(e)}

    def _calculate_hit_rate(self, info: Dict) -> float:
        """Calculate cache hit rate percentage"""
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total = hits + misses

        if total == 0:
            return 0.0

        return round((hits / total) * 100, 2)

    def close(self):
        """Close connection to Redis"""
        if self.client:
            self.client.close()
            self.client = None
