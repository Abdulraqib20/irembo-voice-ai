"""
Memory Service - Redis and Mem0 integration
"""

from typing import List, Dict, Any, Optional
import logging
import json
from datetime import datetime
import redis.asyncio as redis
from src.config.settings import REDIS_URL

logger = logging.getLogger(__name__)

class MemoryService:
    """Service for conversation memory and user context"""

    def __init__(self):
        self.redisClient: Optional[redis.Redis] = None
        self.redisAvailable = True
        self.inMemoryStore: Dict[str, List[Dict[str, Any]]] = {}

    async def _getRedisClient(self) -> Optional[redis.Redis]:
        """Get or create Redis client"""
        if not self.redisAvailable:
            return None

        if self.redisClient is None:
            try:
                self.redisClient = await redis.from_url(REDIS_URL, decode_responses=True)
                await self.redisClient.ping()
            except Exception as e:
                logger.warning(f"Redis not available, using in-memory fallback: {str(e)}")
                self.redisAvailable = False
                return None
        return self.redisClient

    async def saveMessage(self, userId: str, message: str, role: str) -> None:
        """Save conversation message to memory"""
        try:
            client = await self._getRedisClient()

            messageData = {
                "role": role,
                "content": message,
                "timestamp": datetime.utcnow().isoformat()
            }

            if client:
                key = f"conversation:{userId}"
                await client.rpush(key, json.dumps(messageData))
                await client.expire(key, 86400)
                logger.info(f"Saved message for user {userId} to Redis")
            else:
                if userId not in self.inMemoryStore:
                    self.inMemoryStore[userId] = []
                self.inMemoryStore[userId].append(messageData)
                if len(self.inMemoryStore[userId]) > 20:
                    self.inMemoryStore[userId] = self.inMemoryStore[userId][-20:]
                logger.info(f"Saved message for user {userId} to in-memory store")

        except Exception as e:
            logger.error(f"Error saving message: {str(e)}")

    async def getConversationHistory(self, userId: str, limit: int = 10) -> List[Dict[str, str]]:
        """Retrieve conversation history"""
        try:
            client = await self._getRedisClient()

            if client:
                key = f"conversation:{userId}"
                messages = await client.lrange(key, -limit, -1)
                history = []
                for msgStr in messages:
                    msgData = json.loads(msgStr)
                    history.append({
                        "role": msgData["role"],
                        "content": msgData["content"]
                    })
                return history
            else:
                userMessages = self.inMemoryStore.get(userId, [])
                history = []
                for msgData in userMessages[-limit:]:
                    history.append({
                        "role": msgData["role"],
                        "content": msgData["content"]
                    })
                return history

        except Exception as e:
            logger.error(f"Error retrieving conversation history: {str(e)}")
            return []

    async def saveUserContext(self, userId: str, context: Dict[str, Any]) -> None:
        """Save user context information"""
        try:
            client = await self._getRedisClient()
            key = f"user_context:{userId}"

            await client.set(key, json.dumps(context), ex=86400)
            logger.info(f"Saved context for user {userId}")

        except Exception as e:
            logger.error(f"Error saving user context: {str(e)}")
            raise

    async def getUserContext(self, userId: str) -> Optional[Dict[str, Any]]:
        """Retrieve user context"""
        try:
            client = await self._getRedisClient()
            key = f"user_context:{userId}"

            contextStr = await client.get(key)
            if contextStr:
                return json.loads(contextStr)
            return None

        except Exception as e:
            logger.error(f"Error retrieving user context: {str(e)}")
            return None

    async def clearConversation(self, userId: str) -> None:
        """Clear conversation history"""
        try:
            client = await self._getRedisClient()
            await client.delete(f"conversation:{userId}")
            logger.info(f"Cleared conversation for user {userId}")

        except Exception as e:
            logger.error(f"Error clearing conversation: {str(e)}")
            raise

    async def close(self) -> None:
        """Close Redis connection"""
        if self.redisClient:
            await self.redisClient.close()
