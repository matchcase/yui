import chromadb
from chromadb.config import Settings as ChromadbSettings
import logging
import asyncio
from typing import List, Dict
import uuid
from core.models import get_embedding_model

logger = logging.getLogger(__name__)

MEMORY_VECTOR_DB_PATH = "data/memory_store"
MEMORY_TOP_K = 5

client = chromadb.PersistentClient(path=MEMORY_VECTOR_DB_PATH, settings=ChromadbSettings(anonymized_telemetry=False))

# Setup
try:
    memory_collection = client.get_or_create_collection(
        name="conversation_memory",
        metadata={"hnsw:space": "cosine"}
    )
except Exception as e:
    logger.error(f"Error initializing memory database: {e}", exc_info=True)
    raise

async def add_to_memory(
        user_id: str,
        channel_id: str,
        user_message: str,
        bot_response: str,
        attachments: List[str] = [],
) -> None:
    """Add a conversation exchange to memory."""
    try:
        memory_id = str(uuid.uuid4())
        from core.models import get_embedding_model
        embedding_model = get_embedding_model()
        combined_text = f"User: {user_message}\nAssistant: {bot_response}"
        embedding = await embedding_model.aembed_query(combined_text)
        metadata = [{
                "user_id": user_id,
                "channel_id": channel_id,
                "timestamp": str(int(asyncio.get_event_loop().time()))
            }]
        if attachments: # we cannot add a list to memory
            metadata[0]["attachments"] = ", ".join(attachments)
        memory_collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            metadatas=metadata,
            documents=[combined_text]
        )
        logger.info(f"Added memory {memory_id} for user {user_id}")
    except Exception as e:
        logger.error(f"Error adding to memory: {e}", exc_info=True)

async def get_relevant_history(query: str, user_id: str, channel_id: str) -> List[Dict[str, str]]:
    """Retrieve relevant conversation history based on semantic similarity."""
    try:
        embedding_model = get_embedding_model()
        embedding = await embedding_model.aembed_query(query)
        results = memory_collection.query(
            query_embeddings=[embedding],
            n_results=MEMORY_TOP_K,
            where={"$and": [{"user_id": user_id}, {"channel_id": channel_id}]}
        )
        chat_history = []
        if results and results.get("documents"):
            for document in results.get("documents", [[]])[0]:
                if document:
                    parts = document.split("\n")
                    if len(parts) >= 2:
                        user_msg = parts[0].replace("User: ", "")
                        assistant_msg = parts[1].replace("Assistant: ", "")
                        
                        chat_history.append({
                            "user": user_msg,
                            "assistant": assistant_msg
                        })
        
        return chat_history
    except Exception as e:
        logger.error(f"Error retrieving memory: {e}", exc_info=True)
        return []
