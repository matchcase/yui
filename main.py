# Yui - A LangGraph-based Discord Bot!
# AGPLv3 or later
# © Sarthak Shah (matchcase), 2025

import asyncio
import logging
from pathlib import Path
from rich.logging import RichHandler

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)

Path("data/memory_store").mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    logger.info("Starting LangGraph Discord Bot")
    
    try:
        from client.bot import run_discord_bot
        asyncio.run(run_discord_bot())
    except Exception as e:
        logger.error(f"Error starting bot: {e}", exc_info=True)
