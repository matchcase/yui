from client.handlers import setup_handlers
import discord
from discord.ext import commands
from setup.config import config
import logging

logger = logging.getLogger(__name__)

async def run_discord_bot():
    """Initialize and run the Discord bot."""
    intents = discord.Intents.default()
    intents.message_content = True
    intents.members = True
    bot = commands.Bot(command_prefix=config.COMMAND_PREFIX, intents=intents)
    setup_handlers(bot)
    try:
        logger.info("Connecting to Discord...")
        await bot.start(config.DISCORD_TOKEN)
    except Exception as e:
        logger.error(f"Error connecting to Discord: {e}", exc_info=True)
    finally:
        if bot.is_closed():
            logger.info("Yui has finished running.")
