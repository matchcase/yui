import discord
from discord.ext import commands, tasks
import logging
import re
import json
import os
from typing import Dict, Any

from core.agent import create_agent_graph, process_message_with_graph
from tools.reminders import check_reminders, REMINDERS_FILE

logger = logging.getLogger(__name__)

active_conversations: Dict[str, Any] = {}

def setup_handlers(bot: commands.Bot):
    """Set up event handlers and commands for the Discord bot."""

    @bot.event
    async def on_ready():
        """Called when the bot is ready and connected to Discord."""
        logger.info(f"Bot connected as {bot.user.name} (ID: {bot.user.id})")
        logger.info(f"Connected to {len(bot.guilds)} guilds.")
        activity = discord.Activity(type=discord.ActivityType.listening, name="your messages and 'yui'")
        await bot.change_presence(activity=activity)
        if not check_reminders_task.is_running():
            check_reminders_task.start()
            logger.info("Started checking for reminders...")

    @bot.event
    async def on_message(message: discord.Message):
        """Handles incoming messages to the bot."""
        if message.author == bot.user:  # Ignore bot messages
            return

        await bot.process_commands(message)
        ctx = await bot.get_context(message)
        if ctx.valid:
            return

        is_dm = isinstance(message.channel, discord.DMChannel)
        is_mentioned = bot.user in message.mentions
        # "Yui" is the trigger word, but we also want to respond to "Yui!", "Yui," and lowercase strs
        message_content_word_set = set(re.sub(r"[^a-zA-Z]", "", message.content).lower().split())
        contains_trigger_keyword = "yui" in message_content_word_set

        if is_dm or is_mentioned or contains_trigger_keyword:
            # Replace bot mention with "Yui"
            content_for_agent = re.sub(f"<@!?{bot.user.id}>", "Yui", message.content, count=1).strip()
            async with message.channel.typing(): # Show "Bot is typing..."
                try:
                    user_id = str(message.author.id)
                    channel_id = str(message.channel.id)
                    conv_key = f"{channel_id}:{user_id}"
                    if conv_key not in active_conversations:
                        logger.info(f"Creating new agent graph for conversation: {conv_key}")
                        active_conversations[conv_key] = create_agent_graph()
                    agent_graph_instance = active_conversations[conv_key]
                    response_text = await process_message_with_graph(
                        agent_graph=agent_graph_instance,
                        user_message_content=content_for_agent,
                        user_id=user_id,
                        channel_id=channel_id,
                    )
                    if response_text:
                        await message.reply(response_text)
                    else:
                        logger.warning(f"Agent (conv: {conv_key}) returned no response or empty string for input: '{content_for_agent}'.")
                        await message.reply("I'm not sure how to respond to that right now. Please try rephrasing.")
                except Exception as e:
                    logger.error(f"Error processing message via agent (conv: {active_conversations.get(conv_key, 'N/A')}): {e}", exc_info=True)
                    await message.reply("I'm a bit flustered and encountered an error trying to respond. Please try again later!")

    @bot.command(name="reset")
    @commands.cooldown(1, 30, commands.BucketType.user) # Cooldown to prevent spam
    async def reset_conversation(ctx: commands.Context):
        """Resets your current conversation history with the bot."""
        user_id = str(ctx.author.id)
        channel_id = str(ctx.channel.id)
        conv_key = f"{channel_id}:{user_id}"
        if conv_key in active_conversations:
            del active_conversations[conv_key]
            await ctx.reply("Our conversation has been reset. I'm ready for a fresh start!")
            logger.info(f"Reset conversation for {conv_key}")
        else:
            await ctx.reply("We don't seem to have an active conversation to reset. Feel free to start one!")

    @tasks.loop(seconds=30) # Check for reminders every 30s
    async def check_reminders_task():
        await check_reminders(bot)

    @check_reminders_task.before_loop
    async def before_check_reminders_task():
        await bot.wait_until_ready()
        logger.info("check_reminders_task waiting for the bot to be ready.")

    # Create an empty reminders file if it does not exist
    if not os.path.exists(REMINDERS_FILE):
        try:
            with open(REMINDERS_FILE, 'w') as f:
                json.dump([], f)
            logger.info(f"Created empty reminders file at {REMINDERS_FILE}")
        except IOError as e:
            logger.error(f"Could not create reminders file at {REMINDERS_FILE}: {e}")
