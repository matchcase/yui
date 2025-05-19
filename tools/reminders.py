import json
import os
import asyncio
from datetime import datetime, timezone
import discord
from typing import List, Dict, Any, Optional
import logging
from tools import register_tool
import dateparser
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
REMINDERS_FILE = "data/reminders.json"
os.makedirs("data", exist_ok=True)

def parse_remind_time(time_str: str) -> Optional[datetime]:
    """
    Parses a time string into a timezone-aware UTC datetime.
    Supports absolute dates (e.g., '17th May', '2025-06-01 14:00') and relative expressions ('in 2 minutes', '2h 30m').
    Returns a datetime in UTC if parsing is successful and refers to the future, otherwise None.
    """
    if not time_str:
        return None
    # Use dateparser to handle both exact and relative date expressions
    settings = {
        'RETURN_AS_TIMEZONE_AWARE': True,
        'TO_TIMEZONE': 'UTC',
        'TIMEZONE': 'IST',
        'PREFER_DATES_FROM': 'future',
    }
    dt = dateparser.parse(time_str, settings=settings)
    if not dt:
        logger.warning(f"Dateparser could not parse '{time_str}'")
        return None
    # Ensure it's in the future
    now = datetime.now(timezone.utc)
    if dt <= now:
        logger.warning(f"Parsed datetime {dt.isoformat()} is not in the future")
        return None
    return dt


def load_reminders() -> List[Dict[str, Any]]:
    if not os.path.exists(REMINDERS_FILE):
        return []
    try:
        with open(REMINDERS_FILE, 'r') as f:
            content = f.read()
            if not content.strip():
                return []
            return json.loads(content)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Error loading reminders: {e}")
        return []

def save_reminders(reminders: List[Dict[str, Any]]):
    try:
        with open(REMINDERS_FILE, 'w') as f:
            json.dump(reminders, f, indent=4)
    except IOError as e:
        logger.error(f"Error saving reminders: {e}")

class SetReminderArgs(BaseModel):
    time_str: str = Field(description="The time until the reminder (e.g., 'in 2 minutes', '17th May').")
    message: str = Field(description="The reminder message content.")

@register_tool(
    name="set_reminder",
    description=("Sets a reminder for the user. Supports absolute dates like '17th May' and "
                 "relative times like 'in 2h 30m'."),
    args_schema=SetReminderArgs
)

def set_reminder(time_str: str, message: str, user_id: str, channel_id: str) -> str:
    logger.info(f"set_reminder called with time_str='{time_str}', message='{message}'")
    processed = time_str.strip()
    if processed.lower().startswith(('in ', 'for ')):
        processed = ' '.join(processed.split()[1:])
    target_dt = parse_remind_time(processed)
    if not target_dt:
        return (f"Sorry, I couldn't understand the time '{time_str}'. "
                f"Please use formats like 'in 1 hour', '2 days', or exact dates like '17th May'.")
    now = datetime.now(timezone.utc)
    delta = target_dt - now
    reminders = load_reminders()
    reminders.append({
        "user_id": str(user_id),
        "channel_id": str(channel_id),
        "remind_time_iso": target_dt.isoformat(),
        "message": message,
        "set_time_iso": now.isoformat(),
    })
    save_reminders(reminders)
    days = delta.days
    seconds = delta.seconds
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    parts = []
    if days: parts.append(f"{days} day{'s' if days!=1 else ''}")
    if hours: parts.append(f"{hours} hour{'s' if hours!=1 else ''}")
    if minutes: parts.append(f"{minutes} minute{'s' if minutes!=1 else ''}")
    if not parts and secs:
        parts.append(f"{secs} second{'s' if secs!=1 else ''}")
    duration_str = ', '.join(parts) if parts else 'just a moment'
    ts = int(target_dt.timestamp())
    full = f"<t:{ts}:F>"
    rel = f"<t:{ts}:R>"
    return f"Okay, I'll remind you about '{message}' in {duration_str} (that is {full}, {rel})."

async def check_reminders(bot: discord.Client):
    """Periodically checks and sends due reminders."""
    now_utc = datetime.now(timezone.utc)
    reminders_due = []
    active_reminders = load_reminders()
    remaining_reminders = []
    for r in active_reminders:
        try:
            remind_time = datetime.fromisoformat(r["remind_time_iso"])
            if remind_time <= now_utc:
                reminders_due.append(r)
            else:
                remaining_reminders.append(r)
        except Exception as e:
            logger.error(f"Error processing reminder data {r}: {e}", exc_info=True)
            remaining_reminders.append(r) # Keep malformed/problematic reminders for now to avoid data loss
    if reminders_due:
        save_reminders(remaining_reminders) # Update stored reminders
    for r_due in reminders_due:
        try:
            user_id_int = int(r_due["user_id"])
            channel_id_int = int(r_due["channel_id"])
            user = await bot.fetch_user(user_id_int)
            target_channel = bot.get_channel(channel_id_int)
            reminder_message_text = f"Hey {user.mention}, here's your reminder: **{r_due['message']}**"
            destination_to_send = None
            if target_channel:
                try:
                    if isinstance(target_channel, (discord.TextChannel, discord.Thread)):
                         pass
                    destination_to_send = target_channel
                except discord.Forbidden:
                    logger.warning(f"Bot lacks permission to send to channel {channel_id_int} for reminder. Trying DM.")
                    destination_to_send = None # As a fallback, DM the user
            
            if not destination_to_send:
                destination_to_send = user.dm_channel
                if not destination_to_send:
                    destination_to_send = await user.create_dm()
            
            if destination_to_send:
                await destination_to_send.send(reminder_message_text)
                logger.info(f"Sent reminder to user {user_id_int} (channel/DM: {destination_to_send.id}): {r_due['message']}")
            else:
                 logger.warning(f"Could not find a suitable channel or DM to send reminder for user {user_id_int}.")

        except discord.NotFound:
            logger.warning(f"User {r_due.get('user_id')} or Channel {r_due.get('channel_id')} not found for reminder.")
        except discord.Forbidden:
            logger.warning(f"Could not send reminder to user {r_due.get('user_id')} (channel/DM {r_due.get('channel_id')}). Permissions issue.")
        except ValueError:
            logger.error(f"Invalid user_id or channel_id in reminder: {r_due}")
        except Exception as e:
            logger.error(f"Error sending reminder for user {r_due.get('user_id')}: {e}", exc_info=True)
