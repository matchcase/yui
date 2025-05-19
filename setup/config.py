import os
import json
import logging
from pathlib import Path
from InquirerPy import inquirer
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

Path("data").mkdir(parents=True, exist_ok=True)

class ProgramConfig():
    def __init__(self):
        load_dotenv()
        self.config_dict = {}
        if os.path.exists("data/config.json"):
            logger.info("Loading configuration...")
            with open("data/config.json", "r", encoding="utf-8") as f:
                self.config_dict = json.load(f)
        def fetch(value: str, fallback: str = "") -> str:
            return self.config_dict.get(value) or os.getenv(value) or fallback
        
        # Discord configuration
        self.DISCORD_TOKEN = fetch("DISCORD_TOKEN") or inquirer.text(message="Enter your Discord API Key").execute()
        self.COMMAND_PREFIX = "!"

        # API Keys
        self.GEMINI_API_KEY = fetch("GOOGLE_API_KEY")
        self.OPENAI_API_KEY = fetch("OPENAI_API_KEY")
        self.ANTHROPIC_API_KEY = fetch("ANTHROPIC_API_KEY")
        self.OLLAMA_HOST = fetch("OLLAMA_HOST", "localhost:11434")
        
        # Model configuration
        self.GEMINI_MODEL = fetch("GEMINI_MODEL", "gemini-2.0-flash")
        self.OPENAI_MODEL = fetch("OPENAI_MODEL", "gpt-4o")
        self.ANTHROPIC_MODEL = fetch("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        self.OLLAMA_MODEL = fetch("OLLAMA_MODEL")
        
        # Miscellaneous configuration
        self.DEVELOPER_USER_ID = fetch("DEVELOPER_USER_ID") or inquirer.text(message="Enter the User ID of the developer (optional)").execute()

        if not (self.GEMINI_API_KEY or self.OPENAI_API_KEY or self.ANTHROPIC_API_KEY or self.OLLAMA_MODEL):
            provider = inquirer.fuzzy(message="Choose your provider", choices=["Google", "OpenAI", "Anthropic", "Ollama"]).execute()
            api_key = inquirer.text(message="Enter the API Key")
            match provider:
                case "Google":
                    self.GEMINI_API_KEY = api_key.execute()
                case "OpenAI":
                    self.OPENAI_API_KEY = api_key.execute()
                case "Anthropic":
                    self.ANTHROPIC_API_KEY = api_key.execute()
                case "Ollama":
                    self.OLLAMA_MODEL = inquirer.text(message="Enter the name of the model").execute()
        
        self.DEFAULT_LLM_PROVIDER = "gemini" if self.GEMINI_API_KEY else "openai" if self.OPENAI_API_KEY else "anthropic" if self.ANTHROPIC_API_KEY else "ollama"
        self.config_dict_new = {
            "DISCORD_TOKEN": self.DISCORD_TOKEN,
            "GEMINI_API_KEY": self.GEMINI_API_KEY,
            "OPENAI_API_KEY": self.OPENAI_API_KEY,
            "ANTHROPIC_API_KEY": self.ANTHROPIC_API_KEY,
            "OLLAMA_HOST": self.OLLAMA_HOST,
            "GEMINI_MODEL": self.GEMINI_MODEL,
            "OPENAI_MODEL": self.OPENAI_MODEL,
            "ANTHROPIC_MODEL": self.ANTHROPIC_MODEL,
            "OLLAMA_MODEL": self.OLLAMA_MODEL,
            "DEVELOPER_USER_ID": self.DEVELOPER_USER_ID,
            "DEFAULT_LLM_PROVIDER": self.DEFAULT_LLM_PROVIDER,
        }
        
        if self.config_dict != self.config_dict_new:
           with open("data/config.json", "w", encoding="utf-8") as f:
               logger.info("Updating configuration...")
               json.dump(self.config_dict_new, f, ensure_ascii=False, indent=2)
        
config = ProgramConfig()
