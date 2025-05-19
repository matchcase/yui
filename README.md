# Yui
A feature-rich, LangGraph-based Agentic Discord bot with long-term memory!
## Features
- 🧠 Long-term conversation memory using ChromaDB
- 🤖 Supports Google, OpenAI, Anthropic and even local models
- 🌐 (Tool) Can search the web and crawl webpages
- 🔔 (Tool) Can set reminders for users and ping them
- ☑️ (Tool) Can maintain TODO lists for users
- 🎲 (Tool) Can roll dice (you can specify the number of die and faces!)
## Setup
Before running the bot, make sure that you have a local [SearxNG](https://github.com/searxng/searxng) instance running, as this will be used to do web searches.
First, set up a virtual environment and install the dependencies:
```
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```
Next, run the bot:
```
python3 main.py
```
You will be prompted to provide values for your Discord token, provider API token (or Ollama model name) and user ID.
Your bot should be up and running now!
## Planned features
- [ ] Slash commands for tools
- [ ] Multimodal model integration
- [ ] Tool for creating polls
- [ ] Tool for captioning images
- [ ] Tool for searching GIFs
## License
This repository is licensed under AGPLv3 or later. © Sarthak Shah (matchcase), 2025
