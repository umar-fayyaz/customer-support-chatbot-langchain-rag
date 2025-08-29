import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Airtable
AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LOW_LLM = os.getenv("LOW_LLM", "gpt-4.1-nano")
HIGH_LLM = os.getenv("HIGH_LLM", "gpt-5-nano")

# LangSmith
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
