import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().with_name(".env"))

LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

SQLITE_DB_PATH = "chatbot.db"

PDF_CHUNK_SIZE = 1000
PDF_CHUNK_OVERLAP = 200
PDF_RETRIEVER_K = 4


def get_openai_api_key():
    return os.getenv("OPENAI_API_KEY")


def has_openai_credentials():
    return bool(get_openai_api_key())
