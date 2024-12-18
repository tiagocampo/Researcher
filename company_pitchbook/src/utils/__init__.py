"""Utility modules for the company research system."""

import os
from pathlib import Path
from dotenv import load_dotenv

from .html_parser import HTMLParser

def load_env_variables():
    """Load environment variables from .env file."""
    env_path = Path(__file__).parent.parent.parent / '.env'
    load_dotenv(env_path)

def get_api_key() -> str:
    """Get OpenAI API key from environment variables."""
    return os.getenv('OPENAI_API_KEY', '')

__all__ = ['HTMLParser', 'load_env_variables', 'get_api_key'] 