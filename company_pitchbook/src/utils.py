import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any
import json

def load_env_variables() -> None:
    """Load environment variables from .env file"""
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(env_path)

def get_api_key() -> str:
    """Get OpenAI API key from environment variables"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    return api_key

def save_research_results(results: Dict[str, Any], company_name: str) -> str:
    """Save research results to a JSON file"""
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)
    
    filename = f"{company_name.lower().replace(' ', '_')}_research.json"
    filepath = data_dir / filename
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    return str(filepath)

def load_research_results(filepath: str) -> Dict[str, Any]:
    """Load research results from a JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def clean_text(text: str) -> str:
    """Clean and format text for display"""
    return text.strip().replace('\n\n\n', '\n\n')

def format_currency(amount: float) -> str:
    """Format currency values"""
    if amount >= 1_000_000_000:
        return f"${amount/1_000_000_000:.1f}B"
    elif amount >= 1_000_000:
        return f"${amount/1_000_000:.1f}M"
    elif amount >= 1_000:
        return f"${amount/1_000:.1f}K"
    else:
        return f"${amount:.2f}"
