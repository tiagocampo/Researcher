from setuptools import setup, find_packages

setup(
    name="company_pitchbook",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "langchain",
        "langchain-openai",
        "pydantic",
        "beautifulsoup4",
        "duckduckgo-search",
        "python-dotenv",
    ],
    python_requires=">=3.8",
) 