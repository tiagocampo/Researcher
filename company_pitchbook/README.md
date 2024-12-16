# Company Research & Pitchbook Generator

An AI-powered application that conducts comprehensive company research and generates detailed pitchbooks using advanced LLM capabilities and intelligent web navigation.

## Features

### Intelligent Research
- Automated multi-source research using LangChain and LangGraph
- Smart web navigation with relevance scoring
- Parallel research task execution
- Adaptive exploration based on discovered information

### Data Sources
- Company websites and public documents
- News articles and press releases
- Industry reports and market analysis
- Competitor information
- Financial data (when available)

### Analysis Capabilities
- Company overview and history
- Market position analysis
- Competitive landscape
- Financial performance metrics
- SWOT analysis
- Product and service analysis
- Technology stack assessment

### Advanced Technologies
- LangChain for orchestration and LLM integration
- LangGraph for workflow management
- Async processing for improved performance
- Intelligent web navigation with priority-based exploration
- Error handling and retry mechanisms
- Rate limiting and polite scraping

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd company_pitchbook
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## Usage

1. Start the Streamlit application:
   ```bash
   streamlit run src/app.py
   ```

2. Enter company information:
   - Company name
   - Location
   - Website
   - Business model
   - Products/Services

3. Click "Generate Pitchbook" to start the research process

4. View and download results:
   - Interactive research results viewer
   - Downloadable PDF pitchbook
   - Detailed analysis sections
   - Source citations and references

## Project Structure

```
company_pitchbook/
├── src/
│   ├── app.py              # Streamlit interface
│   ├── researcher.py       # Research orchestration
│   ├── web_navigator.py    # Intelligent web navigation
│   ├── agents.py          # Research agents and tools
│   ├── async_research.py  # Async research capabilities
│   ├── generator.py       # Pitchbook generation
│   └── utils.py           # Utility functions
├── tests/                 # Test suite
├── data/                  # Data storage
├── templates/             # Report templates
├── requirements.txt       # Dependencies
└── README.md             # Documentation
```

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key
- `LANGCHAIN_API_KEY`: Your LangChain API key (optional)
- `LANGSMITH_API_KEY`: Your LangSmith API key (optional)

### Customization
- Adjust research parameters in `config.py`
- Modify report templates in `templates/`
- Configure web navigation settings in `web_navigator.py`

## Development

### Running Tests
```bash
pytest tests/
pytest tests/ --cov=src  # With coverage
```

### Type Checking
```bash
mypy src/
```

### Code Style
```bash
black src/
isort src/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and type checking
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built with [LangChain](https://github.com/langchain-ai/langchain)
- Powered by [OpenAI](https://openai.com/)
- UI by [Streamlit](https://streamlit.io/)
