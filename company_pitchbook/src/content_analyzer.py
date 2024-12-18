from typing import Dict, Any
import logging
import json
import re
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.utils.retry import retry_async_operation

class ContentAnalyzer:
    """Analyzes webpage content for company information"""
    
    def __init__(self, llm: ChatOpenAI, max_retries: int = 3):
        self.llm = llm
        self.max_retries = max_retries

    async def analyze(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze webpage content using LLM"""
        # First try to extract structured data
        schema_data = self._extract_schema_org(content.get('html', ''))
        if schema_data:
            return schema_data
        
        # Fall back to LLM analysis
        return await retry_async_operation(
            self._analyze_with_llm,
            max_retries=self.max_retries,
            args=(content.get('text', ''),)
        )

    async def _analyze_with_llm(self, text: str) -> Dict[str, Any]:
        """Analyze text content using LLM"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract company information from the content. Return a JSON object with these fields:
            {
                "description": "Brief company description",
                "products_services": ["List of products/services"],
                "market_info": {"key market insights"},
                "competitors": ["List of competitors"]
            }"""),
            ("user", f"Content: {text[:2000]}")
        ])
        
        response = await self.llm.ainvoke(prompt.format_messages())
        content = response.content.strip()
        
        # Extract JSON from response
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {}

    def _extract_schema_org(self, html: str) -> Dict[str, Any]:
        """Extract schema.org data from HTML"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find all script tags with schema.org data
            schema_tags = soup.find_all('script', type='application/ld+json')
            for tag in schema_tags:
                try:
                    data = json.loads(tag.string)
                    if isinstance(data, dict):
                        if data.get('@type') == 'Organization':
                            return {
                                'description': data.get('description', ''),
                                'products_services': data.get('makesOffer', []),
                                'market_info': {
                                    'industry': data.get('industry', ''),
                                    'size': data.get('numberOfEmployees', '')
                                },
                                'competitors': []  # Schema.org doesn't typically include competitors
                            }
                except (json.JSONDecodeError, AttributeError):
                    continue
            
            return {}
        except Exception as e:
            logging.warning(f"Error extracting schema.org data: {str(e)}")
            return {} 