from typing import Dict, List, Any, Optional
import logging
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.tavily_search import TavilySearchResults
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from src.utils.retry import retry_async_operation

class WebNavigator:
    """Handles web navigation and search operations"""
    
    def __init__(
        self,
        llm: ChatOpenAI,
        max_retries: int = 3,
        max_concurrent_requests: int = 5,
        request_delay: float = 1.0
    ):
        self.llm = llm
        self.max_retries = max_retries
        self.max_concurrent_requests = max_concurrent_requests
        self.request_delay = request_delay
        self.session: Optional[aiohttp.ClientSession] = None
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0'
        }
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        self.search_wrapper = DuckDuckGoSearchAPIWrapper()

    async def __aenter__(self):
        """Initialize aiohttp session"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()

    @tool("navigate_website")
    async def navigate(
        self,
        url_and_context: str,
    ) -> Dict[str, Any]:
        """Navigate to a URL and extract content with improved error handling.
        Args:
            url_and_context: A string containing the URL and research context in JSON format
        """
        try:
            # Parse input
            import json
            input_data = json.loads(url_and_context)
            url = input_data.get('url')
            research_context = input_data.get('research_context', {})
            max_pages = input_data.get('max_pages', 5)
            
            session_created = False
            if not self.session:
                self.session = aiohttp.ClientSession(headers=self.headers)
                session_created = True

            try:
                timeout = aiohttp.ClientTimeout(total=30)
                async with self.session.get(url, allow_redirects=True, timeout=timeout) as response:
                    if response.status == 200:
                        content = await response.text()
                        
                        # Parse HTML and extract text content
                        soup = BeautifulSoup(content, 'html.parser')
                        
                        # Remove script and style elements
                        for script in soup(["script", "style", "meta", "link"]):
                            script.decompose()
                        
                        # Extract main content
                        main_content = soup.get_text(separator='\n', strip=True)
                        
                        # Split content into manageable chunks
                        chunks = self.text_splitter.split_text(main_content)
                        
                        # Extract metadata
                        metadata = {
                            'title': soup.title.string if soup.title else '',
                            'meta_description': soup.find('meta', {'name': 'description'})['content'] if soup.find('meta', {'name': 'description'}) else '',
                            'h1_headers': [h1.get_text(strip=True) for h1 in soup.find_all('h1')],
                            'url': url
                        }
                        
                        return {
                            'url': url,
                            'text': chunks,
                            'html': content,
                            'metadata': metadata,
                            'extracted_data': {
                                'purpose': research_context.get('purpose', ''),
                                'pages_visited': 1,
                                'max_pages': max_pages
                            }
                        }
                    elif response.status == 403:
                        logging.error(f"Access forbidden (403) for {url}. The website may be blocking automated access.")
                        return {
                            'url': url,
                            'error': 'access_forbidden',
                            'error_message': 'Website is blocking automated access. Using alternative sources.',
                            'extracted_data': {
                                'purpose': research_context.get('purpose', ''),
                                'pages_visited': 0,
                                'max_pages': max_pages,
                                'error_type': '403_forbidden',
                                'fallback_required': True
                            }
                        }
                    else:
                        logging.error(f"Failed to navigate to {url}: Status {response.status}")
                        return {
                            'url': url,
                            'error': 'http_error',
                            'error_message': f'HTTP Status {response.status}',
                            'extracted_data': {
                                'purpose': research_context.get('purpose', ''),
                                'pages_visited': 0,
                                'max_pages': max_pages,
                                'error_type': f'http_{response.status}'
                            }
                        }
            except aiohttp.ClientError as e:
                logging.error(f"Network error navigating to {url}: {str(e)}")
                return {
                    'url': url,
                    'error': 'network_error',
                    'error_message': str(e),
                    'extracted_data': {
                        'purpose': research_context.get('purpose', ''),
                        'pages_visited': 0,
                        'max_pages': max_pages,
                        'error_type': 'network_error'
                    }
                }
            finally:
                if session_created and self.session:
                    await self.session.close()
                    self.session = None
        except Exception as e:
            logging.error(f"Error processing navigation request: {str(e)}")
            return {
                'error': 'processing_error',
                'error_message': str(e)
            }

    @tool("web_search")
    async def search(
        self,
        search_input: str,
    ) -> List[Dict[str, Any]]:
        """Search the web using DuckDuckGo with improved result processing.
        Args:
            search_input: A string containing the search query and options in JSON format
        """
        try:
            # Parse input
            import json
            input_data = json.loads(search_input)
            query = input_data.get('query')
            max_results = input_data.get('max_results', 5)
            
            # Use DuckDuckGo API wrapper for more reliable results
            raw_results = self.search_wrapper.results(query, max_results * 2)
            
            processed_results = []
            for result in raw_results:
                # Enhanced relevance scoring using LLM
                relevance = await self._evaluate_source_enhanced(
                    result.get('link', ''),
                    result.get('title', ''),
                    result.get('snippet', ''),
                    query
                )
                
                if relevance > 0.5:  # Only include highly relevant results
                    processed_results.append({
                        'title': result.get('title', ''),
                        'link': result.get('link', ''),
                        'snippet': result.get('snippet', ''),
                        'relevance_score': relevance,
                        'source_type': self._determine_source_type(result.get('link', ''))
                    })
            
            # Sort by relevance and limit to requested number
            processed_results.sort(key=lambda x: x['relevance_score'], reverse=True)
            return processed_results[:max_results]
            
        except Exception as e:
            logging.error(f"Search error: {str(e)}")
            return []

    async def _evaluate_source_enhanced(
        self,
        url: str,
        title: str,
        snippet: str,
        query: str
    ) -> float:
        """Enhanced source evaluation with structured output parsing"""
        try:
            evaluation_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are evaluating a source for company research.
                Consider the following criteria:
                1. Relevance: How well does the content match the query?
                2. Credibility: Is this a reliable source for company information?
                3. Freshness: How recent and up-to-date is the information likely to be?
                4. Content Quality: How comprehensive and accurate is the content?
                
                Respond with a JSON object containing these exact fields:
                {
                    "relevance_score": <float 0-1>,
                    "credibility_score": <float 0-1>,
                    "freshness_score": <float 0-1>,
                    "reasoning": "<brief explanation>"
                }"""),
                ("user", f"URL: {url}\nTitle: {title}\nSnippet: {snippet}\nQuery: {query}")
            ])
            
            response = await self.llm.ainvoke(evaluation_prompt.format_messages())
            content = response.content.strip()
            
            try:
                # Parse JSON response
                import json
                result = json.loads(content)
                
                # Validate scores
                scores = {
                    key: float(value) 
                    for key, value in result.items() 
                    if key.endswith('_score')
                }
                
                # Ensure all scores are between 0 and 1
                scores = {
                    key: max(0.0, min(1.0, value))
                    for key, value in scores.items()
                }
                
                # Calculate weighted average
                weights = {
                    'relevance_score': 0.5,
                    'credibility_score': 0.3,
                    'freshness_score': 0.2
                }
                
                final_score = sum(
                    scores.get(key, 0.0) * weight 
                    for key, weight in weights.items()
                )
                
                return final_score
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logging.warning(f"Error parsing evaluation result: {str(e)}")
                return 0.0
                
        except Exception as e:
            logging.warning(f"Error evaluating source {url}: {str(e)}")
            return 0.0

    def _determine_source_type(self, url: str) -> str:
        """Determine the type of source based on URL patterns"""
        url_lower = url.lower()
        
        # Professional networks and job sites
        if any(domain in url_lower for domain in [
            'linkedin.com', 'glassdoor.com', 'indeed.com'
        ]):
            return 'professional_network'
            
        # Business and financial data sources
        elif any(domain in url_lower for domain in [
            'crunchbase.com', 'bloomberg.com', 'reuters.com',
            'forbes.com', 'ft.com', 'wsj.com'
        ]):
            return 'business_data'
            
        # Official/institutional sources
        elif any(domain in url_lower for domain in ['.gov', '.edu', '.org']):
            return 'institutional'
            
        # News sources
        elif any(domain in url_lower for domain in [
            '.news', 'press', 'techcrunch.com', 'businesswire.com',
            'prnewswire.com'
        ]):
            return 'news'
            
        # Company websites
        elif any(indicator in url_lower for indicator in [
            'about', 'company', 'corporate', 'investor'
        ]):
            return 'company_website'
            
        return 'general'