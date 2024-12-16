from typing import Dict, List, Optional, Set, Tuple, Any, cast
import requests
from bs4 import BeautifulSoup
import logging
from urllib.parse import urljoin, urlparse
import asyncio
import aiohttp
from pydantic import BaseModel, Field
from datetime import datetime
import json
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.outputs import LLMResult
from functools import partial

class WebPage(BaseModel):
    """Represents a web page with its content and metadata"""
    url: str
    title: Optional[str] = None
    content: str
    html: str
    links: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    relevance_score: Optional[float] = None

class NavigationAction(BaseModel):
    """Represents a navigation action to be taken"""
    action_type: str  # "follow_link", "search", "extract", "analyze"
    target_url: Optional[str] = None
    search_query: Optional[str] = None
    extraction_pattern: Optional[Dict[str, Any]] = None
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    context: Dict[str, Any] = Field(default_factory=dict)

class NavigationState(BaseModel):
    """Represents the current state of web navigation"""
    current_url: str
    visited_urls: Set[str] = Field(default_factory=set)
    queued_actions: List[NavigationAction] = Field(default_factory=list)
    extracted_data: Dict[str, Any] = Field(default_factory=dict)
    page_cache: Dict[str, WebPage] = Field(default_factory=dict)

class WebNavigatorTool(BaseTool):
    """Base class for web navigation tools"""
    navigator: 'WebNavigator'
    
    def __init__(self, navigator: 'WebNavigator', **kwargs):
        super().__init__(**kwargs)
        self.navigator = navigator

class AnalyzePageTool(WebNavigatorTool):
    name = "analyze_page"
    description = "Analyze the content of a web page to determine its relevance and extract key information"
    
    async def _arun(
        self,
        page: WebPage,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> Dict[str, Any]:
        return await self.navigator._analyze_page_content(page)

class ExtractLinksTool(WebNavigatorTool):
    name = "extract_links"
    description = "Extract and analyze links from a web page to determine which ones to follow"
    
    def _run(
        self,
        page: WebPage,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> List[str]:
        return self.navigator._extract_links(BeautifulSoup(page.html, 'html5lib'), page.url)

class EvaluateRelevanceTool(WebNavigatorTool):
    name = "evaluate_relevance"
    description = "Evaluate the relevance of a page or link to the current research goal"
    
    async def _arun(
        self,
        page: WebPage,
        research_context: Dict[str, Any],
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> float:
        return await self.navigator._evaluate_page_relevance(page, research_context)

class WebNavigator:
    """Handles intelligent web navigation and content extraction"""
    
    def __init__(
        self,
        llm: ChatOpenAI,
        max_retries: int = 3,
        max_concurrent_requests: int = 5,
        request_delay: float = 0.5
    ):
        self.llm = llm
        self.max_retries = max_retries
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.request_delay = request_delay
        self.session: Optional[aiohttp.ClientSession] = None
        self.state = NavigationState(current_url="")
        
        # Headers to mimic a real browser
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "DNT": "1"
        }
        
        # Initialize tools
        self.tools = [
            AnalyzePageTool(navigator=self),
            ExtractLinksTool(navigator=self),
            EvaluateRelevanceTool(navigator=self)
        ]
        
        # Initialize agent
        self.agent_executor = self._create_agent()
    
    def _create_agent(self) -> AgentExecutor:
        """Create an agent for making navigation decisions"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert web navigation agent.
Your task is to analyze web pages and decide on the next best actions to take.
Consider the following when making decisions:
1. Relevance to the research goal
2. Potential value of information
3. Depth vs breadth tradeoff
4. Avoiding redundant or low-value pages
Always respect robots.txt and implement polite scraping practices."""),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools)
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(headers=self.headers)
        return self.session
    
    async def _fetch_page(self, url: str) -> Optional[WebPage]:
        """Fetch a web page with retry logic"""
        async with self.semaphore:
            for attempt in range(self.max_retries):
                try:
                    session = await self._get_session()
                    async with session.get(url) as response:
                        if response.status == 200:
                            html = await response.text()
                            soup = BeautifulSoup(html, 'html5lib')
                            
                            # Extract title and content
                            title = soup.title.string if soup.title else None
                            content = self._extract_main_content(soup)
                            
                            # Extract links
                            links = self._extract_links(soup, url)
                            
                            return WebPage(
                                url=url,
                                title=title,
                                content=content,
                                html=html,
                                links=links
                            )
                        
                        await asyncio.sleep(self.request_delay)
                        
                except Exception as e:
                    logging.error(f"Error fetching {url}: {str(e)}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.request_delay * (attempt + 1))
                    continue
            
            return None
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from a web page"""
        # Remove unwanted elements
        for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        # Try to find main content area
        main_content = None
        for selector in ['main', 'article', '[role="main"]', '#content', '.content']:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        if not main_content:
            main_content = soup.find('body')
        
        return main_content.get_text(separator=' ', strip=True) if main_content else ''
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract and normalize links from a web page"""
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            # Normalize URL
            full_url = urljoin(base_url, href)
            # Filter valid URLs
            if self._is_valid_url(full_url):
                links.append(full_url)
        return list(set(links))  # Remove duplicates
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if a URL is valid and should be followed"""
        try:
            parsed = urlparse(url)
            return all([
                parsed.scheme in ('http', 'https'),
                parsed.netloc,  # Has domain
                not any(ext in parsed.path.lower() for ext in ['.pdf', '.jpg', '.png', '.gif'])
            ])
        except Exception:
            return False
    
    async def _analyze_page_content(self, page: WebPage) -> Dict[str, Any]:
        """Analyze page content using LLM"""
        try:
            messages = [
                HumanMessage(content="""Analyze the following web page content and extract key information.
Format the output as a JSON object with appropriate keys."""),
                HumanMessage(content=page.content[:4000])  # Limit content length
            ]
            
            response = await self.llm.ainvoke(messages)
            return json.loads(response.content)
            
        except Exception as e:
            logging.error(f"Error analyzing page content: {str(e)}")
            return {}
    
    async def _evaluate_page_relevance(
        self,
        page: WebPage,
        research_context: Dict[str, Any]
    ) -> float:
        """Evaluate page relevance to research goals"""
        try:
            messages = [
                HumanMessage(content=f"""
                Research Context: {json.dumps(research_context)}
                
                Page Title: {page.title}
                Page URL: {page.url}
                
                Rate the relevance of this page to our research goals on a scale of 0.0 to 1.0,
                where 1.0 is highly relevant and 0.0 is not relevant at all.
                Consider factors like:
                - Content relevance to company research
                - Information quality and reliability
                - Potential for unique insights
                
                Return only the numeric score.
                """)
            ]
            
            response = await self.llm.ainvoke(messages)
            score = float(response.content.strip())
            return max(0.0, min(1.0, score))  # Ensure score is between 0 and 1
            
        except Exception as e:
            logging.error(f"Error evaluating page relevance: {str(e)}")
            return 0.0
    
    async def navigate(
        self,
        start_url: str,
        research_context: Dict[str, Any],
        max_pages: int = 10
    ) -> Dict[str, Any]:
        """Navigate through web pages to gather research information"""
        try:
            self.state = NavigationState(current_url=start_url)
            results = {
                'pages_visited': [],
                'extracted_data': {},
                'navigation_path': []
            }
            
            # Initial action
            self.state.queued_actions.append(
                NavigationAction(
                    action_type="follow_link",
                    target_url=start_url,
                    priority=1.0
                )
            )
            
            pages_visited = 0
            while self.state.queued_actions and pages_visited < max_pages:
                # Get next action with highest priority
                action = max(
                    self.state.queued_actions,
                    key=lambda x: x.priority
                )
                self.state.queued_actions.remove(action)
                
                if action.action_type == "follow_link" and action.target_url:
                    if action.target_url in self.state.visited_urls:
                        continue
                    
                    # Fetch and process page
                    page = await self._fetch_page(action.target_url)
                    if not page:
                        continue
                    
                    # Evaluate relevance
                    page.relevance_score = await self._evaluate_page_relevance(
                        page, research_context
                    )
                    
                    if page.relevance_score > 0.5:  # Threshold for relevant pages
                        # Analyze content
                        analysis = await self._analyze_page_content(page)
                        self.state.extracted_data[page.url] = analysis
                        
                        # Add to results
                        results['pages_visited'].append({
                            'url': page.url,
                            'title': page.title,
                            'relevance_score': page.relevance_score
                        })
                        results['extracted_data'][page.url] = analysis
                        results['navigation_path'].append({
                            'from': self.state.current_url,
                            'to': page.url,
                            'action': action.action_type
                        })
                        
                        # Queue new actions for discovered links
                        for link in page.links:
                            if link not in self.state.visited_urls:
                                self.state.queued_actions.append(
                                    NavigationAction(
                                        action_type="follow_link",
                                        target_url=link,
                                        priority=0.5  # Lower priority for new links
                                    )
                                )
                    
                    self.state.visited_urls.add(action.target_url)
                    self.state.current_url = action.target_url
                    pages_visited += 1
                    
                    # Respect rate limiting
                    await asyncio.sleep(self.request_delay)
            
            return results
            
        except Exception as e:
            logging.error(f"Error during navigation: {str(e)}")
            raise
        finally:
            if self.session and not self.session.closed:
                await self.session.close() 