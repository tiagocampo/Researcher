from typing import Dict, List, Any, Optional
import asyncio
import logging
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate
from duckduckgo_search import DDGS

from .web_navigator import WebNavigator
from .async_research import AsyncResearchTask, AsyncResearchOrchestrator

class ResearchResults(BaseModel):
    """Container for research results"""
    company_info: Dict[str, Any] = Field(default_factory=dict)
    market_analysis: Dict[str, Any] = Field(default_factory=dict)
    financial_data: Dict[str, Any] = Field(default_factory=dict)
    competitors: List[Dict[str, Any]] = Field(default_factory=list)
    news_articles: List[Dict[str, Any]] = Field(default_factory=list)
    sources: List[Dict[str, str]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)

class CompanyResearcher:
    """Handles company research using multiple data sources and AI analysis"""
    
    def __init__(
        self,
        company_name: str,
        location: str,
        website: str,
        business_model: str,
        products_services: str,
        llm: Optional[ChatOpenAI] = None
    ):
        self.company_name = company_name
        self.location = location
        self.website = website
        self.business_model = business_model
        self.products_services = products_services
        
        # Initialize LLM if not provided
        self.llm = llm or ChatOpenAI(
            temperature=0,
            model="gpt-4-turbo-preview"
        )
        
        # Initialize components
        self.web_navigator = WebNavigator(
            llm=self.llm,
            max_retries=3,
            max_concurrent_requests=5,
            request_delay=1.0
        )
        
        self.research_orchestrator = AsyncResearchOrchestrator(
            llm=self.llm,
            tools=self._create_research_tools()
        )
    
    def _create_research_tools(self) -> List[Tool]:
        """Create tools for research tasks"""
        return [
            Tool(
                name="search_web",
                func=self._search_web,
                description="Search the web for company information"
            ),
            Tool(
                name="analyze_website",
                func=self._analyze_website,
                description="Analyze company website content"
            ),
            Tool(
                name="find_news",
                func=self._find_news,
                description="Find recent news articles about the company"
            ),
            Tool(
                name="analyze_competitors",
                func=self._analyze_competitors,
                description="Analyze company competitors"
            ),
            Tool(
                name="extract_financials",
                func=self._extract_financials,
                description="Extract financial information if available"
            )
        ]
    
    async def _search_web(self, query: str) -> List[Dict[str, Any]]:
        """Search the web using DuckDuckGo"""
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(
                    query,
                    region='wt-wt',
                    safesearch='moderate',
                    timelimit='y',  # Last year
                    max_results=10
                ))
            return results
        except Exception as e:
            logging.error(f"Error in web search: {str(e)}")
            return []
    
    async def _analyze_website(self) -> Dict[str, Any]:
        """Analyze company website using web navigator"""
        try:
            research_context = {
                'company_name': self.company_name,
                'business_model': self.business_model,
                'products_services': self.products_services,
                'objective': 'company_analysis'
            }
            
            results = await self.web_navigator.navigate(
                start_url=self.website,
                research_context=research_context,
                max_pages=10
            )
            
            # Process and structure the results
            website_analysis = {
                'overview': {},
                'products': [],
                'services': [],
                'team': [],
                'technology': [],
                'customers': [],
                'locations': []
            }
            
            for url, data in results['extracted_data'].items():
                # Merge data into appropriate categories
                for key in website_analysis.keys():
                    if key in data:
                        if isinstance(website_analysis[key], list):
                            website_analysis[key].extend(data[key])
                        elif isinstance(website_analysis[key], dict):
                            website_analysis[key].update(data[key])
            
            return website_analysis
            
        except Exception as e:
            logging.error(f"Error analyzing website: {str(e)}")
            return {}
    
    async def _find_news(self) -> List[Dict[str, Any]]:
        """Find and analyze recent news articles"""
        try:
            # Search for news
            query = f"{self.company_name} {self.location} news"
            news_results = await self._search_web(query)
            
            # Analyze each news article
            analyzed_news = []
            for article in news_results:
                if article['link'] not in self.web_navigator.state.visited_urls:
                    research_context = {
                        'company_name': self.company_name,
                        'objective': 'news_analysis'
                    }
                    
                    article_analysis = await self.web_navigator.navigate(
                        start_url=article['link'],
                        research_context=research_context,
                        max_pages=1
                    )
                    
                    if article_analysis['extracted_data']:
                        analyzed_news.append({
                            'title': article['title'],
                            'url': article['link'],
                            'date': article.get('date'),
                            'analysis': article_analysis['extracted_data']
                        })
            
            return analyzed_news
            
        except Exception as e:
            logging.error(f"Error finding news: {str(e)}")
            return []
    
    async def _analyze_competitors(self) -> List[Dict[str, Any]]:
        """Analyze company competitors"""
        try:
            # Create competitor analysis tasks
            tasks = [
                AsyncResearchTask(
                    id="find_competitors",
                    name="Find Competitors",
                    description="Find main competitors",
                    prompt=f"Find main competitors for {self.company_name} in {self.location}",
                    required_tools=["search_web"]
                ),
                AsyncResearchTask(
                    id="competitor_analysis",
                    name="Analyze Competitors",
                    description="Analyze competitor information",
                    prompt="Analyze each competitor's strengths and weaknesses",
                    dependencies=["find_competitors"]
                )
            ]
            
            # Add tasks to orchestrator
            for task in tasks:
                self.research_orchestrator.add_task(task)
            
            # Execute tasks
            results = await self.research_orchestrator.execute_all()
            
            # Process competitor analysis
            competitors = []
            if "find_competitors" in results and "competitor_analysis" in results:
                competitor_list = results["find_competitors"].content
                analysis = results["competitor_analysis"].content
                
                for competitor in competitor_list:
                    if isinstance(competitor, dict) and 'name' in competitor:
                        competitor_data = {
                            'name': competitor['name'],
                            'website': competitor.get('website', ''),
                            'analysis': next(
                                (a for a in analysis if a['name'] == competitor['name']),
                                {}
                            )
                        }
                        competitors.append(competitor_data)
            
            return competitors
            
        except Exception as e:
            logging.error(f"Error analyzing competitors: {str(e)}")
            return []
    
    async def _extract_financials(self) -> Dict[str, Any]:
        """Extract financial information if available"""
        try:
            # Create financial analysis tasks
            tasks = [
                AsyncResearchTask(
                    id="find_financial_sources",
                    name="Find Financial Sources",
                    description="Find sources of financial information",
                    prompt=f"Find financial information sources for {self.company_name}",
                    required_tools=["search_web"]
                ),
                AsyncResearchTask(
                    id="extract_financial_data",
                    name="Extract Financial Data",
                    description="Extract and analyze financial data",
                    prompt="Extract key financial metrics and indicators",
                    dependencies=["find_financial_sources"]
                )
            ]
            
            # Add tasks to orchestrator
            for task in tasks:
                self.research_orchestrator.add_task(task)
            
            # Execute tasks
            results = await self.research_orchestrator.execute_all()
            
            # Process financial data
            financial_data = {
                'metrics': {},
                'analysis': {},
                'sources': []
            }
            
            if "extract_financial_data" in results:
                financial_data.update(results["extract_financial_data"].content)
            
            return financial_data
            
        except Exception as e:
            logging.error(f"Error extracting financials: {str(e)}")
            return {}
    
    async def research(self) -> ResearchResults:
        """Conduct comprehensive company research"""
        try:
            # Create research tasks
            tasks = [
                AsyncResearchTask(
                    id="website_analysis",
                    name="Website Analysis",
                    description="Analyze company website",
                    prompt=f"Analyze website content for {self.company_name}",
                    required_tools=["analyze_website"]
                ),
                AsyncResearchTask(
                    id="news_analysis",
                    name="News Analysis",
                    description="Analyze recent news",
                    prompt=f"Find and analyze recent news about {self.company_name}",
                    required_tools=["find_news"]
                ),
                AsyncResearchTask(
                    id="competitor_analysis",
                    name="Competitor Analysis",
                    description="Analyze competitors",
                    prompt=f"Analyze competitors of {self.company_name}",
                    required_tools=["analyze_competitors"]
                ),
                AsyncResearchTask(
                    id="financial_analysis",
                    name="Financial Analysis",
                    description="Analyze financials",
                    prompt=f"Analyze financial information for {self.company_name}",
                    required_tools=["extract_financials"]
                )
            ]
            
            # Add tasks to orchestrator
            for task in tasks:
                self.research_orchestrator.add_task(task)
            
            # Execute all research tasks
            results = await self.research_orchestrator.execute_all()
            
            # Compile research results
            research_results = ResearchResults(
                company_info={
                    'name': self.company_name,
                    'location': self.location,
                    'website': self.website,
                    'business_model': self.business_model,
                    'products_services': self.products_services,
                    **results.get('website_analysis', {}).get('content', {})
                },
                market_analysis=results.get('market_analysis', {}).get('content', {}),
                financial_data=results.get('financial_analysis', {}).get('content', {}),
                competitors=results.get('competitor_analysis', {}).get('content', []),
                news_articles=results.get('news_analysis', {}).get('content', []),
                sources=[
                    {'url': url, 'type': 'website'}
                    for url in self.web_navigator.state.visited_urls
                ],
                metadata={
                    'execution_summary': self.research_orchestrator.get_execution_summary(),
                    'navigation_paths': self.web_navigator.state.page_cache
                }
            )
            
            return research_results
            
        except Exception as e:
            logging.error("Error in research process", exc_info=True)
            raise
