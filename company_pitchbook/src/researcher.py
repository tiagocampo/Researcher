from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.web_navigator import WebNavigator
from src.content_analyzer import ContentAnalyzer

class Citation(BaseModel):
    """Source citation"""
    url: str = Field(description="URL of the source")
    title: str = Field(description="Title of the source")
    snippet: str = Field(description="Relevant snippet from the source")
    relevance_score: float = Field(default=0.0, description="Relevance score between 0-1")

class CompanyInfo(BaseModel):
    """Company information"""
    name: str = Field(description="Company name")
    description: Optional[str] = Field(None, description="Company description")
    location: str = Field(description="Company location")
    website: str = Field(description="Company website")
    products_services: List[str] = Field(default_factory=list, description="Products and services")
    market_info: Dict[str, Any] = Field(default_factory=dict, description="Market information")
    competitors: List[str] = Field(default_factory=list, description="Competitors")

class ResearchResult(BaseModel):
    """Research result"""
    company_info: CompanyInfo
    citations: List[Citation] = Field(default_factory=list)
    execution_time: float = Field(default=0.0)

class CompanyResearcher:
    """Handles company research using multiple data sources and AI analysis"""
    
    def __init__(
        self,
        company_name: str,
        location: str,
        website: str,
        business_model: Optional[Dict[str, Any]] = None,
        products_services: Optional[Dict[str, Any]] = None,
        llm: Optional[ChatOpenAI] = None,
        max_retries: int = 3
    ):
        self.company_name = company_name
        self.location = location
        self.website = website
        self.business_model = business_model or {}
        self.products_services = products_services or {}
        
        self.llm = llm or ChatOpenAI(temperature=0, model="gpt-4")
        self.web_navigator = WebNavigator(llm=self.llm, max_retries=max_retries)
        self.content_analyzer = ContentAnalyzer(llm=self.llm)

    async def _search_web(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Perform web search using WebNavigator"""
        return await self.web_navigator.search(query, max_results)

    async def _extract_schema_org(self, html: str) -> Dict[str, Any]:
        """Extract schema.org metadata from HTML"""
        from bs4 import BeautifulSoup
        from src.utils.html_parser import HTMLParser
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            return HTMLParser.extract_schema_metadata(soup) or {}
        except Exception as e:
            logging.error(f"Error extracting schema.org data: {str(e)}")
            return {}

    async def research(self) -> ResearchResult:
        """Perform company research"""
        try:
            start_time = datetime.now()
            company_info = await self._initialize_company_info()
            citations = []
            
            # Try company website first
            website_content = await self.web_navigator.navigate(
                url=self.website,
                research_context={"purpose": "Research company information"},
                max_pages=5
            )
            
            website_error = website_content.get('error')
            if website_error:
                logging.warning(f"Website access error: {website_content.get('error_message')}. Using alternative sources.")
                # Increase search scope when website is inaccessible
                await self._enhanced_search_research(company_info, citations)
            else:
                # Process website content if available
                if website_content:
                    info = await self.content_analyzer.analyze(website_content)
                    self._update_company_info(company_info, info)
                    
                    citations.append(Citation(
                        url=self.website,
                        title=self.company_name,
                        snippet=info.get('description', ''),
                        relevance_score=1.0
                    ))
                
                # Supplement with search results
                await self._supplementary_search_research(company_info, citations)
            
            # Validate and ensure we have meaningful data
            if not citations:
                logging.error("No valid sources found during research")
                raise ValueError("Unable to gather information from any sources")
            
            # Ensure we have at least some basic information
            if not company_info.get('location') or not company_info.get('products_services'):
                await self._fallback_search_research(company_info, citations)
            
            return ResearchResult(
                company_info=CompanyInfo(**company_info),
                citations=citations,
                execution_time=float((datetime.now() - start_time).total_seconds())
            )
            
        except Exception as e:
            logging.error(f"Research failed: {str(e)}")
            raise

    async def _initialize_company_info(self) -> Dict[str, Any]:
        """Initialize company information structure"""
        return {
            'name': self.company_name,
            'location': self.location,
            'website': self.website,
            'products_services': self.products_services.get('main_offerings', []),
            'market_info': {
                'business_model': self.business_model,
                'target_industries': self.business_model.get('target_industries', []),
                'competitive_advantages': self.business_model.get('competitive_advantages', [])
            },
            'competitors': []
        }

    async def _enhanced_search_research(self, company_info: Dict[str, Any], citations: List[Citation]) -> None:
        """Perform enhanced search-based research when website is inaccessible"""
        # Search queries focused on specific information
        search_queries = [
            f"{self.company_name} headquarters location office address",
            f"{self.company_name} products services offerings solutions",
            f"{self.company_name} business model revenue company type",
            f"{self.company_name} market position competitors industry"
        ]
        
        for query in search_queries:
            results = await self.web_navigator.search(query, max_results=5)
            for result in results:
                try:
                    if result['source_type'] in ['business_data', 'professional_network', 'institutional']:
                        content = await self.web_navigator.navigate(
                            url=result['link'],
                            research_context={"purpose": "Research company information"}
                        )
                        
                        if content and not content.get('error'):
                            info = await self.content_analyzer.analyze(content)
                            self._update_company_info(company_info, info)
                            
                            citations.append(Citation(
                                url=result['link'],
                                title=result['title'],
                                snippet=result['snippet'],
                                relevance_score=result['relevance_score']
                            ))
                except Exception as e:
                    logging.warning(f"Error processing search result {result['link']}: {str(e)}")
                    continue

    async def _supplementary_search_research(self, company_info: Dict[str, Any], citations: List[Citation]) -> None:
        """Perform supplementary search-based research"""
        search_results = await self.web_navigator.search(
            f"{self.company_name} company information market competitors",
            max_results=5
        )
        
        for result in results:
            try:
                if result['source_type'] in ['business_data', 'professional_network']:
                    content = await self.web_navigator.navigate(
                        url=result['link'],
                        research_context={"purpose": "Supplement company information"}
                    )
                    
                    if content and not content.get('error'):
                        info = await self.content_analyzer.analyze(content)
                        self._update_company_info(company_info, info)
                        
                        citations.append(Citation(
                            url=result['link'],
                            title=result['title'],
                            snippet=result['snippet'],
                            relevance_score=result['relevance_score']
                        ))
            except Exception as e:
                logging.warning(f"Error processing supplementary result {result['link']}: {str(e)}")
                continue

    async def _fallback_search_research(self, company_info: Dict[str, Any], citations: List[Citation]) -> None:
        """Perform fallback search for missing critical information"""
        missing_info_queries = []
        
        if not company_info.get('location'):
            missing_info_queries.append(f"{self.company_name} headquarters location address")
        
        if not company_info.get('products_services'):
            missing_info_queries.append(f"{self.company_name} main products services offerings")
        
        for query in missing_info_queries:
            results = await self.web_navigator.search(query, max_results=3)
            for result in results:
                try:
                    content = await self.web_navigator.navigate(
                        url=result['link'],
                        research_context={"purpose": "Find missing critical information"}
                    )
                    
                    if content and not content.get('error'):
                        info = await self.content_analyzer.analyze(content)
                        self._update_company_info(company_info, info)
                        
                        citations.append(Citation(
                            url=result['link'],
                            title=result['title'],
                            snippet=result['snippet'],
                            relevance_score=result['relevance_score']
                        ))
                except Exception as e:
                    logging.warning(f"Error in fallback search {result['link']}: {str(e)}")
                    continue

    def _update_company_info(self, company_info: Dict[str, Any], info: Dict[str, Any]) -> None:
        """Update company information with new data"""
        if info.get('description'):
            company_info['description'] = info['description']
        
        # Update location information
        if info.get('location'):
            if isinstance(info['location'], dict):
                company_info['location'] = info['location']
            elif isinstance(info['location'], str):
                company_info['location'] = {'headquarters': info['location'], 'offices': [], 'regions': []}
        
        # Update products/services with deduplication
        if info.get('products_services'):
            if isinstance(info['products_services'], list):
                current_products = set(company_info['products_services'])
                new_products = set(info['products_services'])
                company_info['products_services'] = list(current_products | new_products)
            elif isinstance(info['products_services'], dict):
                for key, value in info['products_services'].items():
                    if key not in company_info['products_services']:
                        company_info['products_services'][key] = value
        
        # Update market info
        if info.get('market_info'):
            company_info['market_info'].update(info['market_info'])
        
        # Update competitors with deduplication
        if info.get('competitors'):
            current_competitors = set(company_info['competitors'])
            new_competitors = set(info['competitors'])
            company_info['competitors'] = list(current_competitors | new_competitors)
