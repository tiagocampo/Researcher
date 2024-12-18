import streamlit as st  # type: ignore
import sys
import asyncio
import logging
from typing import Optional, Dict, Any, Coroutine, TypeVar, TypedDict, cast
from contextlib import asynccontextmanager
import os
from pathlib import Path
import json
from langchain.prompts import ChatPromptTemplate  # type: ignore
from langchain_openai import ChatOpenAI  # type: ignore
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Add parent directory to Python path
current_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(current_dir))

from src.researcher import CompanyResearcher
from src.generator import PitchbookGenerator
from src.utils import load_env_variables, get_api_key
from src.web_navigator import WebNavigator

# Type definitions
class ResearchResults(TypedDict):
    website_data: Optional[Dict[str, Any]]
    search_data: Optional[Dict[str, Any]]
    combined_info: Dict[str, Any]

class CompanyInfo(TypedDict):
    location: Dict[str, Any]
    products_services: Dict[str, Any]
    business_model: Dict[str, Any]

class LocationInfo(BaseModel):
    headquarters: Optional[str] = Field(default=None, description="Main HQ location")
    offices: list[str] = Field(default_factory=list, description="Additional office locations")
    regions: list[str] = Field(default_factory=list, description="Regions where company operates")

class ProductsInfo(BaseModel):
    main_offerings: list[str] = Field(default_factory=list, description="Main products/services")
    categories: list[str] = Field(default_factory=list, description="Product/service categories")
    descriptions: list[str] = Field(default_factory=list, description="Detailed descriptions")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_env_variables()

# Get API key from environment
api_key = get_api_key()
if not api_key:
    st.error("OpenAI API key not found in .env file. Please add it and restart the application.")
    st.stop()

# Set page config
st.set_page_config(
    page_title="Company Pitchbook Generator",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'research_results' not in st.session_state:
    st.session_state.research_results = None
if 'company_info' not in st.session_state:
    st.session_state.company_info = None
if 'company_name' not in st.session_state:
    st.session_state.company_name = ""
if 'company_website' not in st.session_state:
    st.session_state.company_website = ""
if 'location' not in st.session_state:
    st.session_state.location = ""
if 'business_model' not in st.session_state:
    st.session_state.business_model = ""
if 'products_services' not in st.session_state:
    st.session_state.products_services = ""

# Title and description
st.title("🎯 Company Pitchbook Generator")
st.markdown("""
Generate comprehensive pitchbooks for companies using AI-powered research and analysis.
Simply provide the company name and website to get started.
""")

# Set the API key in environment variables for other components
os.environ['OPENAI_API_KEY'] = api_key

T = TypeVar('T')

def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine in the streamlit app"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)

async def initial_research(company_name: str, website: str) -> Dict[str, Any]:
    """Perform initial research on a company"""
    try:
        research_context = {
            "purpose": "Initial company research",
            "company_name": company_name
        }
        
        # Prepare navigation input
        navigation_input = json.dumps({
            "url": website,
            "research_context": research_context,
            "max_pages": 5
        })
        
        async with WebNavigator(llm=self.llm) as navigator:
            website_results = await navigator.navigate(navigation_input)
            
            if website_results.get('error'):
                logging.warning(f"Website access error: {website_results.get('error_message')}")
                # Prepare search input for fallback
                search_input = json.dumps({
                    "query": f"{company_name} company information",
                    "max_results": 10
                })
                search_results = await navigator.search(search_input)
                return {
                    'website_results': website_results,
                    'search_results': search_results
                }
            
            return {
                'website_results': website_results,
                'search_results': []
            }
            
    except Exception as e:
        logging.error(f"Error in initial research: {str(e)}")
        raise

# Input form
with st.form("company_info_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        company_name = st.text_input("Company Name*", value=st.session_state.company_name, placeholder="e.g., Tesla, Inc.")
        company_website = st.text_input("Company Website*", value=st.session_state.company_website, placeholder="e.g., https://tesla.com")
    
    submit_button = st.form_submit_button("Research Company")

if submit_button:
    if not all([company_name, company_website]):
        st.error("Please provide both company name and website.")
    else:
        try:
            progress_placeholder = st.empty()
            with st.spinner("Researching company information..."):
                # Save to session state
                st.session_state.company_name = company_name
                st.session_state.company_website = company_website
                
                progress_placeholder.info("Starting initial research...")
                
                # Perform initial research
                company_info = run_async(initial_research(company_name, company_website))
                
                if company_info:
                    st.session_state.company_info = company_info
                    
                    # Show editable results
                    st.success("Initial research completed! Please review and edit the information below.")
                    
                    # Create columns for editing
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Location information
                        st.subheader("Company Location")
                        location_data = company_info.get('location', {})
                        if not location_data:
                            st.warning("No location information found.")
                            location_data = {}
                        
                        headquarters = st.text_input(
                            "Headquarters",
                            value=location_data.get('headquarters', ''),
                            key="headquarters_input",
                            help="Enter the company's headquarters location"
                        )
                        offices = st.text_area(
                            "Office Locations",
                            value="\n".join(location_data.get('offices', [])),
                            height=100,
                            key="offices_input",
                            help="Enter office locations, one per line"
                        )
                        regions = st.text_area(
                            "Operating Regions",
                            value="\n".join(location_data.get('regions', [])),
                            height=100,
                            key="regions_input",
                            help="Enter operating regions, one per line"
                        )
                        
                        # Update location in session state
                        st.session_state.location = {
                            'headquarters': headquarters,
                            'offices': [office.strip() for office in offices.split('\n') if office.strip()],
                            'regions': [region.strip() for region in regions.split('\n') if region.strip()]
                        }
                        
                        # Business Model information
                        st.subheader("Business Model")
                        business_data = company_info.get('business_model', {})
                        if not business_data:
                            st.warning("No business model information found.")
                            business_data = {}
                        
                        business_type = st.text_input(
                            "Business Type",
                            value=business_data.get('type', ''),
                            key="business_type_input",
                            help="Enter the type of business (e.g., B2B, B2C, SaaS)"
                        )
                        revenue_model = st.text_input(
                            "Revenue Model",
                            value=business_data.get('revenue_model', ''),
                            key="revenue_model_input",
                            help="Enter the revenue model (e.g., Subscription, Licensing)"
                        )
                        market_position = st.text_input(
                            "Market Position",
                            value=business_data.get('market_position', ''),
                            key="market_position_input",
                            help="Enter the market position"
                        )
                        target_industries = st.text_area(
                            "Target Industries",
                            value="\n".join(business_data.get('target_industries', [])),
                            height=100,
                            key="target_industries_input",
                            help="Enter target industries, one per line"
                        )
                        competitive_advantages = st.text_area(
                            "Competitive Advantages",
                            value="\n".join(business_data.get('competitive_advantages', [])),
                            height=100,
                            key="competitive_advantages_input",
                            help="Enter competitive advantages, one per line"
                        )
                        
                        # Update business model in session state
                        st.session_state.business_model = {
                            'type': business_type,
                            'revenue_model': revenue_model,
                            'market_position': market_position,
                            'target_industries': [ind.strip() for ind in target_industries.split('\n') if ind.strip()],
                            'competitive_advantages': [adv.strip() for adv in competitive_advantages.split('\n') if adv.strip()]
                        }
                    
                    with col2:
                        # Products/Services information
                        st.subheader("Products & Services")
                        products_data = company_info.get('products_services', {})
                        if not products_data:
                            st.warning("No products/services information found.")
                            products_data = {}
                        
                        main_offerings = st.text_area(
                            "Main Offerings",
                            value="\n".join(products_data.get('main_offerings', [])),
                            height=100,
                            key="main_offerings_input",
                            help="Enter main product/service offerings, one per line"
                        )
                        categories = st.text_area(
                            "Categories",
                            value="\n".join(products_data.get('categories', [])),
                            height=100,
                            key="categories_input",
                            help="Enter product/service categories, one per line"
                        )
                        descriptions = st.text_area(
                            "Descriptions",
                            value="\n".join(products_data.get('descriptions', [])),
                            height=200,
                            key="descriptions_input",
                            help="Enter detailed descriptions, one per line"
                        )
                        
                        # Update products/services in session state
                        st.session_state.products_services = {
                            'main_offerings': [off.strip() for off in main_offerings.split('\n') if off.strip()],
                            'categories': [cat.strip() for cat in categories.split('\n') if cat.strip()],
                            'descriptions': [desc.strip() for desc in descriptions.split('\n') if desc.strip()]
                        }
                    
                    # Show debug information in expander
                    with st.expander("Debug Information"):
                        st.json({
                            'company_info': company_info,
                            'session_state': {
                                'location': st.session_state.location,
                                'business_model': st.session_state.business_model,
                                'products_services': st.session_state.products_services
                            }
                        })
                else:
                    st.error("No information could be retrieved. Please check the company website and try again.")
                
        except Exception as e:
            logger.error("Error in initial research", exc_info=True)
            st.error(f"An error occurred during research: {str(e)}")
            st.error("Please check the logs for more details.")
            
            # Show detailed error information in expander
            with st.expander("Error Details"):
                st.code(str(e))
                import traceback
                st.code(traceback.format_exc())

# Generate pitchbook button (outside the form)
if st.session_state.company_info is not None:
    if st.button("Generate Pitchbook", key="generate_pitchbook"):
        try:
            with st.spinner("Generating comprehensive pitchbook..."):
                # Initialize researcher
                researcher = CompanyResearcher(
                    company_name=st.session_state.company_name,
                    location=st.session_state.location,
                    website=st.session_state.company_website,
                    business_model=st.session_state.business_model,
                    products_services=st.session_state.products_services,
                    llm=ChatOpenAI(
                        api_key=os.environ['OPENAI_API_KEY'],
                        temperature=0,
                        model="gpt-4o-mini"
                    )
                )
                
                # Run research
                research_results = run_async(researcher.research())
                st.session_state.research_results = research_results
                
                if research_results:
                    # Create generator
                    generator = PitchbookGenerator(research_results)
                    
                    # Generate PDF
                    pdf_bytes = generator.generate()
                    
                    st.success("Pitchbook generated successfully!")
                    
                    # Create tabs for different sections
                    tabs = st.tabs([
                        "Overview",
                        "Download PDF"
                    ])
                    
                    with tabs[0]:
                        # Display research results in a structured way
                        st.subheader("Company Overview")
                        if isinstance(research_results, dict):
                            overview = research_results.get('results', {}).get('website_analysis', {}).get('content', {})
                            st.json(overview)
                        
                        st.subheader("Market Analysis")
                        if isinstance(research_results, dict):
                            market = research_results.get('results', {}).get('market_analysis', {}).get('content', {})
                            st.json(market)
                        
                        st.subheader("Business Model")
                        if isinstance(research_results, dict):
                            business = research_results.get('results', {}).get('competitor_analysis', {}).get('content', {})
                            st.json(business)
                        
                        st.subheader("Products & Services")
                        if isinstance(research_results, dict):
                            products = research_results.get('results', {}).get('website_analysis', {}).get('content', {}).get('products_services', {})
                            st.json(products)
                        
                        st.subheader("Financial Analysis")
                        if isinstance(research_results, dict):
                            financials = research_results.get('results', {}).get('financial_analysis', {}).get('content', {})
                            st.json(financials)
                        
                        st.subheader("SWOT Analysis")
                        if isinstance(research_results, dict):
                            swot = research_results.get('results', {}).get('competitor_analysis', {}).get('content', {}).get('swot', {})
                            if swot:
                                st.json(swot)
                    
                    with tabs[1]:
                        # Download button for PDF
                        st.download_button(
                            label="Download Pitchbook (PDF)",
                            data=pdf_bytes,
                            file_name=f"{st.session_state.company_name.lower().replace(' ', '_')}_pitchbook.pdf",
                            mime="application/pdf"
                        )
        except Exception as e:
            logger.error("Error generating pitchbook", exc_info=True)
            st.error(f"An error occurred while generating the pitchbook: {str(e)}")
