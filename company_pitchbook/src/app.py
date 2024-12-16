import streamlit as st
from pathlib import Path
import sys
import asyncio
import logging
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.researcher import CompanyResearcher
from src.generator import PitchbookGenerator
from src.utils import load_env_variables
from src.web_navigator import WebNavigator
from langchain_openai import ChatOpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_env_variables()

# Set page config
st.set_page_config(
    page_title="Company Pitchbook Generator",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("🎯 Company Pitchbook Generator")
st.markdown("""
Generate comprehensive pitchbooks for companies using AI-powered research and analysis.
Simply provide the company name and website to get started.
""")

@asynccontextmanager
async def get_event_loop():
    """Get or create an event loop"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    try:
        yield loop
    finally:
        if not loop.is_closed():
            loop.close()

async def initial_research(company_name: str, website: str) -> Dict[str, Any]:
    """Perform initial research to gather company information"""
    try:
        llm = ChatOpenAI(
            temperature=0,
            model="gpt-4-turbo-preview"
        )
        
        # Initialize web navigator for initial research
        navigator = WebNavigator(
            llm=llm,
            max_retries=3,
            max_concurrent_requests=5,
            request_delay=1.0
        )
        
        # Research context for initial scan
        research_context = {
            'company_name': company_name,
            'website': website,
            'objective': 'initial_info'
        }
        
        # Navigate company website
        results = await navigator.navigate(
            start_url=website,
            research_context=research_context,
            max_pages=5  # Limit initial scan
        )
        
        # Extract company information
        company_info = {}
        for url, data in results['extracted_data'].items():
            if isinstance(data, dict):
                # Extract location information
                if 'location' in data:
                    company_info['location'] = data['location']
                elif 'headquarters' in data:
                    company_info['location'] = data['headquarters']
                elif 'address' in data:
                    company_info['location'] = data['address']
                
                # Extract business model
                if 'business_model' in data:
                    company_info['business_model'] = data['business_model']
                elif 'about' in data:
                    company_info['business_model'] = data['about']
                
                # Extract products/services
                if 'products' in data:
                    company_info['products_services'] = data['products']
                elif 'services' in data:
                    company_info['products_services'] = data['services']
                elif 'offerings' in data:
                    company_info['products_services'] = data['offerings']
        
        # Fill in missing information with default values
        company_info.setdefault('location', "Location not found - please provide")
        company_info.setdefault('business_model', "Business model not found - please provide")
        company_info.setdefault('products_services', "Products/Services not found - please provide")
        
        return company_info
        
    except Exception as e:
        logger.error("Error in initial research", exc_info=True)
        return {
            'location': "Error finding location - please provide",
            'business_model': "Error finding business model - please provide",
            'products_services': "Error finding products/services - please provide"
        }

def run_async(coro):
    """Run an async coroutine in the streamlit app"""
    async def wrapped():
        async with get_event_loop() as loop:
            return await coro
    return asyncio.run(wrapped())

# Input form
with st.form("company_info_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        company_name = st.text_input("Company Name*", placeholder="e.g., Tesla, Inc.")
        company_website = st.text_input("Company Website*", placeholder="e.g., https://tesla.com")
    
    submit_button = st.form_submit_button("Research Company")

if submit_button:
    if not all([company_name, company_website]):
        st.error("Please provide both company name and website.")
    else:
        try:
            with st.spinner("Researching company information..."):
                # Perform initial research
                company_info = run_async(initial_research(company_name, company_website))
                
                # Show editable results
                st.success("Initial research completed! Please review and edit the information below.")
                
                # Create columns for editing
                col1, col2 = st.columns(2)
                
                with col1:
                    location = st.text_area(
                        "Company Location",
                        value=company_info['location'],
                        height=100
                    )
                    business_model = st.text_area(
                        "Business Model",
                        value=company_info['business_model'],
                        height=100
                    )
                
                with col2:
                    products_services = st.text_area(
                        "Products/Services",
                        value=company_info['products_services'],
                        height=200
                    )
                
                # Generate pitchbook button
                if st.button("Generate Pitchbook"):
                    with st.spinner("Generating comprehensive pitchbook..."):
                        # Initialize researcher
                        researcher = CompanyResearcher(
                            company_name=company_name,
                            location=location,
                            website=company_website,
                            business_model=business_model,
                            products_services=products_services
                        )
                        
                        # Run research
                        research_results = run_async(researcher.research())
                        
                        if research_results:
                            # Create generator
                            generator = PitchbookGenerator(research_results)
                            
                            # Get results
                            pitchbook = generator.generate()
                            
                            st.success("Pitchbook generated successfully!")
                            
                            # Create tabs for different sections
                            tabs = st.tabs([
                                "Company Overview",
                                "Market Analysis",
                                "Business Model",
                                "Products & Services",
                                "Financial Analysis",
                                "SWOT Analysis"
                            ])
                            
                            with tabs[0]:
                                st.markdown(pitchbook.overview)
                            
                            with tabs[1]:
                                st.markdown(pitchbook.market_analysis)
                            
                            with tabs[2]:
                                st.markdown(pitchbook.business_model)
                            
                            with tabs[3]:
                                st.markdown(pitchbook.products_services)
                            
                            with tabs[4]:
                                st.markdown(pitchbook.financial_analysis)
                            
                            with tabs[5]:
                                st.markdown(pitchbook.swot_analysis)
                            
                            # Download button for PDF
                            st.download_button(
                                label="Download Pitchbook (PDF)",
                                data=generator.get_pdf(),
                                file_name=f"{company_name.lower().replace(' ', '_')}_pitchbook.pdf",
                                mime="application/pdf"
                            )
                
        except Exception as e:
            logger.error("Error in main app flow", exc_info=True)
            st.error(f"An unexpected error occurred: {str(e)}")
            raise
