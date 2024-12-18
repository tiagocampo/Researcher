from typing import Dict, List, Optional, Any
from bs4 import BeautifulSoup
import json
import logging

class HTMLParser:
    """Utility class for HTML parsing and metadata extraction"""
    
    # Common selectors for metadata extraction
    COMPANY_SELECTORS = {
        'name': ['.company-name', '#company-name', '[data-company]'],
        'description': ['.about-us', '#about', '.company-description'],
        'founded': ['.founded-date', '.establishment-date'],
        'size': ['.company-size', '.employee-count'],
        'industry': ['.industry', '.sector']
    }
    
    LOCATION_SELECTORS = {
        'headquarters': ['[itemtype*="PostalAddress"]', '.headquarters', '.hq-address'],
        'offices': ['.office-location', '.contact-address', 'address'],
        'regions': ['.regions-served', '.market-presence', '.global-presence']
    }
    
    PRODUCT_SELECTORS = {
        'main_offerings': ['[itemtype*="Product"]', '.products', '.services'],
        'categories': ['.product-category', '.service-type'],
        'descriptions': ['.product-description', '.service-description']
    }
    
    @staticmethod
    def extract_main_content(soup: BeautifulSoup) -> str:
        """Extract main content from a web page"""
        # Remove unwanted elements
        for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        # Try to find main content area
        main_content = None
        
        # Try schema.org markup first
        for element in soup.find_all(True, attrs={'itemtype': True}):
            if 'Organization' in element.get('itemtype', '') or 'LocalBusiness' in element.get('itemtype', ''):
                main_content = element
                break
        
        # Then try common content areas
        if not main_content:
            for selector in ['main', 'article', '[role="main"]', '#content', '.content', '#main', '.main']:
                main_content = soup.select_one(selector)
                if main_content:
                    break
        
        # Fallback to body
        if not main_content:
            main_content = soup.find('body')
        
        return main_content.get_text(separator=' ', strip=True) if main_content else ''
    
    @staticmethod
    def extract_schema_metadata(soup: BeautifulSoup) -> Optional[Dict[str, Any]]:
        """Extract schema.org metadata from the page"""
        try:
            schema_data = {}
            
            # Look for JSON-LD schema data
            for script in soup.find_all('script', type='application/ld+json'):
                try:
                    data = json.loads(script.string)
                    if isinstance(data, dict):
                        if data.get('@type') in ['Organization', 'Corporation', 'LocalBusiness']:
                            schema_data.update(HTMLParser.process_schema_org_data(data))
                except json.JSONDecodeError:
                    continue
            
            # Look for microdata
            for element in soup.find_all(True, attrs={'itemtype': True}):
                if 'Organization' in element.get('itemtype', '') or 'LocalBusiness' in element.get('itemtype', ''):
                    props = {}
                    for prop in element.find_all(True, attrs={'itemprop': True}):
                        props[prop['itemprop']] = prop.get_text(strip=True)
                    schema_data.update(HTMLParser.process_schema_org_data(props))
            
            return schema_data if schema_data else None
            
        except Exception as e:
            logging.error(f"Error extracting schema metadata: {str(e)}")
            return None
    
    @staticmethod
    def process_schema_org_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Process schema.org data into our structure"""
        processed: Dict[str, Dict[str, Any]] = {
            'company_info': {},
            'location': {},
            'products_services': {}
        }
        
        # Map schema.org fields to our structure
        mappings = {
            'company_info': {
                'name': ['name', 'legalName'],
                'description': ['description'],
                'founded': ['foundingDate'],
                'size': ['numberOfEmployees'],
                'industry': ['industry']
            },
            'location': {
                'headquarters': ['address', 'location'],
                'regions': ['areaServed']
            },
            'products_services': {
                'main_offerings': ['makesOffer', 'hasOfferCatalog'],
                'descriptions': ['productSupported', 'serviceType']
            }
        }
        
        for category, fields in mappings.items():
            for our_field, schema_fields in fields.items():
                for schema_field in schema_fields:
                    if schema_field in data:
                        value = data[schema_field]
                        if isinstance(value, (list, set)):
                            processed[category][our_field] = list(value)
                        else:
                            processed[category][our_field] = value
                        break
        
        return processed
    
    @staticmethod
    def extract_html_metadata(soup: BeautifulSoup) -> Optional[Dict[str, Any]]:
        """Extract metadata from HTML using common selectors"""
        try:
            metadata: Dict[str, Dict[str, Any]] = {
                'company_info': {},
                'location': {},
                'products_services': {}
            }
            
            # Extract company information
            for field, selectors in HTMLParser.COMPANY_SELECTORS.items():
                for selector in selectors:
                    elements = soup.select(selector)
                    if elements:
                        metadata['company_info'][field] = elements[0].get_text(strip=True)
                        break
            
            # Extract location information
            for field, selectors in HTMLParser.LOCATION_SELECTORS.items():
                values = []
                for selector in selectors:
                    elements = soup.select(selector)
                    values.extend([elem.get_text(strip=True) for elem in elements])
                if values:
                    if field == 'headquarters':
                        metadata['location'][field] = values[0]
                    else:
                        metadata['location'][field] = values
            
            # Extract products/services information
            for field, selectors in HTMLParser.PRODUCT_SELECTORS.items():
                values = []
                for selector in selectors:
                    elements = soup.select(selector)
                    values.extend([elem.get_text(strip=True) for elem in elements])
                if values:
                    metadata['products_services'][field] = values
            
            return metadata if any(metadata.values()) else None
            
        except Exception as e:
            logging.error(f"Error extracting HTML metadata: {str(e)}")
            return None 