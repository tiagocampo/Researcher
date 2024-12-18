from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

class WebPage(BaseModel):
    """Represents a webpage"""
    url: str
    title: str
    content: str
    html: str

class LocationInfo(BaseModel):
    """Location information"""
    headquarters: Optional[str] = None
    offices: List[str] = Field(default_factory=list)
    regions: List[str] = Field(default_factory=list)

class ProductServiceInfo(BaseModel):
    """Product/service information"""
    main_offerings: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    descriptions: List[str] = Field(default_factory=list)
    features: List[str] = Field(default_factory=list)
    target_markets: List[str] = Field(default_factory=list)

class BusinessModelInfo(BaseModel):
    """Business model information"""
    type: Optional[str] = None
    revenue_model: Optional[str] = None
    market_position: Optional[str] = None
    target_industries: List[str] = Field(default_factory=list)
    competitive_advantages: List[str] = Field(default_factory=list)

class CompanyInfo(BaseModel):
    """Company information"""
    name: Optional[str] = None
    description: Optional[str] = None
    founded: Optional[str] = None
    size: Optional[str] = None
    industry: Optional[str] = None

class CompanyAnalysis(BaseModel):
    """Complete company analysis"""
    company_info: CompanyInfo = Field(default_factory=CompanyInfo)
    location: LocationInfo = Field(default_factory=LocationInfo)
    products_services: ProductServiceInfo = Field(default_factory=ProductServiceInfo)
    business_model: BusinessModelInfo = Field(default_factory=BusinessModelInfo) 