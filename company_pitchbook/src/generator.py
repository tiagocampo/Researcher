from typing import Dict, Any, List
import streamlit as st
from pathlib import Path
import base64
from .researcher import ResearchResult
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Flowable
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import logging

class PitchbookGenerator:
    """Generates PDF pitchbooks from research results"""
    
    def __init__(self, research_results: ResearchResult):
        self.research = research_results
        self.styles = getSampleStyleSheet()
        self._setup_styles()
    
    def _setup_styles(self):
        """Create custom styles for the PDF"""
        self.styles['Heading1'].fontSize = 24
        self.styles['Heading1'].spaceAfter = 30
        
        self.styles['Heading2'].fontSize = 18
        self.styles['Heading2'].spaceAfter = 20
        
        self.styles['BodyText'].fontSize = 12
        self.styles['BodyText'].spaceAfter = 12
    
    def generate(self) -> bytes:
        """Generate the PDF report"""
        try:
            # Create the document
            doc = SimpleDocTemplate(
                self.output_buffer,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            # Create story
            story = []
            
            # Title
            story.append(Paragraph(
                f"Company Analysis: {self.research.overview.get('name', self.research.overview.get('company_name', 'Unknown Company'))}",
                self.styles['Heading1']
            ))
            
            # Executive Summary
            story.append(Paragraph("Executive Summary", self.styles['Heading2']))
            story.extend(self._generate_executive_summary())
            story.append(Spacer(1, 12))
            
            # Company Overview
            story.append(Paragraph("Company Overview", self.styles['Heading2']))
            story.extend(self._generate_company_overview())
            story.append(Spacer(1, 12))
            
            # Market Analysis
            story.append(Paragraph("Market Analysis", self.styles['Heading2']))
            story.extend(self._generate_market_analysis())
            story.append(Spacer(1, 12))
            
            # Market Deep Dive
            story.append(Paragraph("Market Deep Dive", self.styles['Heading2']))
            story.extend(self._generate_market_deep_dive())
            story.append(Spacer(1, 12))
            
            # Products & Services
            story.append(Paragraph("Products & Services", self.styles['Heading2']))
            story.extend(self._generate_products_services())
            story.append(Spacer(1, 12))
            
            # Competitive Analysis
            story.append(Paragraph("Competitive Analysis", self.styles['Heading2']))
            story.extend(self._generate_competitive_analysis())
            story.append(Spacer(1, 12))
            
            # Financial Analysis
            story.append(Paragraph("Financial Analysis", self.styles['Heading2']))
            story.extend(self._generate_financial_analysis())
            story.append(Spacer(1, 12))
            
            # Sector M&A Analysis
            story.append(Paragraph("Sector M&A Analysis", self.styles['Heading2']))
            story.extend(self._generate_sector_ma_analysis())
            story.append(Spacer(1, 12))
            
            # Recent News & Developments
            story.append(Paragraph("Recent News & Developments", self.styles['Heading2']))
            story.extend(self._generate_news_analysis())
            story.append(Spacer(1, 12))
            
            # Sources & Citations
            story.append(Paragraph("Sources & Citations", self.styles['Heading2']))
            story.extend(self._generate_citations())
            
            # Build the document
            doc.build(story)
            
            # Get the value from the buffer
            pdf_bytes = self.output_buffer.getvalue()
            self.output_buffer.close()
            
            return pdf_bytes
            
        except Exception as e:
            logging.error(f"Error generating PDF: {str(e)}")
            raise
    
    def get_pdf(self) -> bytes:
        """Get the generated PDF bytes"""
        return self.generate()

    def _generate_executive_summary(self) -> List[Flowable]:
        """Generate executive summary section"""
        summary = []
        
        # Company snapshot
        summary.append(Paragraph("Company Snapshot", self.styles['Heading3']))
        snapshot_items = [
            f"Company: {self.research.overview.get('name', 'Unknown')}",
            f"Industry: {self.research.overview.get('industry', 'Unknown')}",
            f"Founded: {self.research.overview.get('founded', 'Unknown')}",
            f"Headquarters: {self.research.overview.get('location', 'Unknown')}"
        ]
        for item in snapshot_items:
            summary.append(Paragraph(item, self.styles['Normal']))
        summary.append(Spacer(1, 6))
        
        # Key highlights
        summary.append(Paragraph("Key Highlights", self.styles['Heading3']))
        if self.research.overview.get('description'):
            summary.append(Paragraph(self.research.overview['description'], self.styles['Normal']))
        
        # Market position
        if self.research.market_deep_dive:
            market = self.research.market_deep_dive
            if market.market_sizing:
                summary.append(Paragraph("Market Opportunity", self.styles['Heading3']))
                market_items = [
                    f"TAM: {market.market_sizing.total_addressable_market or 'Unknown'}",
                    f"SAM: {market.market_sizing.serviceable_addressable_market or 'Unknown'}",
                    f"SOM: {market.market_sizing.serviceable_obtainable_market or 'Unknown'}"
                ]
                for item in market_items:
                    summary.append(Paragraph(item, self.styles['Normal']))
        
        return summary

    def _generate_market_deep_dive(self) -> List[Flowable]:
        """Generate market deep dive section"""
        content = []
        
        if not self.research.market_deep_dive:
            content.append(Paragraph("Detailed market analysis not available.", self.styles['Normal']))
            return content
        
        market = self.research.market_deep_dive
        
        # Market sizing
        if market.market_sizing:
            content.append(Paragraph("Market Size Analysis", self.styles['Heading3']))
            sizing_items = [
                f"Total Addressable Market (TAM): {market.market_sizing.total_addressable_market or 'Unknown'}",
                f"Serviceable Addressable Market (SAM): {market.market_sizing.serviceable_addressable_market or 'Unknown'}",
                f"Serviceable Obtainable Market (SOM): {market.market_sizing.serviceable_obtainable_market or 'Unknown'}"
            ]
            for item in sizing_items:
                content.append(Paragraph(item, self.styles['Normal']))
            
            if market.market_sizing.assumptions:
                content.append(Paragraph("Key Assumptions:", self.styles['Heading4']))
                for assumption in market.market_sizing.assumptions:
                    content.append(Paragraph(f"• {assumption}", self.styles['ListBullet']))
            
            if market.market_sizing.growth_drivers:
                content.append(Paragraph("Growth Drivers:", self.styles['Heading4']))
                for driver in market.market_sizing.growth_drivers:
                    content.append(Paragraph(f"• {driver}", self.styles['ListBullet']))
        
        # Regulatory environment
        if market.regulatory_environment:
            content.append(Paragraph("Regulatory Environment", self.styles['Heading3']))
            for reg in market.regulatory_environment:
                content.append(Paragraph(f"Region: {reg.get('region', 'Unknown')}", self.styles['Heading4']))
                content.append(Paragraph(f"Regulations: {reg.get('regulations', 'Unknown')}", self.styles['Normal']))
                content.append(Paragraph(f"Impact: {reg.get('impact', 'Unknown')}", self.styles['Normal']))
                content.append(Spacer(1, 6))
        
        # Geographic analysis
        if market.geographic_analysis:
            content.append(Paragraph("Geographic Analysis", self.styles['Heading3']))
            geo = market.geographic_analysis
            
            if geo.get('key_markets'):
                content.append(Paragraph("Key Markets:", self.styles['Heading4']))
                for market_name in geo['key_markets']:
                    content.append(Paragraph(f"• {market_name}", self.styles['ListBullet']))
            
            if geo.get('market_maturity'):
                content.append(Paragraph("Market Maturity by Region:", self.styles['Heading4']))
                for region, maturity in geo['market_maturity'].items():
                    content.append(Paragraph(f"• {region}: {maturity}", self.styles['ListBullet']))
            
            if geo.get('growth_opportunities'):
                content.append(Paragraph("Growth Opportunities:", self.styles['Heading4']))
                for opportunity in geo['growth_opportunities']:
                    content.append(Paragraph(f"• {opportunity}", self.styles['ListBullet']))
        
        # Entry barriers and success factors
        if market.entry_barriers:
            content.append(Paragraph("Market Entry Barriers", self.styles['Heading3']))
            for barrier in market.entry_barriers:
                content.append(Paragraph(f"• {barrier}", self.styles['ListBullet']))
        
        if market.success_factors:
            content.append(Paragraph("Critical Success Factors", self.styles['Heading3']))
            for factor in market.success_factors:
                content.append(Paragraph(f"• {factor}", self.styles['ListBullet']))
        
        return content

    def _generate_financial_analysis(self) -> List[Flowable]:
        """Generate financial analysis section"""
        content = []
        
        if not self.research.financials:
            content.append(Paragraph("Financial information not available.", self.styles['Normal']))
            return content
        
        financials = self.research.financials
        
        # Key metrics
        if financials.metrics:
            content.append(Paragraph("Key Financial Metrics", self.styles['Heading3']))
            for category, metrics in financials.metrics.items():
                content.append(Paragraph(category.title(), self.styles['Heading4']))
                for metric, value in metrics.items():
                    content.append(Paragraph(f"• {metric}: {value}", self.styles['ListBullet']))
                content.append(Spacer(1, 6))
        
        # Valuation metrics
        if financials.valuation_metrics:
            content.append(Paragraph("Valuation Metrics", self.styles['Heading3']))
            for metric, value in financials.valuation_metrics.items():
                content.append(Paragraph(f"• {metric}: {value}", self.styles['ListBullet']))
        
        # Financial ratios
        if financials.financial_ratios:
            content.append(Paragraph("Financial Ratios", self.styles['Heading3']))
            for ratio, details in financials.financial_ratios.items():
                content.append(Paragraph(f"• {ratio}:", self.styles['ListBullet']))
                content.append(Paragraph(f"  Value: {details.get('value', 'Unknown')}", self.styles['ListBullet2']))
                content.append(Paragraph(f"  Interpretation: {details.get('interpretation', '')}", self.styles['ListBullet2']))
        
        # Growth projections
        if financials.growth_projections:
            content.append(Paragraph("Growth Projections", self.styles['Heading3']))
            for timeframe, projection in financials.growth_projections.items():
                content.append(Paragraph(f"• {timeframe}: {projection}", self.styles['ListBullet']))
        
        # Funding history
        if financials.funding_rounds:
            content.append(Paragraph("Funding History", self.styles['Heading3']))
            for round_info in financials.funding_rounds:
                content.append(Paragraph(
                    f"• {round_info.get('date', 'Unknown')} - {round_info.get('type', 'Unknown')}",
                    self.styles['ListBullet']
                ))
                content.append(Paragraph(f"  Amount: {round_info.get('amount', 'Unknown')}", self.styles['ListBullet2']))
                content.append(Paragraph(f"  Valuation: {round_info.get('valuation', 'Unknown')}", self.styles['ListBullet2']))
                if round_info.get('investors'):
                    content.append(Paragraph("  Investors:", self.styles['ListBullet2']))
                    for investor in round_info['investors']:
                        content.append(Paragraph(f"    - {investor}", self.styles['ListBullet3']))
                if round_info.get('use_of_proceeds'):
                    content.append(Paragraph(f"  Use of Proceeds: {round_info['use_of_proceeds']}", self.styles['ListBullet2']))
                content.append(Spacer(1, 6))
        
        return content

    def _generate_sector_ma_analysis(self) -> List[Flowable]:
        """Generate sector M&A analysis section"""
        content = []
        
        if not self.research.sector_ma:
            content.append(Paragraph("Sector M&A analysis not available.", self.styles['Normal']))
            return content
        
        sector = self.research.sector_ma
        
        # Recent deals
        if sector.recent_deals:
            content.append(Paragraph("Recent M&A Transactions", self.styles['Heading3']))
            for deal in sector.recent_deals:
                content.append(Paragraph(
                    f"• {deal.date} - {deal.acquirer} / {deal.target}",
                    self.styles['ListBullet']
                ))
                if deal.value:
                    content.append(Paragraph(f"  Deal Value: {deal.value}", self.styles['ListBullet2']))
                content.append(Paragraph(f"  Type: {deal.type}", self.styles['ListBullet2']))
                if deal.rationale:
                    content.append(Paragraph(f"  Rationale: {deal.rationale}", self.styles['ListBullet2']))
                if deal.multiples:
                    content.append(Paragraph("  Multiples:", self.styles['ListBullet2']))
                    for multiple, value in deal.multiples.items():
                        content.append(Paragraph(f"    - {multiple}: {value}", self.styles['ListBullet3']))
                content.append(Paragraph(f"  Status: {deal.status}", self.styles['ListBullet2']))
                content.append(Spacer(1, 6))
        
        # Valuation metrics
        if sector.valuation_metrics:
            content.append(Paragraph("Sector Valuation Metrics", self.styles['Heading3']))
            for metric, details in sector.valuation_metrics.items():
                content.append(Paragraph(f"• {metric}:", self.styles['ListBullet']))
                if isinstance(details, dict):
                    for key, value in details.items():
                        content.append(Paragraph(f"  - {key}: {value}", self.styles['ListBullet2']))
                else:
                    content.append(Paragraph(f"  {details}", self.styles['ListBullet2']))
        
        # Consolidation trends
        if sector.consolidation_trends:
            content.append(Paragraph("Industry Consolidation Trends", self.styles['Heading3']))
            for trend in sector.consolidation_trends:
                content.append(Paragraph(f"• {trend}", self.styles['ListBullet']))
        
        # Key players
        if sector.key_players:
            content.append(Paragraph("Key Strategic Players", self.styles['Heading3']))
            for player in sector.key_players:
                content.append(Paragraph(f"• {player}", self.styles['ListBullet']))
        
        return content

    def _generate_citations(self) -> List[Flowable]:
        """Generate sources and citations section"""
        content = []
        
        if not self.research.citations:
            content.append(Paragraph("No citations available.", self.styles['Normal']))
            return content
        
        # Group citations by reliability score
        citations_by_reliability = {}
        for citation in self.research.citations:
            score_bucket = round(citation.reliability_score * 2) / 2  # Round to nearest 0.5
            if score_bucket not in citations_by_reliability:
                citations_by_reliability[score_bucket] = []
            citations_by_reliability[score_bucket].append(citation)
        
        # Sort buckets by reliability score (descending)
        for score in sorted(citations_by_reliability.keys(), reverse=True):
            content.append(Paragraph(f"Sources (Reliability Score: {score})", self.styles['Heading3']))
            for citation in citations_by_reliability[score]:
                content.append(Paragraph(f"• {citation.title}", self.styles['ListBullet']))
                content.append(Paragraph(f"  URL: {citation.url}", self.styles['ListBullet2']))
                content.append(Paragraph(f"  Relevance Score: {citation.relevance_score}", self.styles['ListBullet2']))
                if citation.snippet:
                    content.append(Paragraph(f"  Summary: {citation.snippet}", self.styles['ListBullet2']))
                content.append(Spacer(1, 6))
        
        return content
