from typing import Dict, Any
import streamlit as st
from pathlib import Path
import base64
from .researcher import CompanyResearchResult
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

class PitchbookGenerator:
    def __init__(self, research_results: CompanyResearchResult):
        self.research = research_results
        self.sections = {
            "overview": self.research.overview,
            "market_analysis": self.research.market_analysis,
            "business_model": self.research.business_model,
            "products_services": self.research.products_services,
            "financial_analysis": self.research.financial_analysis,
            "swot_analysis": self.research.swot_analysis
        }
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
    
    def _create_custom_styles(self):
        """Create custom styles for the PDF"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#2c3e50')
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=20,
            textColor=colors.HexColor('#34495e')
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=12,
            leading=14,
            textColor=colors.HexColor('#2c3e50')
        ))
    
    def generate(self) -> CompanyResearchResult:
        """Generate the pitchbook content"""
        return self.research
    
    def _create_swot_table(self) -> Table:
        """Create SWOT analysis table"""
        swot_data = [
            ['Strengths', 'Weaknesses'],
            ['Opportunities', 'Threats']
        ]
        
        table = Table(swot_data, colWidths=[3*inch, 3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, 0), colors.HexColor('#27ae60')),
            ('BACKGROUND', (1, 0), (1, 0), colors.HexColor('#c0392b')),
            ('BACKGROUND', (0, 1), (0, 1), colors.HexColor('#2980b9')),
            ('BACKGROUND', (1, 1), (1, 1), colors.HexColor('#d35400')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 14),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        return table
    
    def get_pdf(self) -> bytes:
        """Generate a PDF version of the pitchbook"""
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Build the document
        story = []
        
        # Title
        story.append(Paragraph("Company Pitchbook", self.styles['CustomTitle']))
        story.append(Spacer(1, 12))
        
        # Sections
        for section, content in self.sections.items():
            # Section title
            title = section.replace('_', ' ').title()
            story.append(Paragraph(title, self.styles['CustomHeading']))
            story.append(Spacer(1, 12))
            
            # Section content
            if section == 'swot_analysis':
                story.append(self._create_swot_table())
            else:
                paragraphs = content.split('\n\n')
                for p in paragraphs:
                    if p.strip():
                        story.append(Paragraph(p, self.styles['CustomBody']))
            
            story.append(Spacer(1, 20))
        
        # Build PDF
        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        return pdf_bytes
    
    def _format_section(self, section: str) -> str:
        """Format a section for display"""
        return f"## {section.replace('_', ' ').title()}\n\n{self.sections[section]}"
    
    def get_html(self) -> str:
        """Generate an HTML version of the pitchbook"""
        html_content = """
        <html>
        <head>
            <style>
                body { 
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    line-height: 1.6;
                    color: #2c3e50;
                }
                h1 { 
                    color: #2c3e50;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }
                h2 { 
                    color: #34495e;
                    margin-top: 30px;
                    border-left: 4px solid #3498db;
                    padding-left: 10px;
                }
                .section {
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 5px;
                    margin: 20px 0;
                }
                .swot-grid {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                    margin: 20px 0;
                }
                .swot-item {
                    padding: 15px;
                    border-radius: 5px;
                    color: white;
                }
                .strengths { background-color: #27ae60; }
                .weaknesses { background-color: #c0392b; }
                .opportunities { background-color: #2980b9; }
                .threats { background-color: #d35400; }
            </style>
        </head>
        <body>
        """
        
        html_content += "<h1>Company Pitchbook</h1>"
        
        for section, content in self.sections.items():
            html_content += f'<div class="section">'
            html_content += f"<h2>{section.replace('_', ' ').title()}</h2>"
            
            if section == 'swot_analysis':
                html_content += '<div class="swot-grid">'
                html_content += f'<div class="swot-item strengths">Strengths</div>'
                html_content += f'<div class="swot-item weaknesses">Weaknesses</div>'
                html_content += f'<div class="swot-item opportunities">Opportunities</div>'
                html_content += f'<div class="swot-item threats">Threats</div>'
                html_content += '</div>'
            
            html_content += f"<div>{content}</div>"
            html_content += "</div>"
        
        html_content += "</body></html>"
        return html_content
