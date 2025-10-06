# tools.py
# Enhanced utilities with smart AnthropicClient for rich data context
# FIXED: String formatting error

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import json
import re
from functools import wraps
import time
import hashlib
import anthropic
from dataclasses import dataclass
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def sort_months_chronologically(months: List[str]) -> List[str]:
    """Sort month strings chronologically (e.g., 'January 2025', 'February 2025')"""
    try:
        month_dates = [(m, pd.to_datetime(m, format='%B %Y')) for m in months]
        sorted_months = sorted(month_dates, key=lambda x: x[1])
        return [m[0] for m in sorted_months]
    except Exception:
        # Fallback to original order if parsing fails
        return months

def get_date_range_chronological(df: pd.DataFrame, month_column: str) -> str:
    """Get chronological date range from DataFrame"""
    try:
        if month_column not in df.columns:
            return "Unknown period"
        
        df_copy = df.copy()
        df_copy['_temp_date'] = pd.to_datetime(df_copy[month_column], format='%B %Y')
        df_copy = df_copy.sort_values('_temp_date')
        
        first_month = df_copy[month_column].iloc[0]
        last_month = df_copy[month_column].iloc[-1]
        
        return f"{first_month} to {last_month}"
    except Exception:
        return "Unknown period"

def ensure_chronological_months(df: pd.DataFrame, month_column: str = 'Month') -> pd.DataFrame:
    """Ensure DataFrame months are in chronological order"""
    if month_column in df.columns:
        try:
            df = df.copy()
            df['_temp_date'] = pd.to_datetime(df[month_column], format='%B %Y')
            df = df.sort_values('_temp_date')
            df = df.drop('_temp_date', axis=1)
        except Exception:
            pass
    return df

@dataclass
class QueryContext:
    """Enhanced context information for user queries"""
    user_id: str
    session_id: str
    query_type: str
    timestamp: datetime
    previous_queries: List[str]
    preferences: Dict[str, Any]
    data_anomalies: List[str]
    discovered_patterns: List[str]

class AnthropicClient:
    """
    Enhanced client for interacting with Anthropic's Claude API
    Provides rich data context instead of generic summaries
    FIXED: String formatting error
    """
    
    def __init__(self):
        self.client = anthropic.Anthropic(
            api_key=os.getenv('ANTHROPIC_API_KEY')
        )
        self.model = os.getenv('ANTHROPIC_MODEL', 'claude-sonnet-4-20250514')
        self.max_tokens = int(os.getenv('ANTHROPIC_MAX_TOKENS', '4000'))
        self.temperature = float(os.getenv('ANTHROPIC_TEMPERATURE', '0.1'))
        self.logger = logging.getLogger(__name__)
        self.available = bool(os.getenv('ANTHROPIC_API_KEY'))
    
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None, 
                     context: Optional[Dict[str, Any]] = None) -> str:
        """Enhanced response generation with mandatory data attribution"""
        try:
            if not self.available:
                return "Claude AI is not available. Please configure your ANTHROPIC_API_KEY in the .env file."
            
            # Extract mandatory data attribution
            data_attribution = self._extract_mandatory_data_attribution(context) if context else ""
            
            # Format the prompt with mandatory attribution
            formatted_prompt = f"{data_attribution}\n\n{prompt}" if data_attribution else prompt
            
            messages = [{"role": "user", "content": formatted_prompt}]
            
            # Add constrained context if provided
            if context:
                context_str = self._format_enhanced_context(context)
                messages[0]["content"] = f"{context_str}\n\n{formatted_prompt}"
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt or self._get_enhanced_system_prompt(context),
                messages=messages
            )
            
            # Ensure response starts with data attribution
            response_text = response.content[0].text
            if not response_text.startswith("Based on"):
                response_text = f"{data_attribution}\n\n{response_text}"
            
            return response_text
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return f"Data analysis error: {str(e)}"
    
    def _format_enhanced_context(self, context: Dict[str, Any]) -> str:
        """Format constrained context with mandatory data attribution and no hallucination"""
        context_parts = []
        
        # MANDATORY DATA ATTRIBUTION (Always first)
        data_structure = context.get('raw_data_samples', {}).get('data_structure_sample', {})
        
        context_parts.append("DATA ANALYSIS SCOPE:")
        
        # FIXED: Safe number formatting
        total_records = data_structure.get('total_records', 0)
        if isinstance(total_records, (int, float)) and total_records > 0:
            context_parts.append(f"Total Records: {int(total_records):,}")
        else:
            context_parts.append(f"Total Records: {total_records}")
        
        context_parts.append(f"Date Range: {data_structure.get('date_range', 'Unknown')}")
        
        # FIXED: Safe number formatting for campaigns and terms
        campaigns = data_structure.get('unique_campaigns', 0)
        terms = data_structure.get('unique_terms', 0)
        
        if isinstance(campaigns, (int, float)):
            context_parts.append(f"Campaigns: {int(campaigns)} unique campaigns")
        else:
            context_parts.append(f"Campaigns: {campaigns} unique campaigns")
            
        if isinstance(terms, (int, float)):
            context_parts.append(f"Terms: {int(terms)} unique keywords")
        else:
            context_parts.append(f"Terms: {terms} unique keywords")
        
        context_parts.append("Data Type: Campaign and term aggregates (NOT month-by-month)")
        context_parts.append("Source: Live Google Ads and CRM data")
        context_parts.append("")
        
        # Handle filtered data table if present
        if context.get('context_type') == 'filtered_data_analysis' and context.get('data_table'):
            context_parts.append("FILTERED DATA TABLE (ACTUAL RESULTS):")
            data_table = context.get('data_table', [])
            
            for i, row in enumerate(data_table[:20]):
                row_parts = []
                for key, value in row.items():
                    if isinstance(value, (int, float)) and value != 0:
                        if isinstance(value, float):
                            row_parts.append(f"{key}: {value:.1f}")
                        else:
                            row_parts.append(f"{key}: {value}")
                    elif value and str(value).strip():
                        row_parts.append(f"{key}: {value}")
                
                if row_parts:
                    context_parts.append(f"  Row {i+1}: {', '.join(row_parts)}")
            
            if len(data_table) > 20:
                context_parts.append(f"  ... and {len(data_table) - 20} more rows")
            
            context_parts.append("")
        
        # NEW: Handle junk notes data - COMPREHENSIVE VERSION
        elif context.get('context_type') == 'junk_notes_analysis' and context.get('junk_notes_data'):
            junk_notes_data = context.get('junk_notes_data', {})
            
            if junk_notes_data.get('data_available'):
                context_parts.append("=" * 80)
                context_parts.append("JUNK LEAD NOTES ANALYSIS - ACTUAL NOTES FROM CRM")
                context_parts.append("=" * 80)
                context_parts.append(f"Total Junk Leads: {junk_notes_data.get('total_junk_leads', 0)}")
                context_parts.append(f"Leads with Notes: {junk_notes_data.get('junk_leads_with_notes', 0)}")
                context_parts.append(f"Sample Size Provided: {junk_notes_data.get('sample_size_provided', 0)}")
                context_parts.append(f"Junk Rate: {junk_notes_data.get('junk_rate_percent', 0):.1f}%")
                context_parts.append("")
                
                # Display actual notes records
                junk_records = junk_notes_data.get('junk_lead_records_with_notes', [])
                if junk_records:
                    context_parts.append("ACTUAL JUNK LEAD NOTES (Detailed Sample):")
                    context_parts.append("-" * 80)
                    
                    # Show first 50 records with notes
                    for i, record in enumerate(junk_records[:50], 1):
                        month = record.get('Month Year', 'Unknown')
                        campaign = record.get('Campaign', 'Unknown')
                        term = record.get('Term', 'Unknown')
                        notes = record.get('Notes', 'N/A')
                        reason = record.get('Reason', '')
                        status = record.get('Status', '')
                        lead_stage = record.get('Lead Stage', '')
                        score = record.get('Score', 0)
                        location = record.get('Locations', '')
                        
                        context_parts.append(f"\n{i}. {month} | {campaign}")
                        context_parts.append(f"   Keyword: '{term}'")
                        context_parts.append(f"   Status: {status} | Lead Stage: {lead_stage} | Score: {score}")
                        if location:
                            context_parts.append(f"   Location: {location}")
                        if notes and notes != 'N/A' and str(notes).strip():
                            context_parts.append(f"   Notes: {notes}")
                        if reason and str(reason).strip():
                            context_parts.append(f"   Reason: {reason}")
                    
                    if len(junk_records) > 50:
                        context_parts.append(f"\n... and {len(junk_records) - 50} more junk leads with notes available")
                    
                    context_parts.append("")
                
                # Display pattern analysis if available
                notes_patterns = junk_notes_data.get('notes_patterns', {})
                if notes_patterns and notes_patterns.get('reason_categories_found'):
                    context_parts.append("IDENTIFIED JUNK REASONS (From Pattern Analysis):")
                    context_parts.append("-" * 80)
                    
                    reason_counts = notes_patterns.get('reason_categories_found', {})
                    sorted_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)
                    
                    for reason, count in sorted_reasons:
                        clean_reason = reason.replace('_', ' ').title()
                        context_parts.append(f"  • {clean_reason}: {count} occurrences")
                    
                    context_parts.append("")
                    
                    # Show example notes for each category
                    category_examples = notes_patterns.get('example_notes_by_category', {})
                    if category_examples:
                        context_parts.append("EXAMPLE NOTES BY CATEGORY:")
                        context_parts.append("-" * 80)
                        for category in sorted_reasons[:5]:  # Top 5 categories
                            category_key = category[0]
                            if category_key in category_examples:
                                examples = category_examples[category_key]
                                clean_category = category_key.replace('_', ' ').title()
                                context_parts.append(f"\n{clean_category}:")
                                for example in examples[:2]:  # 2 examples per category
                                    context_parts.append(f"  - \"{example}\"")
                        context_parts.append("")
                
                # Display breakdowns
                junk_by_campaign = junk_notes_data.get('junk_by_campaign', {})
                if junk_by_campaign:
                    context_parts.append("JUNK LEADS BY CAMPAIGN:")
                    context_parts.append("-" * 80)
                    sorted_campaigns = sorted(junk_by_campaign.items(), key=lambda x: x[1], reverse=True)
                    for campaign, count in sorted_campaigns[:10]:
                        campaign_name = campaign if campaign else '[Unnamed Campaign]'
                        context_parts.append(f"  {campaign_name}: {count} junk leads")
                    if len(sorted_campaigns) > 10:
                        context_parts.append(f"  ... and {len(sorted_campaigns) - 10} more campaigns")
                    context_parts.append("")
                
                junk_by_term = junk_notes_data.get('junk_by_term', {})
                if junk_by_term:
                    context_parts.append("TOP JUNK KEYWORDS:")
                    context_parts.append("-" * 80)
                    sorted_terms = sorted(junk_by_term.items(), key=lambda x: x[1], reverse=True)
                    for term, count in sorted_terms[:15]:
                        context_parts.append(f"  '{term}': {count} junk leads")
                    if len(sorted_terms) > 15:
                        context_parts.append(f"  ... and {len(sorted_terms) - 15} more terms")
                    context_parts.append("")
                
                junk_by_source = junk_notes_data.get('junk_by_source', {})
                if junk_by_source:
                    context_parts.append("JUNK LEADS BY SOURCE:")
                    sorted_sources = sorted(junk_by_source.items(), key=lambda x: x[1], reverse=True)
                    for source, count in sorted_sources:
                        context_parts.append(f"  {source}: {count} junk leads")
                    context_parts.append("")
                
                junk_by_month = junk_notes_data.get('junk_by_month', {})
                if junk_by_month:
                    context_parts.append("JUNK LEADS BY MONTH:")
                    # Sort chronologically
                    try:
                        sorted_months = sorted(junk_by_month.items(), 
                                            key=lambda x: pd.to_datetime(x[0], format='%B %Y'))
                        for month, count in sorted_months:
                            context_parts.append(f"  {month}: {count} junk leads")
                    except:
                        for month, count in junk_by_month.items():
                            context_parts.append(f"  {month}: {count} junk leads")
                    context_parts.append("")
                
                context_parts.append("=" * 80)
                context_parts.append("")
            else:
                context_parts.append("ERROR: Junk notes data not available")
                error_msg = junk_notes_data.get('error', junk_notes_data.get('message', 'Unknown error'))
                context_parts.append(f"Reason: {error_msg}")
                
                # Show suggestions if available
                suggestions = junk_notes_data.get('suggested_filters', {})
                if suggestions:
                    context_parts.append("\nAvailable data to help troubleshoot:")
                    if suggestions.get('contact_types_available'):
                        context_parts.append("Contact Types in dataset:")
                        for ct, count in list(suggestions['contact_types_available'].items())[:5]:
                            context_parts.append(f"  - {ct}: {count}")
                    if suggestions.get('statuses_available'):
                        context_parts.append("Statuses in dataset:")
                        for status, count in list(suggestions['statuses_available'].items())[:5]:
                            context_parts.append(f"  - {status}: {count}")
                context_parts.append("")
        
        # Handle junk keywords data if present - COMPREHENSIVE VERSION
        elif context.get('context_type') == 'junk_keywords_analysis' and context.get('junk_keywords_data'):
            junk_data = context.get('junk_keywords_data', [])
            junk_summary = context.get('junk_summary', {})
            
            # Add header with comprehensive data info
            context_parts.append("=" * 80)
            context_parts.append("COMPLETE JUNK KEYWORDS DATA - NO SAMPLING, ALL RECORDS")
            context_parts.append("=" * 80)
            context_parts.append(f"Total Records Provided: {len(junk_data)}")
            context_parts.append(f"Total Junk Leads: {junk_summary.get('total_junk_leads', 0)}")
            context_parts.append(f"Contact Type: {junk_summary.get('contact_type_analyzed', 'Unknown')}")
            context_parts.append(f"Junk Rate: {junk_summary.get('junk_rate_percent', 0):.1f}%")
            context_parts.append("")
            
            # CRITICAL: Show ALL data, no limits
            context_parts.append("MONTH-BY-MONTH KEYWORD BREAKDOWN (ALL RECORDS):")
            for i, row in enumerate(junk_data):
                month_year = row.get('Month_Year', 'Unknown')
                campaign = row.get('Campaign', 'Unknown')
                term = row.get('Term', 'Unknown')
                junk_count = row.get('Junk_Lead_Count', 0)
                
                if junk_count > 0:
                    context_parts.append(f"{month_year} | {campaign} | '{term}' → {junk_count} junk leads")
            
            context_parts.append("")
            context_parts.append(f"✓ All {len(junk_data)} records included above")
            context_parts.append("")
            
            # Add top junk terms summary - show ALL
            top_terms = context.get('top_junk_terms', {})
            if top_terms:
                context_parts.append("TOP JUNK TERMS ACROSS ALL MONTHS (COMPLETE LIST):")
                sorted_terms = sorted(top_terms.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=True)
                
                for term, count in sorted_terms:
                    if isinstance(count, (int, float)) and count > 0:
                        context_parts.append(f"  '{term}' → {int(count)} total junk leads")
                
                context_parts.append("")
                context_parts.append(f"✓ All {len(sorted_terms)} unique terms listed above")
            
            context_parts.append("=" * 80)
            context_parts.append("")
        
        # ACTUAL CAMPAIGN PERFORMANCE DATA ONLY - SHOW ALL CAMPAIGNS
        else:
            campaign_attribution = context.get('campaign_attribution', {})
            campaign_perf = campaign_attribution.get('campaign_performance', {})
            
            if campaign_perf:
                context_parts.append("ACTUAL CAMPAIGN PERFORMANCE (Total Period):")
                context_parts.append("ALL CAMPAIGNS - Complete List:")
                
                # Sort by total leads to show systematically
                sorted_campaigns = sorted(
                    campaign_perf.items(), 
                    key=lambda x: x[1].get('Total_Leads', 0) if isinstance(x[1].get('Total_Leads', 0), (int, float)) else 0, 
                    reverse=True
                )
                
                # CRITICAL CHANGE: Show ALL campaigns, not just those with leads > 0
                for campaign, data in sorted_campaigns:
                    total_leads = data.get('Total_Leads', 0)
                    quality_rate = data.get('Quality_Rate', 0)
                    junk_rate = data.get('Junk_Rate', 0)
                    
                    # FIXED: Safe number formatting
                    if isinstance(total_leads, (int, float)):
                        leads_str = f"{int(total_leads)} leads"
                    else:
                        leads_str = f"{total_leads} leads"
                    
                    if isinstance(quality_rate, (int, float)):
                        quality_str = f"{quality_rate:.1f}% quality"
                    else:
                        quality_str = f"{quality_rate}% quality"
                    
                    if isinstance(junk_rate, (int, float)):
                        junk_str = f"{junk_rate:.1f}% junk"
                    else:
                        junk_str = f"{junk_rate}% junk"
                    
                    # Display campaign name, handle empty/unnamed campaigns
                    campaign_name = campaign if campaign else '[Unnamed Campaign]'
                    context_parts.append(f"  {campaign_name}: {leads_str}, {quality_str}, {junk_str}")
                
                context_parts.append(f"\nTotal Campaigns Listed: {len(sorted_campaigns)}")
                context_parts.append("")
            
            # ACTUAL TERM PERFORMANCE DATA ONLY
            term_perf = campaign_attribution.get('term_performance', {})
            
            if term_perf:
                context_parts.append("ACTUAL TERM PERFORMANCE (Total Period):")
                sorted_terms = sorted(
                    term_perf.items(), 
                    key=lambda x: x[1].get('Total_Leads', 0) if isinstance(x[1].get('Total_Leads', 0), (int, float)) else 0, 
                    reverse=True
                )
                
                # Show top 15 terms instead of 10 for better coverage
                for term, data in sorted_terms[:15]:
                    total_leads = data.get('Total_Leads', 0)
                    quality_rate = data.get('Quality_Rate', 0)
                    
                    if isinstance(total_leads, (int, float)) and total_leads > 0:
                        if isinstance(quality_rate, (int, float)):
                            context_parts.append(f"  '{term}': {int(total_leads)} total leads, {quality_rate:.1f}% quality")
                        else:
                            context_parts.append(f"  '{term}': {int(total_leads)} total leads, {quality_rate}% quality")
                
                context_parts.append(f"NOTE: Showing top 15 of {len(sorted_terms)} total terms")
                context_parts.append("These are aggregate totals for the full period, not monthly breakdowns")
                context_parts.append("")
        
        # ACTUAL JUNK REASONS FROM NOTES (if not already shown above)
        if context.get('context_type') not in ['junk_keywords_analysis', 'junk_notes_analysis']:
            notes_analysis = context.get('notes_analysis', {})
            junk_reasons = notes_analysis.get('junk_reasons', {})
            
            if junk_reasons:
                context_parts.append("ACTUAL JUNK REASONS (From Notes Analysis):")
                sorted_reasons = sorted(junk_reasons.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=True)
                
                for reason, count in sorted_reasons[:5]:
                    if isinstance(count, (int, float)) and count > 0:
                        clean_reason = reason.replace('_', ' ').title()
                        context_parts.append(f"  {clean_reason}: {int(count)} total occurrences")
                context_parts.append("")
        
        # ACTUAL LEAD QUALITY DISTRIBUTION
        quality_breakdown = context.get('lead_quality_breakdown', {})
        quality_dist = quality_breakdown.get('quality_distribution', {})
        
        if quality_dist:
            total_leads = quality_dist.get('total_leads', 0)
            quality_count = quality_dist.get('high_quality_count', 0)
            quality_pct = quality_dist.get('quality_percentage', 0)
            junk_count = quality_dist.get('junk_count', 0)
            
            if isinstance(total_leads, (int, float)) and total_leads > 0:
                context_parts.append("LEAD QUALITY DISTRIBUTION:")
                context_parts.append(f"Total Leads: {int(total_leads):,}")
                
                if isinstance(quality_count, (int, float)) and isinstance(quality_pct, (int, float)):
                    context_parts.append(f"High Quality (Score >70): {int(quality_count):,} ({quality_pct:.2f}%)")
                
                if isinstance(junk_count, (int, float)):
                    junk_pct = junk_count / total_leads * 100
                    context_parts.append(f"Junk Leads: {int(junk_count):,} ({junk_pct:.1f}%)")
                
                context_parts.append("")
        
        # CONTACT TYPE PERFORMANCE
        contact_type_insights = context.get('contact_type_insights', {})
        contact_analysis = contact_type_insights.get('contact_type_analysis', {})
        
        if contact_analysis:
            context_parts.append("CONTACT TYPE PERFORMANCE:")
            for contact_type, data in contact_analysis.items():
                count = data.get('count', 0)
                conversion_rate = data.get('conversion_rate', 0)
                
                if isinstance(count, (int, float)) and count > 0:
                    if isinstance(conversion_rate, (int, float)):
                        context_parts.append(f"  {contact_type}: {int(count):,} leads, {conversion_rate:.1f}% conversion")
                    else:
                        context_parts.append(f"  {contact_type}: {int(count):,} leads, {conversion_rate}% conversion")
            context_parts.append("")
        
        # DATA LIMITATIONS
        question_focus = context.get('question_focus', '')
        if question_focus:
            context_parts.append("DATA LIMITATIONS:")
            
            if 'monthly' in question_focus or 'month-on-month' in str(context.get('original_question', '')).lower():
                context_parts.append("  Month-by-month breakdown not available in current dataset")
                context_parts.append("  Only aggregate totals for full period are available")
            
            context_parts.append("  Data represents totals across the full analysis period")
            context_parts.append("  Geographic and temporal breakdowns limited to available granularity")
            context_parts.append("")
        
        # ACTUAL DATA SAMPLES
        raw_samples = context.get('raw_data_samples', {})
        
        if raw_samples.get('high_quality_samples'):
            context_parts.append("HIGH QUALITY LEAD SAMPLES:")
            for sample in raw_samples['high_quality_samples'][:2]:
                campaign = sample.get('Campaign', 'N/A')
                score = sample.get('Score', 0)
                location = sample.get('Location', 'N/A')
                
                if isinstance(score, (int, float)):
                    context_parts.append(f"  Campaign: {campaign}, Score: {score:.0f}, Location: {location}")
                else:
                    context_parts.append(f"  Campaign: {campaign}, Score: {score}, Location: {location}")
            context_parts.append("")
        
        if raw_samples.get('junk_samples'):
            context_parts.append("JUNK LEAD SAMPLES:")
            for sample in raw_samples['junk_samples'][:2]:
                campaign = sample.get('Campaign', 'N/A')
                term = sample.get('Term', 'N/A')
                notes = sample.get('Notes', 'N/A')
                context_parts.append(f"  Campaign: {campaign}, Term: {term}")
                if notes and notes != 'N/A':
                    context_parts.append(f"    Reason: {notes[:50]}...")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _get_enhanced_system_prompt(self, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate constrained system prompt that prevents hallucination"""
        data_attribution = ""
        if context and context.get('raw_data_samples', {}).get('data_structure_sample'):
            data_info = context['raw_data_samples']['data_structure_sample']
            total_records = data_info.get('total_records', 'Unknown')
            date_range = data_info.get('date_range', 'Unknown')
            campaigns = data_info.get('unique_campaigns', 0)
            
            # FIXED: Safe number formatting
            if isinstance(total_records, (int, float)) and total_records > 0:
                records_str = f"{int(total_records):,} records"
            else:
                records_str = f"{total_records} records"
            
            if isinstance(campaigns, (int, float)):
                campaigns_str = f"{int(campaigns)} campaigns"
            else:
                campaigns_str = f"{campaigns} campaigns"
            
            data_attribution = f"Based on analysis of {records_str} from {date_range} ({campaigns_str}):"
        
        base_prompt = f"""You are a professional data analyst providing factual analysis of Google Ads and CRM data.

    MANDATORY RESPONSE FORMAT:
    1. ALWAYS start your response with: "{data_attribution if data_attribution else 'Based on analysis of available data:'}"
    2. Answer ONLY the specific question asked - do not provide unsolicited analysis
    3. Use professional, business-appropriate language
    4. Maximum 1-2 emojis per response (only if user's question contains emojis)
    5. Reference only data explicitly provided in the context

    STRICT DATA CONSTRAINTS:
    - Never invent month-by-month breakdowns unless explicitly provided in context
    - Never fabricate specific numbers, dates, or detailed scenarios
    - If asked for monthly data but only totals available, state: "Monthly breakdown not available. Showing aggregate totals for [period]."
    - Only reference actual campaign names, terms, and performance metrics from the context
    - Do not create detailed month-by-month patterns or trends not in the data

    COMPREHENSIVE LISTING REQUIREMENTS:
    - When user asks for campaign optimization or "all campaigns", list EVERY campaign with metrics
    - Use a structured table or bullet list format for easy scanning
    - Include campaign name, lead count, quality rate, and junk rate for each
    - Sort by a relevant metric (leads, quality rate, or junk rate)
    - Never say "top campaigns" or sample data when comprehensive data is available

    RESPONSE GUIDELINES:
    - If asked for "list": Provide simple, factual list based on actual data
    - If asked for "frequency": Show aggregate totals with clear period attribution
    - If asked for "analysis": Limit to data provided, acknowledge limitations
    - If asked for "optimization": Show ALL campaigns to identify winners and losers
    - If specific data unavailable: "Data not available for [specific request]. Available data shows: [actual data]"

    DATA STRUCTURE UNDERSTANDING:
    - Lead Source "Web Mail" = Google Ads traffic
    - Contact Type "Junk" = Poor quality leads
    - Score 0-100 (>70 = high quality)
    - Campaign/Term data shows aggregate performance, not time series
    - Notes contain actual reasons for lead outcomes

    Your role is to provide accurate, constrained analysis based solely on the data context provided. Acknowledge data limitations rather than fill gaps with assumptions."""

        if context:
            question_focus = context.get('question_focus', '')
            
            # NEW: Add specific instructions for junk notes analysis
            if question_focus == 'junk_notes_analysis':
                base_prompt += """

    SPECIFIC INSTRUCTIONS FOR JUNK NOTES ANALYSIS:
    You have been provided with actual Notes data from junk leads in the CRM. Your task is to analyze this data systematically.

    REQUIRED ANALYSIS STRUCTURE:
    1. **Overview**: State total junk leads, junk rate, and how many have notes
    2. **Root Cause Categories**: Group the junk reasons into 3-5 main categories based on the actual notes content
    3. **Top Reasons**: Identify the top 3-5 specific reasons for junk leads with frequency counts
    4. **Pattern Analysis**: 
    - Which campaigns generate the most junk leads?
    - Which keywords/terms are associated with junk leads?
    - Are there temporal patterns (certain months worse than others)?
    5. **Actionable Recommendations**: Provide 3-5 specific, actionable steps to reduce junk leads

    CRITICAL REQUIREMENTS:
    - Base your categorization ONLY on the actual notes content provided
    - Reference specific examples from the notes to support your findings
    - Use the pattern analysis data if provided (reason_categories_found)
    - Cite actual campaign names and keywords from the data
    - Quantify findings with actual counts and percentages from the data
    - Do NOT invent junk reasons not present in the notes
    - If notes are sparse or unclear, acknowledge this limitation

    EXAMPLE CATEGORIZATION:
    If notes show: "not interested", "looking for office space", "wrong location", "budget too high"
    Then categorize as:
    - Product Mismatch (office space vs warehouse)
    - Geographic Mismatch (wrong location)
    - Budget Constraints (pricing issues)
    - Low Intent (not interested)

    Be specific and data-driven in your analysis."""
            
            elif 'monthly' in question_focus or 'month-on-month' in str(context.get('original_question', '')).lower():
                base_prompt += """

    SPECIFIC CONSTRAINT FOR THIS QUERY:
    User is asking for month-by-month data. If monthly breakdown is not available in the context, clearly state:
    "Month-by-month breakdown not available in current dataset. Showing aggregate totals for the full [period] instead."
    Do not fabricate monthly patterns or specific monthly numbers."""
        
        return base_prompt
    
    def create_focused_prompt(self, question: str, context: Dict[str, Any]) -> str:
        """Create a focused prompt based on the question and rich context"""
        question_focus = context.get('question_focus', 'general')
        
        base_prompt = f"Based on my actual Google Ads and CRM data analysis, {question}\n\n"
        
        if question_focus == 'specific_campaign_optimization':
            campaign_name = self._extract_campaign_name_from_context(context)
            base_prompt += f"SPECIFIC REQUEST: Optimize {campaign_name} campaign performance.\n"
            base_prompt += "Use the exact campaign data provided to give specific optimization recommendations.\n"
        elif question_focus == 'lead_quality_analysis':
            base_prompt += "QUALITY FOCUS: Analyze the lead quality using the specific data provided.\n"
        elif question_focus == 'budget_optimization':
            base_prompt += "BUDGET FOCUS: Optimize budget allocation using the specific utilization data provided.\n"
        
        return base_prompt
    
    def _extract_campaign_name_from_context(self, context: Dict[str, Any]) -> str:
        """Extract campaign name from context"""
        campaign_data = context.get('campaign_specific_data', {})
        if campaign_data and hasattr(campaign_data, 'keys'):
            return list(campaign_data.keys())[0] if campaign_data else 'Unknown Campaign'
        return 'Unknown Campaign'
    
    def _extract_mandatory_data_attribution(self, context: Dict[str, Any]) -> str:
        """Extract mandatory data attribution for every response"""
        try:
            data_structure = context.get('raw_data_samples', {}).get('data_structure_sample', {})
            
            if data_structure:
                total_records = data_structure.get('total_records', 0)
                date_range = data_structure.get('date_range', 'Unknown period')
                campaigns = data_structure.get('unique_campaigns', 0)
                
                # FIXED: Safe number formatting
                if isinstance(total_records, (int, float)) and total_records > 0:
                    records_str = f"{int(total_records):,} records"
                else:
                    records_str = f"{total_records} records"
                
                if isinstance(campaigns, (int, float)):
                    campaigns_str = f"{int(campaigns)} campaigns"
                else:
                    campaigns_str = f"{campaigns} campaigns"
                
                return f"Based on analysis of {records_str} from {date_range} ({campaigns_str}):"
            else:
                return "Based on available data analysis:"
                
        except Exception as e:
            return "Based on data analysis:"

class DataValidator:
    """Enhanced data validation with anomaly detection"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_campaign_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced campaign data validation with anomaly detection"""
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'quality_score': 100.0,
            'column_analysis': {},
            'anomalies_detected': []
        }
        
        critical_columns = ['Month', 'Campaign ID', 'Daily Budget', 'Cost', 'Impressions', 'Clicks', 'Conversions']
        optional_columns = ['CTR %', 'Conversion Rate', 'Mobile Clicks', 'Desktop Clicks', 'Tablet Clicks']
        
        missing_critical = [col for col in critical_columns if col not in df.columns]
        if missing_critical:
            validation['errors'].append(f"Missing critical columns: {missing_critical}")
            validation['is_valid'] = False
            validation['quality_score'] -= 40
        
        missing_optional = [col for col in optional_columns if col not in df.columns]
        if missing_optional:
            validation['warnings'].append(f"Missing optional columns: {missing_optional}")
            validation['quality_score'] -= 10
        
        if 'Daily Budget' in df.columns and 'Cost' in df.columns:
            utilization = df['Cost'] / df['Daily Budget'].replace(0, np.nan)
            
            extreme_under = (utilization < 0.5).sum()
            extreme_over = (utilization > 1.5).sum()
            
            if extreme_under > 0:
                validation['anomalies_detected'].append(f"CRITICAL: {extreme_under} records with <50% budget utilization")
            if extreme_over > 0:
                validation['anomalies_detected'].append(f"WARNING: {extreme_over} records with >150% budget utilization")
        
        if all(col in df.columns for col in ['Clicks', 'Conversions']):
            impossible_conversions = df['Conversions'] > df['Clicks']
            if impossible_conversions.any():
                validation['errors'].append(f"{impossible_conversions.sum()} records have more conversions than clicks")
                validation['quality_score'] -= 20
        
        return validation
    
    def validate_crm_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced CRM data validation with quality analysis"""
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'quality_score': 100.0,
            'column_analysis': {},
            'quality_insights': []
        }
        
        critical_columns = ['Month Year', 'ID', 'Email', 'Lead Source', 'Lead Stage']
        optional_columns = ['Score', 'Business Type', 'Location', 'Created Time', 'Notes', 'Contact Type']
        
        missing_critical = [col for col in critical_columns if col not in df.columns]
        if missing_critical:
            validation['errors'].append(f"Missing critical CRM columns: {missing_critical}")
            validation['is_valid'] = False
            validation['quality_score'] -= 40
        
        if 'Contact Type' in df.columns:
            junk_rate = (df['Contact Type'] == 'Junk').mean() * 100
            validation['quality_insights'].append(f"Junk rate: {junk_rate:.1f}%")
            
            if junk_rate > 50:
                validation['warnings'].append(f"HIGH JUNK RATE: {junk_rate:.1f}% of leads are junk")
        
        if 'Score' in df.columns:
            high_quality = (df['Score'] > 70).mean() * 100
            validation['quality_insights'].append(f"High quality rate: {high_quality:.1f}%")
            
            if high_quality < 5:
                validation['warnings'].append(f"LOW QUALITY: Only {high_quality:.1f}% of leads are high quality")
        
        if 'Notes' in df.columns:
            notes_coverage = df['Notes'].notna().mean() * 100
            validation['quality_insights'].append(f"Notes coverage: {notes_coverage:.1f}%")
        
        return validation

class MetricsCalculator:
    """Enhanced metrics calculator with advanced CRM analysis"""
    
    @staticmethod
    def calculate_campaign_metrics(df: pd.DataFrame) -> Dict[str, float]:
        """Calculate standard campaign metrics using exact column names"""
        metrics = {}
        
        if all(col in df.columns for col in ['Impressions', 'Clicks']):
            total_impressions = df['Impressions'].sum()
            total_clicks = df['Clicks'].sum()
            metrics['ctr'] = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
        
        if all(col in df.columns for col in ['Cost', 'Clicks']):
            total_cost = df['Cost'].sum()
            total_clicks = df['Clicks'].sum()
            metrics['cpc'] = (total_cost / total_clicks) if total_clicks > 0 else 0
        
        if all(col in df.columns for col in ['Conversions', 'Clicks']):
            total_conversions = df['Conversions'].sum()
            total_clicks = df['Clicks'].sum()
            metrics['conversion_rate'] = (total_conversions / total_clicks * 100) if total_clicks > 0 else 0
        
        if all(col in df.columns for col in ['Cost', 'Conversions']):
            total_cost = df['Cost'].sum()
            total_conversions = df['Conversions'].sum()
            metrics['cost_per_conversion'] = (total_cost / total_conversions) if total_conversions > 0 else 0
        
        if all(col in df.columns for col in ['Cost', 'Daily Budget']):
            total_cost = df['Cost'].sum()
            total_daily_budget = df['Daily Budget'].sum()
            metrics['budget_utilization'] = (total_cost / total_daily_budget * 100) if total_daily_budget > 0 else 0
        
        return metrics
    
    @staticmethod
    def calculate_lead_quality_metrics(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate enhanced lead quality metrics"""
        metrics = {}
        
        if 'ID' in df.columns:
            metrics['total_leads'] = len(df)
        
        if 'Score' in df.columns:
            scores = df['Score'].dropna()
            if len(scores) > 0:
                metrics['average_score'] = float(scores.mean())
                metrics['score_distribution'] = {
                    'high_quality': len(scores[scores > 70]),
                    'medium_quality': len(scores[(scores >= 40) & (scores <= 70)]),
                    'low_quality': len(scores[scores < 40])
                }
                metrics['quality_percentage'] = len(scores[scores > 70]) / len(scores) * 100
        
        if 'Contact Type' in df.columns:
            contact_dist = df['Contact Type'].value_counts()
            metrics['contact_type_distribution'] = contact_dist.to_dict()
            
            junk_count = contact_dist.get('Junk', 0)
            metrics['junk_percentage'] = junk_count / len(df) * 100
        
        if 'Lead Source' in df.columns:
            source_dist = df['Lead Source'].value_counts()
            metrics['top_sources'] = source_dist.head(5).to_dict()
            
            google_ads_leads = len(df[df['Lead Source'] == 'Web Mail'])
            metrics['google_ads_leads'] = google_ads_leads
            metrics['google_ads_percentage'] = google_ads_leads / len(df) * 100
        
        return metrics
    
    @staticmethod
    def calculate_device_metrics(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate device performance metrics using exact column names"""
        device_metrics = {}
        devices = ['Mobile', 'Desktop', 'Tablet']
        
        total_clicks = 0
        total_cost = 0
        total_conversions = 0
        
        for device in devices:
            clicks_col = f'{device} Clicks'
            cost_col = f'{device} Cost'
            conv_col = f'{device} Conversions'
            
            if all(col in df.columns for col in [clicks_col, cost_col, conv_col]):
                total_clicks += df[clicks_col].sum()
                total_cost += df[cost_col].sum()
                total_conversions += df[conv_col].sum()
        
        for device in devices:
            clicks_col = f'{device} Clicks'
            impressions_col = f'{device} Impressions'
            cost_col = f'{device} Cost'
            conv_col = f'{device} Conversions'
            
            if all(col in df.columns for col in [clicks_col, cost_col, conv_col]):
                device_clicks = df[clicks_col].sum()
                device_cost = df[cost_col].sum()
                device_conversions = df[conv_col].sum()
                device_impressions = df[impressions_col].sum() if impressions_col in df.columns else 0
                
                device_metrics[device.lower()] = {
                    'click_share': (device_clicks / total_clicks * 100) if total_clicks > 0 else 0,
                    'cost_share': (device_cost / total_cost * 100) if total_cost > 0 else 0,
                    'conversion_share': (device_conversions / total_conversions * 100) if total_conversions > 0 else 0,
                    'ctr': (device_clicks / device_impressions * 100) if device_impressions > 0 else 0,
                    'cpc': (device_cost / device_clicks) if device_clicks > 0 else 0,
                    'conversion_rate': (device_conversions / device_clicks * 100) if device_clicks > 0 else 0,
                    'cost_per_conversion': (device_cost / device_conversions) if device_conversions > 0 else 0
                }
        
        return device_metrics

class VisualizationHelper:
    """Helper class for creating visualizations"""
    
    def __init__(self):
        self.color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        self.google_ads_colors = {
            'primary': '#4285f4',
            'secondary': '#34a853',
            'accent': '#ea4335',
            'warning': '#fbbc04'
        }
    
    def create_monthly_performance_chart(self, monthly_data: pd.DataFrame) -> go.Figure:
        """Create monthly performance chart with chronological month sorting"""
        try:
            # CRITICAL FIX: Sort months chronologically before visualization
            if 'Month' in monthly_data.columns:
                unique_months = monthly_data['Month'].unique().tolist()
                sorted_months = sort_months_chronologically(unique_months)
                
                # Create categorical with correct order
                monthly_data = monthly_data.copy()
                monthly_data['Month'] = pd.Categorical(
                    monthly_data['Month'], 
                    categories=sorted_months, 
                    ordered=True
                )
                monthly_data = monthly_data.sort_values('Month')
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Monthly Spend vs Daily Budget', 'Conversions', 'Budget Utilization %', 'Conversion Rate %'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            months = monthly_data['Month'].tolist()
            
            # Monthly spend vs daily budget
            if all(col in monthly_data.columns for col in ['Cost', 'Daily Budget']):
                fig.add_trace(
                    go.Bar(x=months, y=monthly_data['Cost'],
                          name='Actual Spend', marker_color=self.google_ads_colors['primary']),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=months, y=monthly_data['Daily Budget'],
                              mode='lines+markers', name='Daily Budget',
                              line=dict(color=self.google_ads_colors['accent'], dash='dash')),
                    row=1, col=1
                )
            
            # Conversions
            if 'Conversions' in monthly_data.columns:
                fig.add_trace(
                    go.Scatter(x=months, y=monthly_data['Conversions'],
                              mode='lines+markers', name='Conversions',
                              marker=dict(size=8, color=self.google_ads_colors['secondary'])),
                    row=1, col=2
                )
            
            # Budget utilization
            if 'Budget Utilization' in monthly_data.columns:
                utilization = monthly_data['Budget Utilization'] * 100
                colors = ['green' if x <= 100 else 'red' for x in utilization]
                fig.add_trace(
                    go.Bar(x=months, y=utilization,
                          name='Utilization %', marker_color=colors),
                    row=2, col=1
                )
                fig.add_hline(y=100, line_dash="dash", line_color="black", row=2, col=1)
            
            # Conversion rate
            if 'Conversion Rate' in monthly_data.columns:
                conv_rate = monthly_data['Conversion Rate'] * 100
                fig.add_trace(
                    go.Scatter(x=months, y=conv_rate,
                              mode='lines+markers', name='Conv Rate %',
                              marker=dict(size=8, color=self.google_ads_colors['warning'])),
                    row=2, col=2
                )
            
            fig.update_layout(
                title="Campaign Performance Overview - Chronological Analysis",
                showlegend=True,
                height=600,
                template="plotly_white"
            )
            
            return fig
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error creating monthly performance chart: {str(e)}")
            return go.Figure().add_annotation(text=f"Chart error: {str(e)}", 
                                            xref="paper", yref="paper", x=0.5, y=0.5)
    
    def create_lead_quality_dashboard(self, crm_data: pd.DataFrame) -> go.Figure:
        """Create comprehensive lead quality dashboard"""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Lead Quality Distribution', 'Contact Type Performance', 
                               'Junk Rate by Source', 'Quality Score Distribution'),
                specs=[[{"type": "pie"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "histogram"}]]
            )
            
            if 'Contact Type' in crm_data.columns:
                contact_counts = crm_data['Contact Type'].value_counts()
                fig.add_trace(
                    go.Pie(labels=contact_counts.index, values=contact_counts.values,
                           name="Contact Types"),
                    row=1, col=1
                )
            
            if all(col in crm_data.columns for col in ['Contact Type', 'Status']):
                contact_conversion = crm_data.groupby('Contact Type').apply(
                    lambda x: (x['Status'].isin(['Hot', 'Converted']).sum() / len(x)) * 100
                ).sort_values(ascending=False)
                
                fig.add_trace(
                    go.Bar(x=contact_conversion.index, y=contact_conversion.values,
                           name='Conversion Rate %', marker_color=self.google_ads_colors['secondary']),
                    row=1, col=2
                )
            
            if all(col in crm_data.columns for col in ['Lead Source', 'Contact Type']):
                source_junk = crm_data.groupby('Lead Source').apply(
                    lambda x: (x['Contact Type'] == 'Junk').mean() * 100
                ).sort_values(ascending=False)
                
                fig.add_trace(
                    go.Bar(x=source_junk.index, y=source_junk.values,
                           name='Junk Rate %', marker_color=self.google_ads_colors['accent']),
                    row=2, col=1
                )
            
            if 'Score' in crm_data.columns:
                scores = crm_data['Score'].dropna()
                fig.add_trace(
                    go.Histogram(x=scores, nbinsx=20, name='Score Distribution',
                                marker_color=self.google_ads_colors['primary']),
                    row=2, col=2
                )
            
            fig.update_layout(
                title="Lead Quality Analysis Dashboard",
                height=600,
                template="plotly_white"
            )
            
            return fig
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error creating lead quality dashboard: {str(e)}")
            return go.Figure().add_annotation(text=f"Dashboard error: {str(e)}", 
                                            xref="paper", yref="paper", x=0.5, y=0.5)

class CacheManager:
    """Manages caching with enhanced invalidation"""
    
    def __init__(self, cache_dir: str = 'data/cache'):
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(__name__)
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_key(self, data: Any) -> str:
        """Generate cache key for data"""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def cache_analysis_result(self, key: str, result: Dict[str, Any], 
                            ttl_minutes: int = 15) -> None:
        """Cache analysis result with TTL"""
        cache_data = {
            'result': result,
            'timestamp': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(minutes=ttl_minutes)).isoformat()
        }
        
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        try:
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Error caching result: {str(e)}")
    
    def get_cached_result(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached result if still valid"""
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                expires_at = datetime.fromisoformat(cache_data['expires_at'])
                if datetime.now() < expires_at:
                    return cache_data['result']
                else:
                    os.remove(cache_file)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving cached result: {str(e)}")
            return None
    
    def clear_expired_cache(self) -> None:
        """Clear all expired cache files"""
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.cache_dir, filename)
                    try:
                        with open(filepath, 'r') as f:
                            cache_data = json.load(f)
                        
                        expires_at = datetime.fromisoformat(cache_data['expires_at'])
                        if datetime.now() >= expires_at:
                            os.remove(filepath)
                    except:
                        os.remove(filepath)
                        
        except Exception as e:
            self.logger.error(f"Error clearing expired cache: {str(e)}")

# Enhanced utility functions
def timing_decorator(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger = logging.getLogger(func.__module__)
        logger.info(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        
        return result
    return wrapper

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers with default for zero division"""
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ZeroDivisionError):
        return default

def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change between two values"""
    if old_value == 0:
        return 0.0 if new_value == 0 else 100.0
    return ((new_value - old_value) / old_value) * 100

def format_currency(amount: float) -> str:
    """Format amount as currency"""
    return f"${amount:,.2f}"

def format_percentage(value: float) -> str:
    """Format value as percentage"""
    return f"{value:.2f}%"

def format_number(value: Union[int, float]) -> str:
    """Format number with appropriate commas"""
    if isinstance(value, float):
        return f"{value:,.2f}"
    else:
        return f"{value:,}"

def calculate_budget_utilization(cost: float, daily_budget: float) -> float:
    """Calculate budget utilization percentage using Daily Budget"""
    return safe_divide(cost, daily_budget, 0.0) * 100

def get_utilization_status(utilization_pct: float) -> str:
    """Get utilization status based on percentage"""
    if utilization_pct < 70:
        return "Significantly Underutilized"
    elif utilization_pct < 90:
        return "Underutilized" 
    elif utilization_pct <= 100:
        return "Well Utilized"
    elif utilization_pct <= 120:
        return "Over Utilized"
    else:
        return "Significantly Over Budget"

def get_performance_grade(conversion_rate: float) -> str:
    """Get performance grade based on conversion rate"""
    if conversion_rate >= 5.0:
        return "Excellent"
    elif conversion_rate >= 3.0:
        return "Good"
    elif conversion_rate >= 1.0:
        return "Fair"
    else:
        return "Needs Improvement"

def detect_lead_quality_crisis(quality_percentage: float) -> str:
    """Detect lead quality crisis level - FIXED for actual score distribution"""
    if quality_percentage < 0.5:  # Less than 0.5% have scores >0
        return "CRITICAL CRISIS"
    elif quality_percentage < 1:  # Less than 1% have scores >0  
        return "SEVERE ISSUE"
    elif quality_percentage < 5:  # Less than 5% have scores >0
        return "NEEDS IMPROVEMENT"
    else:
        return "ACCEPTABLE"

def extract_campaign_name_from_question(question: str) -> Optional[str]:
    """Extract campaign name from user question"""
    question_lower = question.lower()
    
    if 'lead-maximiseconv' in question_lower or 'lead maximise' in question_lower:
        return 'Lead-MaximiseConv'
    elif 'maximise conv' in question_lower:
        return 'Lead-MaximiseConv'
    
    campaign_pattern = r'campaign[:\s]+([a-zA-Z0-9\-_]+)'
    match = re.search(campaign_pattern, question, re.IGNORECASE)
    if match:
        return match.group(1)
    
    return None

def sanitize_input(text: str) -> str:
    """Sanitize user input to prevent injection attacks"""
    sanitized = re.sub(r'[<>"\';()&]', '', text)
    return sanitized.strip()

def validate_month_format(month_str: str) -> bool:
    """Validate month format (e.g., 'January 2025')"""
    pattern = r'^[A-Za-z]+ \d{4}'
    return bool(re.match(pattern, month_str))

class ConfigManager:
    """Enhanced configuration manager"""
    
    def __init__(self, config_file: str = 'config.json'):
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or environment variables"""
        config = {}
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
            except Exception:
                pass
        
        env_mappings = {
            'ANTHROPIC_API_KEY': ('anthropic', 'api_key'),
            'ANTHROPIC_MODEL': ('anthropic', 'model'),
            'ANTHROPIC_MAX_TOKENS': ('anthropic', 'max_tokens'),
            'ANTHROPIC_TEMPERATURE': ('anthropic', 'temperature'),
            'GOOGLE_ADS_SHEET_ID': ('google_sheets', 'ads_sheet_id'),
            'CRM_SHEET_ID': ('google_sheets', 'crm_sheet_id'),
            'ADS_WORKSHEET_NAME': ('google_sheets', 'ads_worksheet'),
            'CRM_WORKSHEET_NAME': ('google_sheets', 'crm_worksheet')
        }
        
        for env_var, (section, key) in env_mappings.items():
            if env_var in os.environ:
                if section not in config:
                    config[section] = {}
                
                value = os.environ[env_var]
                if key in ['max_tokens']:
                    value = int(value)
                elif key in ['temperature']:
                    value = float(value)
                
                config[section][key] = value
        
        return config
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(section, {}).get(key, default)

class Logger:
    """Enhanced logging setup"""
    
    @staticmethod
    def setup_logging(log_level: str = 'INFO', log_file: str = 'data/app.log') -> None:
        """Setup application logging"""
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

class BusinessContext:
    """Enhanced business context with dynamic updates"""
    
    def __init__(self, context_file: str = 'data/business_context.md'):
        self.context_file = context_file
        self.context = self._load_context()
    
    def _load_context(self) -> str:
        """Load business context from markdown file"""
        try:
            if os.path.exists(self.context_file):
                with open(self.context_file, 'r', encoding='utf-8') as f:
                    return f.read()
            return self._get_default_context()
        except Exception:
            return self._get_default_context()
    
    def _get_default_context(self) -> str:
        """Get enhanced default business context"""
        return """# Google Ads Campaign & CRM Analysis Context

## Data Structure
- **Campaign Data**: Monthly performance with Daily Budget utilization tracking
- **CRM Data**: Lead records with rich attribution data
- **Key Correlation**: Lead Source "Web Mail" = Google Ads traffic

## Critical Columns
- **Campaign**: Google Ads campaign names
- **Term**: Search keywords triggering ads
- **Contact Type**: Lead quality indicator (3PL, Direct, Broker, Junk)
- **Notes**: Unstructured feedback on lead outcomes
- **Score**: Lead quality score (0-100, >70 = high quality)

## Business Metrics
- **Target Quality Rate**: >5% high-quality leads
- **Budget Utilization**: 80-100% optimal range
- **Junk Rate**: <30% acceptable
"""
    
    def get_context_for_query(self, query_type: str) -> str:
        """Get relevant context based on query type"""
        return self.context
    
    def update_context(self, new_context: str) -> None:
        """Update business context"""
        try:
            os.makedirs(os.path.dirname(self.context_file), exist_ok=True)
            with open(self.context_file, 'w', encoding='utf-8') as f:
                f.write(new_context)
            self.context = new_context
        except Exception as e:
            logging.getLogger(__name__).error(f"Error updating context: {str(e)}")

class SmartContextBuilder:
    """Smart context builder that creates rich context for Claude"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def build_context_for_question(self, question: str, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build smart context based on specific question and analysis data"""
        try:
            question_focus = self._classify_question(question)
            
            context = {
                'question_focus': question_focus,
                'original_question': question,  # ADD THIS
                'data_anomalies': self._extract_relevant_anomalies(question, analysis_data),
                'discovered_patterns': self._extract_relevant_patterns(question, analysis_data),
                'raw_data_samples': analysis_data.get('crm_analysis', {}).get('data_samples', {}),
            }
            
            # NEW: Handle junk notes queries
            if question_focus == 'junk_notes_analysis':
                context.update(self._build_junk_notes_context(question, analysis_data))
            elif question_focus == 'specific_campaign_optimization':
                context.update(self._build_campaign_context(question, analysis_data))
            elif question_focus == 'lead_quality_analysis':
                context.update(self._build_quality_context(question, analysis_data))
            elif question_focus == 'budget_optimization':
                context.update(self._build_budget_context(question, analysis_data))
            elif question_focus == 'optimization_strategy':
                context.update(self._build_optimization_context(question, analysis_data))
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error building context: {str(e)}")
            return {'error': str(e)}
    
    def _classify_question(self, question: str) -> str:
        """Classify question type for targeted context"""
        question_lower = question.lower()
        
        # NEW: Detect notes/junk analysis queries
        if any(term in question_lower for term in ['notes', 'why junk', 'junk reason', 'rejected', 'lost leads']):
            return 'junk_notes_analysis'
        
        if extract_campaign_name_from_question(question):
            return 'specific_campaign_optimization'
        elif any(term in question_lower for term in ['quality', 'junk', 'score']):
            return 'lead_quality_analysis'
        elif any(term in question_lower for term in ['budget', 'utilization', 'spend']):
            return 'budget_optimization'
        elif any(term in question_lower for term in ['optimize', 'improve', 'fix']):
            return 'optimization_strategy'
        else:
            return 'general_analysis'
    
    def _extract_relevant_anomalies(self, question: str, analysis_data: Dict) -> List[str]:
        """Extract anomalies relevant to the specific question"""
        all_anomalies = []
        
        campaign_anomalies = analysis_data.get('campaign_analysis', {}).get('data_anomalies', [])
        all_anomalies.extend(campaign_anomalies)
        
        crm_analysis = analysis_data.get('crm_analysis', {})
        quality_data = crm_analysis.get('lead_quality_patterns', {}).get('quality_distribution', {})
        
        if quality_data:
            quality_pct = quality_data.get('quality_percentage', 0)
            if quality_pct < 1:
                all_anomalies.append(f"CRITICAL: Lead quality crisis - only {quality_pct:.2f}% high quality leads")
            
            junk_count = quality_data.get('junk_count', 0)
            total_leads = quality_data.get('total_leads', 1)
            junk_pct = junk_count / total_leads * 100
            
            if junk_pct > 60:
                all_anomalies.append(f"CRITICAL: {junk_pct:.1f}% junk rate - lead generation failing")
        
        return all_anomalies
    
    def _extract_relevant_patterns(self, question: str, analysis_data: Dict) -> List[str]:
        """Extract patterns relevant to the specific question"""
        all_patterns = []
        
        campaign_patterns = analysis_data.get('campaign_analysis', {}).get('performance_patterns', [])
        all_patterns.extend(campaign_patterns)
        
        crm_analysis = analysis_data.get('crm_analysis', {})
        quality_indicators = crm_analysis.get('quality_indicators', [])
        all_patterns.extend(quality_indicators)
        
        campaign_insights = crm_analysis.get('campaign_term_insights', {}).get('performance_insights', [])
        all_patterns.extend(campaign_insights)
        
        return all_patterns
    
    def _build_campaign_context(self, question: str, analysis_data: Dict) -> Dict[str, Any]:
        """Build context for campaign-specific questions"""
        campaign_name = extract_campaign_name_from_question(question)
        
        return {
            'campaign_attribution': analysis_data.get('crm_analysis', {}).get('google_ads_attribution', {}),
            'campaign_specific_data': analysis_data.get('crm_analysis', {}).get('campaign_specific_insights', {}).get(campaign_name, {}),
            'notes_analysis': analysis_data.get('crm_analysis', {}).get('notes_text_analysis', {}),
            'conversion_barriers': analysis_data.get('crm_analysis', {}).get('conversion_barriers', {})
        }
    
    def _build_quality_context(self, question: str, analysis_data: Dict) -> Dict[str, Any]:
        """Build context for lead quality questions"""
        return {
            'lead_quality_breakdown': analysis_data.get('crm_analysis', {}).get('lead_quality_patterns', {}),
            'notes_analysis': analysis_data.get('crm_analysis', {}).get('notes_text_analysis', {}),
            'contact_type_insights': analysis_data.get('crm_analysis', {}).get('contact_type_performance', {}),
            'conversion_barriers': analysis_data.get('crm_analysis', {}).get('conversion_barriers', {})
        }
    
    def _build_budget_context(self, question: str, analysis_data: Dict) -> Dict[str, Any]:
        """Build context for budget optimization questions"""
        return {
            'budget_efficiency': analysis_data.get('campaign_analysis', {}).get('budget_efficiency', {}),
            'overall_metrics': analysis_data.get('campaign_analysis', {}).get('overall_metrics', {}),
            'monthly_trends': analysis_data.get('campaign_analysis', {}).get('monthly_trends', {})
        }
    
    def _build_optimization_context(self, question: str, analysis_data: Dict) -> Dict[str, Any]:
        """Build context for optimization strategy questions"""
        return {
            'lead_quality_breakdown': analysis_data.get('crm_analysis', {}).get('lead_quality_patterns', {}),
            'campaign_attribution': analysis_data.get('crm_analysis', {}).get('google_ads_attribution', {}),
            'conversion_barriers': analysis_data.get('crm_analysis', {}).get('conversion_barriers', {}),
            'budget_efficiency': analysis_data.get('campaign_analysis', {}).get('budget_efficiency', {}),
            'notes_analysis': analysis_data.get('crm_analysis', {}).get('notes_text_analysis', {}),
            'contact_type_insights': analysis_data.get('crm_analysis', {}).get('contact_type_performance', {})
        }
    
    def _build_junk_notes_context(self, question: str, analysis_data: Dict) -> Dict[str, Any]:
        """Build context for junk notes analysis questions"""
        return {
            'context_type': 'junk_notes_analysis',
            'junk_notes_data': analysis_data.get('crm_analysis', {}).get('junk_notes_analysis', {}),
            'notes_patterns': analysis_data.get('crm_analysis', {}).get('notes_text_analysis', {}),
            'lead_quality_breakdown': analysis_data.get('crm_analysis', {}).get('lead_quality_patterns', {}),
            'campaign_attribution': analysis_data.get('crm_analysis', {}).get('google_ads_attribution', {}),
        }