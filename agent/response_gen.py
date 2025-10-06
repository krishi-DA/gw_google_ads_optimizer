# response_gen.py
# Enhanced response generator that provides rich data context to Claude
# FIXED: Ensures filtered data reaches Claude in readable format

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from analyzer import CampaignCRMAnalyzer
import pandas as pd
import re

def sort_months_chronologically(months: List[str]) -> List[str]:
    """Sort month strings chronologically"""
    try:
        month_dates = [(m, pd.to_datetime(m, format='%B %Y')) for m in months]
        sorted_months = sorted(month_dates, key=lambda x: x[1])
        return [m[0] for m in sorted_months]
    except Exception:
        return months

def get_date_range_chronological(df: pd.DataFrame, month_column: str) -> str:
    """Get chronological date range from DataFrame"""
    try:
        df_copy = df.copy()
        df_copy['_temp_date'] = pd.to_datetime(df_copy[month_column], format='%B %Y')
        df_copy = df_copy.sort_values('_temp_date')
        return f"{df_copy[month_column].iloc[0]} to {df_copy[month_column].iloc[-1]}"
    except Exception:
        return "Unknown period"

class ResponseGenerator:
    """
    Enhanced response generator that provides Claude with clean, accessible data
    FIXED: Proper data flow for filtered queries and direct Claude consumption
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.analyzer = CampaignCRMAnalyzer()
    
    # NEW: Primary method for Claude data access
    def generate_claude_context(self, question: str) -> Dict[str, Any]:
        """
        MAIN METHOD: Generate context optimized for Claude consumption
        Routes to filtered data or enhanced context based on question type
        """
        try:
            question_lower = question.lower()
            
            # Determine if question needs filtered data analysis
            needs_filtered_data = self._requires_filtered_data_analysis(question_lower)
            
            if needs_filtered_data:
                self.logger.info(f"Routing to filtered data analysis for: {question[:50]}...")
                return self._generate_filtered_data_context(question)
            else:
                self.logger.info(f"Routing to enhanced context analysis for: {question[:50]}...")
                return self._generate_enhanced_analysis_context(question)
                
        except Exception as e:
            self.logger.error(f"Error generating Claude context: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._generate_error_context(str(e))
    
    def _requires_filtered_data_analysis(self, question_lower: str) -> bool:
        """Determine if question requires filtered data analysis"""
        
        # Direct filtered data indicators
        filtered_data_patterns = [
            # Contact type + junk analysis
            ('contact type' in question_lower and any(ct in question_lower for ct in ['3pl', 'junk', 'direct', 'broker'])),
            
            # Keyword/term analysis with filtering
            ('junk' in question_lower and any(kw in question_lower for kw in ['keyword', 'term', 'creating'])),
            
            # Campaign-wise analysis
            ('campaign-wise' in question_lower or 'campaign wise' in question_lower),
            
            # Month-by-month analysis
            ('month-by-month' in question_lower or 'month-on-month' in question_lower),
            
            # List queries for specific breakdowns
            ('list' in question_lower and any(item in question_lower for item in ['keyword', 'term', 'campaign'])),
            
            # Specific combination queries
            ('creating junk leads' in question_lower),
            ('junk leads for' in question_lower)
        ]
        
        return any(filtered_data_patterns)
    
    def _generate_filtered_data_context(self, question: str) -> Dict[str, Any]:
        """
        Generate context for questions requiring filtered data analysis
        Returns actual data tables that Claude can read directly
        """
        try:
            # Extract filters from question
            filters = self._extract_filters_from_question(question)
            self.logger.info(f"Extracted filters: {filters}")
            
            # Check if this is a junk keywords query
            if 'junk' in question.lower() and any(kw in question.lower() for kw in ['keyword', 'term', 'creating']):
                return self._generate_junk_keywords_context(question, filters)
            
            # General filtered analysis
            filtered_result = self.analyzer.get_filtered_lead_analysis(
                contact_type=filters.get('contact_type'),
                lead_stage=filters.get('lead_stage'),
                status=filters.get('status'),
                campaign=filters.get('campaign'),
                include_junk_only=filters.get('include_junk_only', False),
                group_by=filters.get('group_by', ['Campaign', 'Term', 'Month Year'])
            )
            
            # Check if we got data back
            if not filtered_result.get('data_available', False):
                return {
                    'context_type': 'filtered_data_error',
                    'question': question,
                    'error_message': filtered_result.get('error', 'No data found matching the specified criteria'),
                    'filters_applied': filters,
                    'suggestion': 'Try broadening your search criteria or check if data exists for the specified filters',
                    'available_data_info': self._get_available_data_summary()
                }
            
            # Return structured context with actual data
            return {
                'context_type': 'filtered_data_analysis',
                'question': question,
                'data_available': True,
                'data_table': filtered_result.get('data_table', []),  # Actual data rows
                'data_summary': {
                    'total_rows': filtered_result.get('grouped_result_count', 0),
                    'original_lead_count': filtered_result.get('original_count', 0),
                    'filtered_lead_count': filtered_result.get('filtered_count', 0),
                    'unique_campaigns': len(set([row.get('Campaign', '') for row in filtered_result.get('data_table', [])])),
                    'unique_terms': len(set([row.get('Term', '') for row in filtered_result.get('data_table', [])])),
                    'date_range': self._get_date_range_from_data(filtered_result.get('data_table', []))
                },
                'filters_applied': filters,
                'analysis_instructions': (
                    "CRITICAL: The 'data_table' contains actual filtered data rows. "
                    "Each row represents aggregated leads for that Campaign + Term + Month combination. "
                    "Analyze this data directly to answer the user's question. "
                    "Do NOT say data is unavailable - the data is right here in the data_table array."
                ),
                'summary_stats': filtered_result.get('summary_stats', {}),
                'data_attribution': f"Filtered from {filtered_result.get('original_count', 0)} total CRM records"
            }
            
        except Exception as e:
            self.logger.error(f"Error generating filtered data context: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._generate_error_context(str(e))
    
    def _generate_junk_keywords_context(self, question: str, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate context specifically for junk keywords queries
        Uses the direct junk analysis method from analyzer
        """
        try:
            contact_type = filters.get('contact_type', '3pl')  # Default to 3pl if not specified
            
            # Use the direct junk keywords method
            junk_result = self.analyzer.query_junk_keywords_by_contact_type(contact_type)
            
            if not junk_result.get('data_available', False):
                return {
                    'context_type': 'junk_keywords_error',
                    'question': question,
                    'error_message': junk_result.get('error', f'No junk leads found for contact type "{contact_type}"'),
                    'contact_type_searched': contact_type,
                    'suggestion': junk_result.get('message', 'Try a different contact type or check data availability'),
                    'available_contact_types': junk_result.get('available_contact_types', {}),
                    'sample_data': junk_result.get('sample_leads', [])
                }
            
            # Return structured junk keywords context
            return {
                'context_type': 'junk_keywords_analysis',
                'question': question,
                'data_available': True,
                'junk_keywords_data': junk_result.get('month_campaign_term_breakdown', []),  # Actual junk data
                'junk_summary': {
                    'total_junk_leads': junk_result.get('total_junk_leads', 0),
                    'total_leads_for_contact_type': junk_result.get('total_leads_for_contact_type', 0),
                    'junk_rate_percent': junk_result.get('junk_rate_percent', 0),
                    'contact_type_analyzed': contact_type
                },
                'top_junk_terms': junk_result.get('top_junk_terms', {}),
                'analysis_instructions': (
                    "CRITICAL: The 'junk_keywords_data' contains actual junk leads broken down by "
                    "Month + Campaign + Term combinations. Each row shows real junk lead counts. "
                    "Use this data to provide the month-by-month, campaign-wise breakdown requested. "
                    "The 'top_junk_terms' shows the most problematic keywords overall."
                ),
                'data_attribution': f"Junk leads analysis for Contact Type = {contact_type}"
            }
            
        except Exception as e:
            self.logger.error(f"Error generating junk keywords context: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._generate_error_context(str(e))
    
    def _generate_enhanced_analysis_context(self, question: str) -> Dict[str, Any]:
        """
        Generate enhanced context for general analysis questions
        Uses simplified data structures for Claude consumption
        """
        try:
            # Load data and get Claude-readable context
            claude_data = self.analyzer.get_claude_readable_data(question)
            
            # Determine analysis focus
            analysis_focus = self._identify_analysis_focus(question)
            
            # Get relevant analysis based on focus
            if analysis_focus == 'campaign_performance':
                campaign_analysis = self.analyzer.analyze_campaign_performance()
                context_data = self._simplify_campaign_context(campaign_analysis)
            elif analysis_focus == 'lead_quality':
                crm_analysis = self.analyzer.analyze_crm_performance()
                context_data = self._simplify_crm_context(crm_analysis)
            elif analysis_focus == 'budget_optimization':
                campaign_analysis = self.analyzer.analyze_campaign_performance()
                context_data = self._simplify_budget_context(campaign_analysis)
            else:
                # Comprehensive analysis
                campaign_analysis = self.analyzer.analyze_campaign_performance()
                crm_analysis = self.analyzer.analyze_crm_performance()
                context_data = self._simplify_comprehensive_context(campaign_analysis, crm_analysis)
            
            return {
                'context_type': 'enhanced_analysis',
                'question': question,
                'analysis_focus': analysis_focus,
                'data_available': claude_data.get('data_availability', {}).get('status') == 'available',
                'data_overview': claude_data.get('quick_stats', {}),
                'sample_records': claude_data.get('sample_records', []),
                'analysis_data': context_data,
                'data_attribution': self._generate_data_attribution_context(),
                'analysis_instructions': (
                    "Use the analysis_data to provide insights. The sample_records show "
                    "the data structure and types available for analysis."
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error generating enhanced analysis context: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._generate_error_context(str(e))
    
    def _extract_filters_from_question(self, question: str) -> Dict[str, Any]:
        """Extract filter parameters from question text"""
        question_lower = question.lower()
        filters = {}
        
        # Contact Type extraction (case-insensitive matching)
        if '3pl' in question_lower:
            filters['contact_type'] = '3pl'  # Use lowercase for consistent matching
        elif 'direct' in question_lower and ('contact type' in question_lower or 'contact_type' in question_lower):
            filters['contact_type'] = 'direct'
        elif 'broker' in question_lower and ('contact type' in question_lower or 'contact_type' in question_lower):
            filters['contact_type'] = 'broker'
        
        # Junk leads indicator
        if any(phrase in question_lower for phrase in ['junk leads', 'junk lead', 'creating junk']):
            filters['include_junk_only'] = True
        
        # Lead Stage
        if 'lead stage' in question_lower:
            if 'qualified' in question_lower:
                filters['lead_stage'] = 'Qualified'
            elif 'unqualified' in question_lower:
                filters['lead_stage'] = 'Unqualified'
        
        # Status filtering
        if 'status' in question_lower:
            if 'hot' in question_lower:
                filters['status'] = 'Hot'
            elif 'cold' in question_lower:
                filters['status'] = 'Cold'
            elif 'lost' in question_lower:
                filters['status'] = 'Lost'
        
        # Campaign specific filtering
        campaign_patterns = [
            ('lead-maximiseconv', 'Lead-MaximiseConv'),
            ('maximise conv', 'MaximiseConv'),
            ('lead maximise', 'Lead-Maximise')
        ]
        for pattern, campaign_name in campaign_patterns:
            if pattern in question_lower:
                filters['campaign'] = campaign_name
                break
        
        # Grouping strategy based on question phrasing
        if 'month-by-month' in question_lower or 'month-on-month' in question_lower:
            if 'campaign-wise' in question_lower or 'campaign wise' in question_lower:
                filters['group_by'] = ['Month Year', 'Campaign', 'Term']  # Month first for chronological sorting
            else:
                filters['group_by'] = ['Month Year', 'Term']
        elif 'campaign-wise' in question_lower or 'campaign wise' in question_lower:
            if 'term' in question_lower or 'keyword' in question_lower:
                filters['group_by'] = ['Campaign', 'Term', 'Month Year']
            else:
                filters['group_by'] = ['Campaign', 'Month Year']
        else:
            # Default comprehensive grouping
            filters['group_by'] = ['Month Year', 'Campaign', 'Term']
        
        return filters
    
    def _identify_analysis_focus(self, question: str) -> str:
        """Identify the primary focus of the analysis question"""
        question_lower = question.lower()
        
        if any(term in question_lower for term in ['budget', 'spend', 'cost', 'utilization', 'daily budget']):
            return 'budget_optimization'
        elif any(term in question_lower for term in ['lead', 'quality', 'score', 'junk', 'source', 'crm']):
            return 'lead_quality'
        elif any(term in question_lower for term in ['campaign', 'performance', 'conversion', 'ctr', 'cpc']):
            return 'campaign_performance'
        elif any(term in question_lower for term in ['device', 'mobile', 'desktop', 'tablet']):
            return 'device_performance'
        elif any(term in question_lower for term in ['optimize', 'improve', 'recommend', 'fix']):
            return 'optimization'
        else:
            return 'comprehensive'
    
    def _simplify_campaign_context(self, campaign_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Simplify campaign analysis for Claude consumption"""
        if not campaign_analysis.get('data_available', False):
            return {'error': 'No campaign data available'}
        
        return {
            'performance_overview': campaign_analysis.get('overview', {}),
            'monthly_performance': campaign_analysis.get('monthly_performance', {}),
            'budget_summary': campaign_analysis.get('budget_analysis', {}),
            'device_summary': campaign_analysis.get('device_insights', {}),
            'top_recommendations': campaign_analysis.get('recommendations', [])[:3]
        }
    
    def _simplify_crm_context(self, crm_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Simplify CRM analysis for Claude consumption"""
        if not crm_analysis.get('data_available', False):
            return {'error': 'No CRM data available'}
        
        return {
            'lead_overview': crm_analysis.get('overview', {}),
            'quality_summary': crm_analysis.get('lead_quality', {}),
            'source_performance': crm_analysis.get('source_performance', {}),
            'google_ads_summary': crm_analysis.get('google_ads_analysis', {}),
            'contact_type_breakdown': crm_analysis.get('contact_type_analysis', {}),
            'top_recommendations': crm_analysis.get('recommendations', [])[:3]
        }
    
    def _simplify_budget_context(self, campaign_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Simplify budget analysis for Claude consumption"""
        if not campaign_analysis.get('data_available', False):
            return {'error': 'No campaign data available for budget analysis'}
        
        budget_data = campaign_analysis.get('budget_analysis', {})
        return {
            'budget_utilization': budget_data.get('utilization_percent', 0),
            'total_spend': budget_data.get('total_spend', 0),
            'total_budget': budget_data.get('total_budget', 0),
            'utilization_status': budget_data.get('status', 'Unknown'),
            'monthly_breakdown': campaign_analysis.get('monthly_performance', {}).get('monthly_data', [])
        }
    
    def _simplify_comprehensive_context(self, campaign_analysis: Dict[str, Any], crm_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Simplify comprehensive analysis for Claude consumption"""
        return {
            'system_overview': {
                'campaign_data_available': campaign_analysis.get('data_available', False),
                'crm_data_available': crm_analysis.get('data_available', False),
                'total_spend': campaign_analysis.get('overview', {}).get('total_spend', 0),
                'total_leads': crm_analysis.get('overview', {}).get('total_leads', 0),
                'overall_efficiency': self._calculate_overall_efficiency(campaign_analysis, crm_analysis)
            },
            'key_metrics': {
                'budget_utilization': campaign_analysis.get('budget_analysis', {}).get('utilization_percent', 0),
                'lead_quality': crm_analysis.get('lead_quality', {}).get('quality_percentage', 0),
                'junk_rate': crm_analysis.get('overview', {}).get('junk_leads', 0) / max(crm_analysis.get('overview', {}).get('total_leads', 1), 1) * 100,
                'google_ads_performance': crm_analysis.get('google_ads_analysis', {})
            },
            'critical_issues': self._identify_critical_issues(campaign_analysis, crm_analysis),
            'top_opportunities': self._identify_top_opportunities(campaign_analysis, crm_analysis)
        }
    
    def _get_date_range_from_data(self, data_table: List[Dict]) -> str:
        """Extract date range from data table"""
        try:
            if not data_table:
                return 'No data'
            
            months = [row.get('Month Year', '') for row in data_table if row.get('Month Year')]
            if not months:
                return 'No date data'
            
            # Sort chronologically
            unique_months = list(set(months))
            month_dates = [(m, pd.to_datetime(m, format='%B %Y')) for m in unique_months]
            sorted_months = sorted(month_dates, key=lambda x: x[1])
            
            if len(sorted_months) == 1:
                return sorted_months[0][0]
            else:
                return f"{sorted_months[0][0]} to {sorted_months[-1][0]}"
                
        except Exception as e:
            return f"Date range error: {str(e)}"
    
    def _get_available_data_summary(self) -> Dict[str, Any]:
        """Get summary of available data for error context"""
        try:
            if self.analyzer.crm_data is not None:
                df = self.analyzer.crm_data
                return {
                    'total_records': len(df),
                    'contact_types': df['Contact Type'].value_counts().to_dict() if 'Contact Type' in df.columns else {},
                    'lead_sources': df['Lead Source'].value_counts().to_dict() if 'Lead Source' in df.columns else {},
                    'date_range': f"{df['Month Year'].min()} to {df['Month Year'].max()}" if 'Month Year' in df.columns else 'Unknown'
                }
            else:
                return {'error': 'No CRM data loaded'}
        except Exception as e:
            return {'error': str(e)}
    
    def _generate_data_attribution_context(self) -> str:
        """Generate clear data attribution context with chronological sorting"""
        try:
            attribution = []
            
            if self.analyzer.campaign_data is not None:
                campaign_count = len(self.analyzer.campaign_data)
                if 'Month' in self.analyzer.campaign_data.columns:
                    df_sorted = self.analyzer.campaign_data.copy()
                    df_sorted['_temp_date'] = pd.to_datetime(df_sorted['Month'], format='%B %Y', errors='coerce')
                    df_sorted = df_sorted.sort_values('_temp_date')
                    date_range = f"{df_sorted['Month'].iloc[0]} to {df_sorted['Month'].iloc[-1]}"
                else:
                    date_range = "Unknown period"
                attribution.append(f"Campaign Data: {campaign_count:,} records ({date_range})")
            
            if self.analyzer.crm_data is not None:
                crm_count = len(self.analyzer.crm_data)
                if 'Month Year' in self.analyzer.crm_data.columns:
                    df_sorted = self.analyzer.crm_data.copy()
                    df_sorted['_temp_date'] = pd.to_datetime(df_sorted['Month Year'], format='%B %Y', errors='coerce')
                    df_sorted = df_sorted.sort_values('_temp_date')
                    crm_range = f"{df_sorted['Month Year'].iloc[0]} to {df_sorted['Month Year'].iloc[-1]}"
                else:
                    crm_range = "Unknown period"
                attribution.append(f"CRM Data: {crm_count:,} lead records ({crm_range})")
            
            return " | ".join(attribution)
            
        except Exception as e:
            self.logger.error(f"Data attribution error: {str(e)}")
            return f"Data attribution error: {str(e)}"
    
    def _calculate_overall_efficiency(self, campaign_analysis: Dict, crm_analysis: Dict) -> Dict[str, Any]:
        """Calculate overall system efficiency metrics"""
        try:
            total_spend = campaign_analysis.get('overview', {}).get('total_spend', 0)
            total_leads = crm_analysis.get('overview', {}).get('total_leads', 0)
            
            return {
                'cost_per_lead': total_spend / max(total_leads, 1),
                'efficiency_status': 'Calculating...'
            }
        except Exception:
            return {'error': 'Cannot calculate efficiency'}
    
    def _identify_critical_issues(self, campaign_analysis: Dict, crm_analysis: Dict) -> List[str]:
        """Identify critical issues across campaign and CRM data"""
        issues = []
        
        try:
            # Budget issues
            budget_util = campaign_analysis.get('budget_analysis', {}).get('utilization_percent', 0)
            if budget_util < 50:
                issues.append(f"Severe budget underutilization: {budget_util:.1f}%")
            elif budget_util > 150:
                issues.append(f"Budget overspend: {budget_util:.1f}%")
            
            # Lead quality issues
            quality_pct = crm_analysis.get('lead_quality', {}).get('quality_percentage', 0)
            if quality_pct < 5:
                issues.append(f"Critical lead quality crisis: {quality_pct:.1f}% high quality")
            
            # Junk rate issues
            total_leads = crm_analysis.get('overview', {}).get('total_leads', 1)
            junk_leads = crm_analysis.get('overview', {}).get('junk_leads', 0)
            junk_rate = junk_leads / total_leads * 100
            if junk_rate > 50:
                issues.append(f"High junk rate: {junk_rate:.1f}%")
            
        except Exception as e:
            issues.append(f"Issue identification error: {str(e)}")
        
        return issues
    
    def _identify_top_opportunities(self, campaign_analysis: Dict, crm_analysis: Dict) -> List[str]:
        """Identify top optimization opportunities"""
        opportunities = []
        
        try:
            # Budget optimization opportunities
            budget_util = campaign_analysis.get('budget_analysis', {}).get('utilization_percent', 0)
            if budget_util < 80:
                opportunities.append("Increase budget allocation to capture more leads")
            
            # Device optimization
            device_summary = campaign_analysis.get('device_insights', {})
            if device_summary:
                opportunities.append("Optimize high-performing device campaigns")
            
            # Lead quality improvement
            quality_pct = crm_analysis.get('lead_quality', {}).get('quality_percentage', 0)
            if quality_pct < 20:
                opportunities.append("Implement stricter lead qualification criteria")
            
        except Exception as e:
            opportunities.append(f"Opportunity identification error: {str(e)}")
        
        return opportunities
    
    def _generate_error_context(self, error_message: str) -> Dict[str, Any]:
        """Generate error context for Claude"""
        return {
            'context_type': 'error',
            'error_message': error_message,
            'data_available': False,
            'suggestions': [
                'Check Google Sheets connectivity',
                'Verify data format and column headers',
                'Ensure sufficient data exists for the requested analysis',
                'Try broadening your search criteria'
            ],
            'timestamp': datetime.now().isoformat()
        }
    
    # LEGACY METHODS: Keep for compatibility but route through new system
    def generate_quick_insights(self, question: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        LEGACY METHOD: Route through new Claude context system
        Maintains compatibility while using improved data flow
        """
        try:
            # Use the new Claude context system
            claude_context = self.generate_claude_context(question)
            
            # Convert to old format for compatibility
            return {
                'question': question,
                'context_type': claude_context.get('context_type', 'unknown'),
                'data_available': claude_context.get('data_available', False),
                'insights': claude_context,
                'error': claude_context.get('error_message') if claude_context.get('context_type') == 'error' else None
            }
            
        except Exception as e:
            self.logger.error(f"Error in legacy generate_quick_insights: {str(e)}")
            return {
                'question': question,
                'error': str(e),
                'data_available': False
            }
    
    def generate_enhanced_context(self, question: str) -> Dict[str, Any]:
        """
        LEGACY METHOD: Route through new enhanced analysis context
        """
        try:
            return self._generate_enhanced_analysis_context(question)
        except Exception as e:
            return self._generate_error_context(str(e))
    
    def generate_filtered_analysis_context(self, question: str) -> Dict[str, Any]:
        """
        LEGACY METHOD: Route through new filtered data context
        """
        try:
            return self._generate_filtered_data_context(question)
        except Exception as e:
            return self._generate_error_context(str(e))
    
    # Keep existing report generation methods but simplify them
    def generate_executive_summary(self, analysis_type: str = 'full', months: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate executive summary with enhanced data context"""
        try:
            self.analyzer.load_data(months)
            
            summary = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'analysis_period': months or 'All available data',
                    'report_type': analysis_type,
                    'data_attribution': self._generate_data_attribution_context()
                },
                'key_performance_indicators': {},
                'critical_alerts': [],
                'strategic_recommendations': []
            }
            
            if analysis_type in ['campaign', 'full']:
                campaign_analysis = self.analyzer.analyze_campaign_performance(months)
                summary.update(self._format_executive_campaign_summary(campaign_analysis))
            
            if analysis_type in ['crm', 'full']:
                crm_analysis = self.analyzer.analyze_crm_performance(months)
                summary.update(self._format_executive_crm_summary(crm_analysis))
            
            if analysis_type == 'full':
                cross_channel = self.analyzer.cross_channel_analysis()
                summary.update(self._format_executive_cross_channel_summary(cross_channel))
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating executive summary: {str(e)}")
            return self._generate_error_context(str(e))
    
    def generate_detailed_report(self, analysis_type: str = 'full', months: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate detailed report with enhanced insights"""
        try:
            self.analyzer.load_data(months)
            
            report = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'analysis_period': months or 'All available data',
                    'report_type': analysis_type,
                    'data_attribution': self._generate_data_attribution_context()
                },
                'executive_summary': {},
                'detailed_analysis': {},
                'actionable_recommendations': []
            }
            
            if analysis_type in ['campaign', 'full']:
                campaign_analysis = self.analyzer.analyze_campaign_performance(months)
                report['detailed_analysis']['campaign_performance'] = self._simplify_campaign_context(campaign_analysis)
            
            if analysis_type in ['crm', 'full']:
                crm_analysis = self.analyzer.analyze_crm_performance(months)
                report['detailed_analysis']['lead_generation'] = self._simplify_crm_context(crm_analysis)
            
            if analysis_type == 'full':
                cross_channel = self.analyzer.cross_channel_analysis()
                report['detailed_analysis']['cross_channel'] = cross_channel
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating detailed report: {str(e)}")
            return self._generate_error_context(str(e))
    
    # Simplified formatting methods
    def _format_executive_campaign_summary(self, campaign_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Format campaign analysis for executive summary"""
        if not campaign_analysis.get('data_available', False):
            return {'campaign_summary': 'No campaign data available'}
        
        overview = campaign_analysis.get('overview', {})
        budget = campaign_analysis.get('budget_analysis', {})
        
        return {
            'campaign_summary': {
                'total_investment': f"${overview.get('total_spend', 0):,.0f}",
                'budget_utilization': f"{budget.get('utilization_percent', 0):.1f}%",
                'total_conversions': f"{overview.get('total_conversions', 0):,}",
                'utilization_status': budget.get('status', 'Unknown')
            }
        }
    
    def _format_executive_crm_summary(self, crm_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Format CRM analysis for executive summary"""
        if not crm_analysis.get('data_available', False):
            return {'lead_summary': 'No CRM data available'}
        
        overview = crm_analysis.get('overview', {})
        quality = crm_analysis.get('lead_quality', {})
        
        return {
            'lead_summary': {
                'total_leads': f"{overview.get('total_leads', 0):,}",
                'quality_percentage': f"{quality.get('quality_percentage', 0):.1f}%",
                'google_ads_leads': f"{overview.get('google_ads_leads', 0):,}",
                'junk_rate': f"{(overview.get('junk_leads', 0) / max(overview.get('total_leads', 1), 1) * 100):.1f}%"
            }
        }
    
    def _format_executive_cross_channel_summary(self, cross_channel: Dict[str, Any]) -> Dict[str, Any]:
        """Format cross-channel analysis for executive summary"""
        if not cross_channel.get('data_available', False):
            return {'efficiency_summary': 'No cross-channel data available'}
        
        return {
            'efficiency_summary': {
                'average_cost_per_lead': f"${cross_channel.get('average_cost_per_lead', 0):.2f}",
                'most_efficient_month': cross_channel.get('best_efficiency_month', 'Unknown'),
                'least_efficient_month': cross_channel.get('worst_efficiency_month', 'Unknown')
            }
        }