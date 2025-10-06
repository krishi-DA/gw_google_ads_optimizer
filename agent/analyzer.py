# analyzer.py
# Enhanced Core analysis engine for Google Ads campaigns and CRM data
# Built to extract deep insights from 5,500+ CRM rows with rich column analysis
# FIXED: Proper data structure for Claude API consumption

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta
from data_reader import GoogleSheetsDataReader
import json
import statistics
from collections import defaultdict, Counter
import re
from textblob import TextBlob

class CampaignCRMAnalyzer:
    """
    Enhanced analysis engine for Google Ads campaigns and CRM data
    Extracts deep insights from rich CRM data structure for Claude analysis
    FIXED: Improved data accessibility for Claude API
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_reader = GoogleSheetsDataReader()
        self.campaign_data = None
        self.crm_data = None
        self.monthly_campaign_data = None
        self.monthly_crm_data = None
        
    def load_data(self, months: Optional[List[str]] = None):
        """Load and cache data from Google Sheets"""
        try:
            self.campaign_data = self.data_reader.read_campaign_data(months)
            self.crm_data = self.data_reader.read_crm_data(months)
            self.monthly_campaign_data = self.data_reader.get_monthly_aggregated_campaign_data()
            self.monthly_crm_data = self.data_reader.get_monthly_aggregated_crm_data()
            
            self.logger.info("Data loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    # NEW: Primary method for Claude data access
    def get_claude_readable_data(self, query_context: str = "") -> Dict[str, Any]:
        """
        MAIN METHOD: Provides Claude with properly structured, accessible data
        NOW with full dataset context and aggregations
        """
        try:
            if self.crm_data is None:
                self.load_data()
            
            # CRITICAL: Make it explicit this is a large dataset
            readable_data = {
                'IMPORTANT_CONTEXT': {
                    'total_crm_records': len(self.crm_data),
                    'total_campaign_records': len(self.campaign_data) if self.campaign_data is not None else 0,
                    'note': 'The sample_records below are EXAMPLES ONLY. The full dataset contains thousands of records as indicated above.'
                },
                
                'data_availability': self._check_data_availability(),
                'data_structure': self._get_data_structure_info(),
                'quick_stats': self._get_quick_stats(),
                
                # For budget questions, include aggregated campaign performance
                'campaign_aggregations': self.get_campaign_performance_for_budget_optimization('3pl') if '3pl' in query_context.lower() or 'budget' in query_context.lower() else None,
                
                'sample_records_for_reference_only': self._get_sample_records_for_claude(),
                'query_context': query_context
            }
            
            return readable_data
            
        except Exception as e:
            self.logger.error(f"Error getting Claude readable data: {str(e)}")
            return {'error': str(e), 'data_available': False}
    
    def _check_data_availability(self) -> Dict[str, Any]:
        """Check what data is available for analysis"""
        availability = {
            'status': 'available',
            'crm_records': len(self.crm_data) if self.crm_data is not None else 0,
            'campaign_records': len(self.campaign_data) if self.campaign_data is not None else 0,
            'key_columns_present': [],
            'date_range': 'Unknown'
        }
        
        if self.crm_data is not None:
            # Check key columns
            key_columns = ['Campaign', 'Term', 'Contact Type', 'Lead Source', 'Status', 'Month Year', 'Score']
            availability['key_columns_present'] = [col for col in key_columns if col in self.crm_data.columns]
            
            # Date range
            if 'Month Year' in self.crm_data.columns:
                availability['date_range'] = f"{self.crm_data['Month Year'].min()} to {self.crm_data['Month Year'].max()}"
        
        return availability
    
    def _get_data_structure_info(self) -> Dict[str, Any]:
        """Provide Claude with clear info about data structure"""
        if self.crm_data is None:
            return {'error': 'No CRM data available'}
        
        df = self.crm_data
        
        structure_info = {
            'total_records': len(df),
            'columns_available': list(df.columns),
            'contact_types': df['Contact Type'].value_counts().to_dict() if 'Contact Type' in df.columns else {},
            'lead_sources': df['Lead Source'].value_counts().to_dict() if 'Lead Source' in df.columns else {},
            'campaigns_available': df['Campaign'].nunique() if 'Campaign' in df.columns else 0,
            'terms_available': df['Term'].nunique() if 'Term' in df.columns else 0,
            'months_covered': df['Month Year'].nunique() if 'Month Year' in df.columns else 0
        }
        
        return structure_info
    
    def _get_sample_records_for_claude(self) -> List[Dict[str, Any]]:
        """Provide Claude with sample records to understand data format"""
        if self.crm_data is None:
            return []
        
        df = self.crm_data
        samples = []
        
        # Get diverse samples
        sample_types = [
            ('Google Ads Lead', df[df['Lead Source'] == 'Web Mail']),
            ('3PL Contact', df[df['Contact Type'].astype(str).str.contains('3pl', case=False, na=False)]),
            ('Junk Lead', df[df['Contact Type'] == 'Junk']),
            ('High Score Lead', df[df['Score'] > 70] if 'Score' in df.columns else pd.DataFrame())
        ]
        
        for sample_type, sample_df in sample_types:
            if len(sample_df) > 0:
                sample_record = sample_df.iloc[0]
                samples.append({
                    'type': sample_type,
                    'data': {
                        'Campaign': sample_record.get('Campaign', ''),
                        'Term': sample_record.get('Term', ''),
                        'Contact_Type': sample_record.get('Contact Type', ''),
                        'Lead_Source': sample_record.get('Lead Source', ''),
                        'Status': sample_record.get('Status', ''),
                        'Score': sample_record.get('Score', 0),
                        'Month_Year': sample_record.get('Month Year', ''),
                        'Location': sample_record.get('Locations', ''),
                        'Area_Requirement': sample_record.get('Area Requirement', 0)
                    }
                })
        
        return samples
    
    def _get_filterable_data_table(self) -> pd.DataFrame:
        """Provide Claude with a clean, filterable data table"""
        if self.crm_data is None:
            return pd.DataFrame()
        
        df = self.crm_data.copy()
        
        # Select key columns for Claude analysis
        key_columns = ['Month Year', 'Campaign', 'Term', 'Contact Type', 'Lead Source', 'Status', 'Score', 'Locations', 'Area Requirement']
        available_columns = [col for col in key_columns if col in df.columns]
        
        # Create clean table
        clean_table = df[available_columns].copy()
        
        # Clean data for better readability
        for col in clean_table.columns:
            if clean_table[col].dtype == 'object':
                clean_table[col] = clean_table[col].fillna('Unknown')
            else:
                clean_table[col] = clean_table[col].fillna(0)
        
        return clean_table.head(100)  # Limit to first 100 records for Claude
    
    def _get_quick_stats(self) -> Dict[str, Any]:
        """Provide Claude with quick stats for context"""
        if self.crm_data is None:
            return {}
        
        df = self.crm_data
        
        stats = {
            'total_leads': len(df),
            'google_ads_leads': len(df[df['Lead Source'] == 'Web Mail']) if 'Lead Source' in df.columns else 0,
            'junk_leads': len(df[df['Contact Type'] == 'Junk']) if 'Contact Type' in df.columns else 0,
            'contact_type_breakdown': df['Contact Type'].value_counts().to_dict() if 'Contact Type' in df.columns else {},
            'avg_score': float(df['Score'].mean()) if 'Score' in df.columns else 0,
            'unique_campaigns': df['Campaign'].nunique() if 'Campaign' in df.columns else 0,
            'unique_terms': df['Term'].nunique() if 'Term' in df.columns else 0
        }
        
        # Calculate junk rate
        if stats['total_leads'] > 0:
            stats['junk_rate_percent'] = (stats['junk_leads'] / stats['total_leads']) * 100
        
        return stats
    
    # NEW: Direct query methods for Claude
    def query_junk_keywords_by_contact_type(self, contact_type: str = "3pl") -> Dict[str, Any]:
        """
        FIXED: Get junk keywords by contact type for Claude
        """
        try:
            if self.crm_data is None:
                self.load_data()

            df = self.crm_data.copy()
            
            # FIXED: Handle array-type Contact_Type filtering
            def matches_contact_type(contact_type_value, target_type):
                if pd.isna(contact_type_value):
                    return False
                contact_str = str(contact_type_value).lower()
                return target_type.lower() in contact_str

            # FIXED: Filter for specific contact type
            contact_filtered = df[df.apply(lambda row: matches_contact_type(row.get('Contact Type'), contact_type), axis=1)]

            if len(contact_filtered) == 0:
                return {
                    'error': f'No leads found for contact type containing "{contact_type}"',
                    'available_contact_types': self._get_contact_type_sample(df)
                }

            # FIXED: Define junk criteria for actual data
            def is_junk_lead(row):
                contact_type_val = row.get('Contact Type', '')
                status_val = row.get('Status', '')
                score_val = row.get('Score', 0)
                
                # Check if explicitly marked as junk
                if pd.notna(contact_type_val):
                    contact_str = str(contact_type_val).lower()
                    if 'junk' in contact_str:
                        return True
                
                # Check status-based junk indicators
                if pd.notna(status_val):
                    status_str = str(status_val).lower()
                    if any(indicator in status_str for indicator in ['junk', 'lost', 'closed', 'cold']):
                        return True
                
                # Score-based (very low scores)
                if pd.notna(score_val) and float(score_val) < 0:  # Negative scores
                    return True
                    
                return False

            junk_leads = contact_filtered[contact_filtered.apply(is_junk_lead, axis=1)]

            if len(junk_leads) == 0:
                return {
                    'message': f'No junk leads found for contact type "{contact_type}"',
                    'total_leads_for_contact_type': len(contact_filtered),
                    'sample_leads': contact_filtered[['Campaign', 'Term', 'Status', 'Score', 'Month Year']].head(5).to_dict('records')
                }

            # Group by month, campaign, and term
            junk_analysis = junk_leads.groupby(['Month Year', 'Campaign', 'Term']).agg({
                'ID': 'count',
                'Score': 'mean'
            }).reset_index()

            junk_analysis.columns = ['Month_Year', 'Campaign', 'Term', 'Junk_Lead_Count', 'Avg_Score']
            junk_analysis['Avg_Score'] = junk_analysis['Avg_Score'].round(2)

            # Sort chronologically
            junk_analysis['_temp_date'] = pd.to_datetime(junk_analysis['Month_Year'], format='%B %Y', errors='coerce')
            junk_analysis = junk_analysis.sort_values(['_temp_date', 'Junk_Lead_Count'], ascending=[True, False])
            junk_analysis = junk_analysis.drop('_temp_date', axis=1)

            # Get top junk terms overall
            top_junk_terms = junk_leads['Term'].value_counts().head(10).to_dict()

            return {
                'query_summary': f'Junk keywords for Contact Type "{contact_type}"',
                'total_junk_leads': len(junk_leads),
                'total_leads_for_contact_type': len(contact_filtered),
                'junk_rate_percent': round((len(junk_leads) / len(contact_filtered)) * 100, 2),
                'month_campaign_term_breakdown': junk_analysis.to_dict('records'),
                'top_junk_terms': top_junk_terms,
                'data_available': True
            }

        except Exception as e:
            self.logger.error(f"Error querying junk keywords: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'error': str(e),
                'data_available': False
            }
    
    def get_filtered_lead_analysis(
        self,
        contact_type: Optional[str] = None,
        lead_stage: Optional[str] = None,
        status: Optional[str] = None,
        campaign: Optional[str] = None,
        include_junk_only: bool = False,
        group_by: List[str] = ['Campaign', 'Term', 'Month Year']
    ) -> Dict[str, Any]:
        """
        ENHANCED: Get filtered and grouped lead data with better Claude accessibility
        Returns structured data that Claude can easily parse and analyze
        """
        try:
            if self.crm_data is None:
                self.load_data()
            
            df = self.crm_data.copy()
            original_count = len(df)
            
            # Step 1: Filter by contact type (handle case-insensitive)
            if contact_type:
                contact_type_lower = contact_type.lower()
                df = df[df['Contact Type'].astype(str).str.lower().str.contains(contact_type_lower, na=False)]
            
            # Step 2: Filter to junk leads if requested
            if include_junk_only:
                junk_mask = (
                    df['Status'].astype(str).str.lower().str.contains('junk', na=False) |
                    df['Lead Stage'].astype(str).str.lower().str.contains('junk', na=False) |
                    df['Status'].astype(str).str.lower().isin(['closed', 'lost']) |
                    (df['Score'] < 3)
                )
                df = df[junk_mask]
            
            # Step 3: Apply other filters
            if lead_stage:
                df = df[df['Lead Stage'] == lead_stage]
            
            if status:
                df = df[df['Status'].astype(str).str.contains(status, case=False, na=False)]
            
            if campaign:
                df = df[df['Campaign'].str.contains(campaign, case=False, na=False)]
            
            # Check if we have any data left
            if len(df) == 0:
                return {
                    'query_result': 'No data found',
                    'filter_summary': {
                        'contact_type': contact_type,
                        'include_junk_only': include_junk_only,
                        'lead_stage': lead_stage,
                        'status': status,
                        'campaign': campaign
                    },
                    'original_count': original_count,
                    'filtered_count': 0,
                    'data_available': False,
                    'suggested_alternative': 'Try broadening your filter criteria'
                }
            
            # Ensure grouping columns exist
            valid_group_cols = [col for col in group_by if col in df.columns]
            
            if not valid_group_cols:
                return {
                    'error': f'No valid grouping columns found from {group_by}',
                    'available_columns': list(df.columns),
                    'data_available': False
                }
            
            # Remove rows with NaN in grouping columns
            df = df.dropna(subset=valid_group_cols)
            
            if len(df) == 0:
                return {
                    'error': 'No data after removing NaN values in grouping columns',
                    'data_available': False
                }
            
            # Group and aggregate
            result = df.groupby(valid_group_cols, dropna=False).agg({
                'ID': 'count',
                'Score': 'mean',
                'Email': lambda x: x.notna().sum()
            }).reset_index()
            
            result.columns = valid_group_cols + ['Lead_Count', 'Avg_Score', 'Leads_With_Contact']
            
            # Sort chronologically if Month Year is present
            if 'Month Year' in result.columns:
                result['_temp_date'] = pd.to_datetime(result['Month Year'], format='%B %Y', errors='coerce')
                sort_cols = ['_temp_date']
                if 'Campaign' in result.columns:
                    sort_cols.insert(0, 'Campaign')
                if 'Term' in result.columns:
                    sort_cols.insert(1, 'Term')
                
                result = result.sort_values(sort_cols)
                result = result.drop('_temp_date', axis=1)
            
            # Round numeric columns
            result['Avg_Score'] = result['Avg_Score'].round(2)
            
            # Return structured data for Claude
            return {
                'query_result': 'Data found and processed',
                'filter_summary': {
                    'contact_type': contact_type,
                    'include_junk_only': include_junk_only,
                    'lead_stage': lead_stage,
                    'status': status,
                    'campaign': campaign,
                    'group_by': valid_group_cols
                },
                'original_count': original_count,
                'filtered_count': len(df),
                'grouped_result_count': len(result),
                'data_table': result.to_dict('records'),  # Convert to list of dicts for Claude
                'data_available': True,
                'summary_stats': {
                    'total_leads_in_result': result['Lead_Count'].sum(),
                    'avg_score_overall': result['Avg_Score'].mean().round(2),
                    'top_terms_by_count': result.nlargest(5, 'Lead_Count')[['Term', 'Lead_Count']].to_dict('records') if 'Term' in result.columns else []
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in get_filtered_lead_analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'error': str(e),
                'data_available': False
            }
    
    # Keep existing analysis methods but make them return cleaner structures
    def analyze_campaign_performance(self, months: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Simplified campaign performance analysis for Claude consumption
        """
        try:
            if self.monthly_campaign_data is None:
                self.load_data(months)
            
            if self.monthly_campaign_data is None or len(self.monthly_campaign_data) == 0:
                return {'error': 'No campaign data available', 'data_available': False}
            
            # Simplified analysis structure
            analysis = {
                'data_available': True,
                'overview': self._calculate_overall_metrics(),
                'monthly_performance': self._get_monthly_performance_summary(),
                'budget_analysis': self._get_budget_summary(),
                'device_insights': self._get_device_summary(),
                'recommendations': self._generate_campaign_recommendations()[:3]  # Limit to top 3
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing campaign performance: {str(e)}")
            return {'error': str(e), 'data_available': False}
    
    def get_campaign_performance_for_budget_optimization(self, contact_type: str = "3pl") -> Dict[str, Any]:
            """PRIMARY METHOD for budget optimization questions"""
            try:
                if self.crm_data is None or self.campaign_data is None:
                    self.load_data()
                
                crm_df = self.crm_data.copy()
                campaign_df = self.campaign_data.copy()
                
                # Filter CRM data
                filtered_crm = crm_df[crm_df['Contact Type'].astype(str).str.contains(contact_type, case=False, na=False)]
                
                if len(filtered_crm) == 0:
                    return {'error': f'No leads for "{contact_type}"', 'data_available': False}
                
                # CRM aggregation
                crm_agg = filtered_crm.groupby('Campaign').agg({
                    'ID': 'count',
                    'Score': 'mean',
                    'Status': lambda x: x.isin(['Hot', 'Converted']).sum()
                }).reset_index()
                crm_agg.columns = ['Campaign', 'Total_Leads', 'Avg_Score', 'CRM_Conversions']
                
                # Junk count
                junk_df = filtered_crm[filtered_crm['Contact Type'].astype(str).str.contains('junk', case=False, na=False)]
                junk_agg = junk_df.groupby('Campaign').size().reset_index(name='Junk_Leads')
                crm_agg = crm_agg.merge(junk_agg, on='Campaign', how='left').fillna({'Junk_Leads': 0})
                
                # Campaign aggregation
                camp_agg = campaign_df.groupby('Campaign Name').agg({
                    'Cost': 'sum',
                    'Clicks': 'sum',
                    'Daily Budget': 'mean'
                }).reset_index()
                
                # Merge
                merged = crm_agg.merge(camp_agg, left_on='Campaign', right_on='Campaign Name', how='left')
                merged.fillna({'Cost': 0, 'Clicks': 0, 'Daily Budget': 0}, inplace=True)
                
                # Metrics
                merged['Cost_Per_Lead'] = merged['Cost'] / merged['Total_Leads'].replace(0, 1)
                merged['Lead_Quality_Rate'] = ((merged['Total_Leads'] - merged['Junk_Leads']) / merged['Total_Leads'] * 100).round(2)
                merged['Conversion_Rate'] = (merged['CRM_Conversions'] / merged['Total_Leads'] * 100).round(2)
                merged['ROI_Score'] = (merged['Lead_Quality_Rate'] * 0.5 + merged['Conversion_Rate'] * 0.5).round(2)
                
                merged = merged.sort_values('ROI_Score', ascending=False)
                
                return {
                    'data_available': True,
                    'unique_campaigns': int(merged['Campaign'].nunique()),
                    'contact_type_filter': contact_type,
                    'overall_summary': {
                        'total_spend': float(merged['Cost'].sum()),
                        'total_leads_generated': int(merged['Total_Leads'].sum()),
                        'total_junk_leads': int(merged['Junk_Leads'].sum()),
                        'overall_cost_per_lead': float(merged['Cost'].sum() / merged['Total_Leads'].sum())
                    },
                    'campaign_performance_table': merged[['Campaign', 'Total_Leads', 'Cost', 'Cost_Per_Lead', 'Lead_Quality_Rate', 'ROI_Score', 'Daily Budget']].to_dict('records'),
                    'top_5_campaigns_by_roi': merged.head(5)[['Campaign', 'Total_Leads', 'Cost_Per_Lead', 'ROI_Score']].to_dict('records'),
                    'budget_optimization_recommendations': []
                }
                
            except Exception as e:
                self.logger.error(f"Budget optimization error: {str(e)}")
                import traceback
                traceback.print_exc()
                return {'data_available': False, 'error': str(e)}

    def _generate_budget_allocation_recommendations(self, campaign_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate specific budget allocation recommendations"""
        recommendations = []
        
        # Top performers - increase budget
        top_campaigns = campaign_df.head(3)
        for _, row in top_campaigns.iterrows():
            current_budget = row.get('Daily Budget', 0)
            recommended_increase = current_budget * 0.3  # 30% increase
            recommendations.append({
                'campaign': row['Campaign'],
                'action': 'INCREASE',
                'current_daily_budget': float(current_budget),
                'recommended_daily_budget': float(current_budget + recommended_increase),
                'reason': f"High ROI ({row['ROI_Score']:.1f}), quality leads ({row['Lead_Quality_Rate']:.1f}%)",
                'priority': 'HIGH'
            })
        
        # Poor performers - reduce or pause
        bottom_campaigns = campaign_df.tail(2)
        for _, row in bottom_campaigns.iterrows():
            if row['ROI_Score'] < 30:  # Low threshold
                recommendations.append({
                    'campaign': row['Campaign'],
                    'action': 'REDUCE or PAUSE',
                    'current_daily_budget': float(row.get('Daily Budget', 0)),
                    'recommended_daily_budget': 0,
                    'reason': f"Low ROI ({row['ROI_Score']:.1f}), high junk rate ({row['Junk_Leads']}/{row['Total_Leads']})",
                    'priority': 'HIGH'
                })
        
        return recommendations
    
    def analyze_crm_performance(self, months: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Simplified CRM performance analysis for Claude consumption
        """
        try:
            if self.monthly_crm_data is None:
                self.load_data(months)
            
            if self.crm_data is None or len(self.crm_data) == 0:
                return {'error': 'No CRM data available', 'data_available': False}
            
            # Simplified analysis structure
            analysis = {
                'data_available': True,
                'overview': self._get_crm_overview(),
                'lead_quality': self._get_lead_quality_summary(),
                'source_performance': self._get_source_summary(),
                'google_ads_analysis': self._get_google_ads_summary(),
                'contact_type_analysis': self._get_contact_type_summary(),
                'recommendations': self._generate_crm_recommendations()[:3]  # Limit to top 3
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing CRM performance: {str(e)}")
            return {'error': str(e), 'data_available': False}
    
    # Simplified helper methods for cleaner output
    def _get_monthly_performance_summary(self) -> Dict[str, Any]:
        """Get simplified monthly performance"""
        if self.monthly_campaign_data is None:
            return {}
        
        df = self.monthly_campaign_data
        
        return {
            'total_months': len(df),
            'best_month': df.loc[df['Conversion Rate'].idxmax(), 'Month'] if 'Conversion Rate' in df.columns else 'Unknown',
            'worst_month': df.loc[df['Conversion Rate'].idxmin(), 'Month'] if 'Conversion Rate' in df.columns else 'Unknown',
            'monthly_data': df[['Month', 'Cost', 'Conversions', 'Conversion Rate']].to_dict('records') if all(col in df.columns for col in ['Month', 'Cost', 'Conversions', 'Conversion Rate']) else []
        }
    
    def _get_budget_summary(self) -> Dict[str, Any]:
        """Get simplified budget analysis"""
        if self.monthly_campaign_data is None:
            return {}
        
        df = self.monthly_campaign_data
        total_cost = df['Cost'].sum() if 'Cost' in df.columns else 0
        total_budget = df['Daily Budget'].sum() if 'Daily Budget' in df.columns else 0
        utilization = (total_cost / total_budget * 100) if total_budget > 0 else 0
        
        return {
            'total_spend': float(total_cost),
            'total_budget': float(total_budget),
            'utilization_percent': float(utilization),
            'status': 'Under-utilized' if utilization < 80 else 'Over-utilized' if utilization > 100 else 'Well-utilized'
        }
    
    def _get_device_summary(self) -> Dict[str, Any]:
        """Get simplified device performance"""
        device_analysis = self._analyze_device_performance()
        device_metrics = device_analysis.get('device_metrics', {})
        
        summary = {}
        for device, metrics in device_metrics.items():
            summary[device] = {
                'clicks': metrics.get('clicks', 0),
                'conversions': metrics.get('conversions', 0),
                'conversion_rate': metrics.get('conversion_rate', 0),
                'cost': metrics.get('cost', 0)
            }
        
        return summary
    
    def _get_crm_overview(self) -> Dict[str, Any]:
        """Get simplified CRM overview - FIXED for actual data format"""
        if self.crm_data is None:
            return {}

        df = self.crm_data

        # FIXED: Handle array-type Contact_Type
        def is_junk_lead(contact_type):
            if pd.isna(contact_type):
                return False
            contact_str = str(contact_type).lower()
            return 'junk' in contact_str

        # FIXED: Count junk leads properly
        junk_leads = df.apply(lambda row: is_junk_lead(row.get('Contact Type')), axis=1).sum()
        
        # FIXED: Count Google Ads leads
        google_ads_leads = len(df[df['Lead Source'] == 'Web Mail']) if 'Lead Source' in df.columns else 0
        
        # FIXED: Handle score calculation with actual score distribution
        total_leads = len(df)
        avg_score = float(df['Score'].mean()) if 'Score' in df.columns else 0
        
        # FIXED: Use more realistic quality threshold since most scores are 0
        high_quality_leads = len(df[df['Score'] > 0]) if 'Score' in df.columns else 0  # Changed from >70 to >0
        conversion_rate = len(df[df['Status'].isin(['Hot', 'Converted'])]) / len(df) * 100 if 'Status' in df.columns and len(df) > 0 else 0

        return {
            'total_leads': total_leads,
            'avg_score': avg_score,
            'google_ads_leads': google_ads_leads,
            'junk_leads': junk_leads,
            'high_quality_leads': high_quality_leads,
            'conversion_rate': conversion_rate
        }

    def _get_lead_quality_summary(self) -> Dict[str, Any]:
        """Get simplified lead quality analysis - FIXED for actual data format"""
        if self.crm_data is None:
            return {}

        df = self.crm_data

        # FIXED: Handle array-type Contact_Type for junk detection
        def is_junk_lead(contact_type):
            if pd.isna(contact_type):
                return False
            contact_str = str(contact_type).lower()
            return 'junk' in contact_str

        junk_count = df.apply(lambda row: is_junk_lead(row.get('Contact Type')), axis=1).sum()
        total_leads = len(df)
        
        # FIXED: Use realistic quality thresholds
        if 'Score' in df.columns:
            scores = df['Score'].dropna()
            avg_score = float(scores.mean()) if len(scores) > 0 else 0
            
            # FIXED: Adjust quality thresholds for actual data
            high_quality_count = len(scores[scores > 0])  # Any score > 0
            medium_quality_count = len(scores[scores == 0])  # Exactly 0
            low_quality_count = junk_count  # Junk leads
            
            quality_percentage = (high_quality_count / total_leads * 100) if total_leads > 0 else 0
        else:
            avg_score = 0
            high_quality_count = 0
            medium_quality_count = 0
            low_quality_count = junk_count
            quality_percentage = 0

        return {
            'high_quality_count': high_quality_count,
            'quality_percentage': quality_percentage,
            'avg_score': avg_score,
            'score_distribution': {
                'high_score_above_0': high_quality_count,
                'zero_score': medium_quality_count,
                'junk_leads': low_quality_count
            },
            'junk_count': junk_count,
            'junk_rate': (junk_count / total_leads * 100) if total_leads > 0 else 0
        }
    
    def _get_source_summary(self) -> Dict[str, Any]:
        """Get simplified source analysis"""
        if self.crm_data is None:
            return {}
        
        df = self.crm_data
        
        return {
            'source_distribution': df['Lead Source'].value_counts().to_dict() if 'Lead Source' in df.columns else {},
            'top_sources': df['Lead Source'].value_counts().head(5).to_dict() if 'Lead Source' in df.columns else {}
        }
    
    def _get_google_ads_summary(self) -> Dict[str, Any]:
        """Get simplified Google Ads analysis"""
        if self.crm_data is None:
            return {}
        
        df = self.crm_data
        google_ads = df[df['Lead Source'] == 'Web Mail'] if 'Lead Source' in df.columns else pd.DataFrame()
        
        if len(google_ads) == 0:
            return {'error': 'No Google Ads leads found'}
        
        return {
            'total_google_ads_leads': len(google_ads),
            'junk_rate': len(google_ads[google_ads['Contact Type'] == 'Junk']) / len(google_ads) * 100 if 'Contact Type' in google_ads.columns else 0,
            'avg_score': float(google_ads['Score'].mean()) if 'Score' in google_ads.columns else 0,
            'top_campaigns': google_ads['Campaign'].value_counts().head(5).to_dict() if 'Campaign' in google_ads.columns else {},
            'top_terms': google_ads['Term'].value_counts().head(5).to_dict() if 'Term' in google_ads.columns else {}
        }
    
    def _get_contact_type_sample(self, df) -> Dict[str, int]:
        """Get sample of contact types for debugging"""
        contact_types = {}
        for contact_type in df['Contact Type'].dropna().head(10):
            contact_str = str(contact_type)
            if len(contact_str) > 50:
                contact_str = contact_str[:50] + "..."
            contact_types[contact_str] = contact_types.get(contact_str, 0) + 1
        return contact_types
    
    def _get_contact_type_summary(self) -> Dict[str, Any]:
        """Get simplified contact type analysis"""
        if self.crm_data is None:
            return {}
        
        df = self.crm_data
        
        summary = {}
        for contact_type in df['Contact Type'].value_counts().head(5).index:
            type_data = df[df['Contact Type'] == contact_type]
            summary[contact_type] = {
                'count': len(type_data),
                'percentage': len(type_data) / len(df) * 100,
                'avg_score': float(type_data['Score'].mean()) if 'Score' in type_data.columns else 0,
                'conversion_rate': len(type_data[type_data['Status'].isin(['Hot', 'Converted'])]) / len(type_data) * 100 if 'Status' in type_data.columns and len(type_data) > 0 else 0
            }
        
        return summary
    
    # Keep existing calculation methods for compatibility
    def _calculate_overall_metrics(self) -> Dict[str, Any]:
        """Calculate overall campaign metrics using exact column names"""
        try:
            df = self.monthly_campaign_data
            
            if df is None or len(df) == 0:
                return self._get_empty_metrics()
            
            # Sum totals using exact column names
            total_cost = df['Cost'].sum() if 'Cost' in df.columns else 0
            total_daily_budget = df['Daily Budget'].sum() if 'Daily Budget' in df.columns else 0
            total_impressions = df['Impressions'].sum() if 'Impressions' in df.columns else 0
            total_clicks = df['Clicks'].sum() if 'Clicks' in df.columns else 0
            total_conversions = df['Conversions'].sum() if 'Conversions' in df.columns else 0
            
            # Calculate metrics
            budget_utilization = (total_cost / total_daily_budget * 100) if total_daily_budget > 0 else 0
            overall_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
            overall_conversion_rate = (total_conversions / total_clicks * 100) if total_clicks > 0 else 0
            overall_cpc = (total_cost / total_clicks) if total_clicks > 0 else 0
            overall_cost_per_conversion = (total_cost / total_conversions) if total_conversions > 0 else 0
            
            return {
                'total_spend': float(total_cost),
                'total_daily_budget': float(total_daily_budget),
                'budget_utilization': float(budget_utilization),
                'total_impressions': int(total_impressions),
                'total_clicks': int(total_clicks),
                'total_conversions': int(total_conversions),
                'overall_ctr': float(overall_ctr),
                'overall_conversion_rate': float(overall_conversion_rate),
                'overall_cpc': float(overall_cpc),
                'overall_cost_per_conversion': float(overall_cost_per_conversion),
                'active_campaigns': int(df['Active Campaigns'].sum()) if 'Active Campaigns' in df.columns else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating overall metrics: {str(e)}")
            return self._get_empty_metrics()
    
    def _analyze_device_performance(self) -> Dict[str, Any]:
        """Analyze device performance using exact column names"""
        try:
            df = self.monthly_campaign_data
            
            if df is None or len(df) == 0:
                return self._get_empty_device_analysis()
            
            # Device columns as per your headers
            device_metrics = {}
            devices = ['Mobile', 'Desktop', 'Tablet']
            
            total_clicks = 0
            total_cost = 0
            total_conversions = 0
            
            # Calculate totals first
            for device in devices:
                clicks_col = f'{device} Clicks'
                cost_col = f'{device} Cost'
                conversions_col = f'{device} Conversions'
                
                if all(col in df.columns for col in [clicks_col, cost_col, conversions_col]):
                    total_clicks += df[clicks_col].sum()
                    total_cost += df[cost_col].sum()
                    total_conversions += df[conversions_col].sum()
            
            # Calculate device-specific metrics
            for device in devices:
                clicks_col = f'{device} Clicks'
                impressions_col = f'{device} Impressions'
                cost_col = f'{device} Cost'
                conversions_col = f'{device} Conversions'
                
                device_clicks = df[clicks_col].sum() if clicks_col in df.columns else 0
                device_impressions = df[impressions_col].sum() if impressions_col in df.columns else 0
                device_cost = df[cost_col].sum() if cost_col in df.columns else 0
                device_conversions = df[conversions_col].sum() if conversions_col in df.columns else 0
                
                # Calculate percentages and metrics
                click_share = (device_clicks / total_clicks * 100) if total_clicks > 0 else 0
                cost_share = (device_cost / total_cost * 100) if total_cost > 0 else 0
                conversion_share = (device_conversions / total_conversions * 100) if total_conversions > 0 else 0
                
                device_ctr = (device_clicks / device_impressions * 100) if device_impressions > 0 else 0
                device_cpc = (device_cost / device_clicks) if device_clicks > 0 else 0
                device_conv_rate = (device_conversions / device_clicks * 100) if device_clicks > 0 else 0
                device_cost_per_conv = (device_cost / device_conversions) if device_conversions > 0 else 0
                
                device_metrics[device.lower()] = {
                    'clicks': int(device_clicks),
                    'impressions': int(device_impressions),
                    'cost': float(device_cost),
                    'conversions': int(device_conversions),
                    'click_share': float(click_share),
                    'cost_share': float(cost_share),
                    'conversion_share': float(conversion_share),
                    'ctr': float(device_ctr),
                    'cpc': float(device_cpc),
                    'conversion_rate': float(device_conv_rate),
                    'cost_per_conversion': float(device_cost_per_conv)
                }
            
            # Find best performing device
            best_device = 'mobile'  # default
            best_conv_rate = 0
            for device, metrics in device_metrics.items():
                if metrics['conversion_rate'] > best_conv_rate:
                    best_conv_rate = metrics['conversion_rate']
                    best_device = device
            
            return {
                'device_metrics': device_metrics,
                'best_performing_device': best_device,
                'total_clicks': int(total_clicks),
                'total_cost': float(total_cost),
                'total_conversions': int(total_conversions)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing device performance: {str(e)}")
            return self._get_empty_device_analysis()
    
    def _generate_campaign_recommendations(self) -> List[str]:
        """Enhanced campaign recommendations"""
        recommendations = []
        try:
            budget_analysis = self._analyze_budget_efficiency()
            
            if budget_analysis.get('underutilized_months'):
                months = ', '.join(budget_analysis['underutilized_months'][:3])
                recommendations.append(f"Increase daily budget allocation in {months} to maximize opportunity")
            
            if budget_analysis.get('over_utilized_months'):
                months = ', '.join(budget_analysis['over_utilized_months'][:3])
                recommendations.append(f"Consider budget increases for {months} to avoid missed opportunities")
            
            # Device recommendations
            device_analysis = self._analyze_device_performance()
            best_device = device_analysis.get('best_performing_device', 'mobile')
            recommendations.append(f"Focus optimization efforts on {best_device} campaigns based on performance")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return ["Continue monitoring campaign performance"]
    
    def _generate_crm_recommendations(self) -> List[str]:
        """Enhanced CRM recommendations"""
        recommendations = []
        try:
            if self.crm_data is None:
                return ["No CRM data available for recommendations"]
            
            df = self.crm_data
            
            # Lead quality recommendations
            if 'Score' in df.columns:
                avg_score = df['Score'].mean()
                if avg_score < 5:
                    recommendations.append("Lead quality is critically low - review targeting and qualification criteria")
            
            # Junk rate recommendations
            if 'Contact Type' in df.columns:
                junk_rate = len(df[df['Contact Type'] == 'Junk']) / len(df) * 100
                if junk_rate > 40:
                    recommendations.append("High junk rate detected - optimize targeting to reduce wasted leads")
            
            # Google Ads specific recommendations
            if 'Lead Source' in df.columns:
                google_ads = df[df['Lead Source'] == 'Web Mail']
                if len(google_ads) > 0:
                    ga_junk_rate = len(google_ads[google_ads['Contact Type'] == 'Junk']) / len(google_ads) * 100
                    if ga_junk_rate > 50:
                        recommendations.append("Google Ads campaigns showing high junk rate - review keywords and targeting")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating CRM recommendations: {str(e)}")
            return ["Continue monitoring lead quality and sources"]
    
    # Keep existing cross-channel analysis but simplify output
    def cross_channel_analysis(self) -> Dict[str, Any]:
        """Simplified cross-channel analysis"""
        try:
            if self.monthly_campaign_data is None or self.monthly_crm_data is None:
                self.load_data()
            
            cost_per_lead = self._calculate_cost_per_lead()
            
            return {
                'data_available': len(cost_per_lead) > 0,
                'cost_per_lead_by_month': cost_per_lead,
                'average_cost_per_lead': statistics.mean(cost_per_lead.values()) if cost_per_lead else 0,
                'best_efficiency_month': min(cost_per_lead.items(), key=lambda x: x[1])[0] if cost_per_lead else 'Unknown',
                'worst_efficiency_month': max(cost_per_lead.items(), key=lambda x: x[1])[0] if cost_per_lead else 'Unknown'
            }
            
        except Exception as e:
            self.logger.error(f"Error in cross-channel analysis: {str(e)}")
            return {'data_available': False, 'error': str(e)}
    
    def _calculate_cost_per_lead(self) -> Dict[str, float]:
        """Calculate cost per lead by month"""
        try:
            cost_per_lead = {}
            
            if self.monthly_campaign_data is None or self.monthly_crm_data is None:
                return cost_per_lead
            
            # Match months between campaign and CRM data
            for _, campaign_row in self.monthly_campaign_data.iterrows():
                month = campaign_row['Month']
                cost = campaign_row.get('Cost', 0)
                
                # Find corresponding CRM data (Month vs Month Year)
                crm_match = self.monthly_crm_data[self.monthly_crm_data['Month Year'] == month]
                if not crm_match.empty:
                    leads = crm_match.iloc[0]['Total Leads']
                    cost_per_lead[month] = float(cost / leads) if leads > 0 else 0
            
            return cost_per_lead
            
        except Exception as e:
            self.logger.error(f"Error calculating cost per lead: {str(e)}")
            return {}
    
    def _analyze_budget_efficiency(self) -> Dict[str, Any]:
        """Analyze budget efficiency using Cost / Daily Budget"""
        try:
            df = self.monthly_campaign_data
            
            if df is None or len(df) == 0 or 'Cost' not in df.columns or 'Daily Budget' not in df.columns:
                return self._get_empty_budget_analysis()
            
            # Calculate utilization for each month
            df = df.copy()
            df['Budget_Util_Pct'] = (df['Cost'] / df['Daily Budget'] * 100).replace([np.inf, -np.inf], 0)
            
            # Overall utilization
            total_cost = df['Cost'].sum()
            total_budget = df['Daily Budget'].sum()
            avg_utilization = (total_cost / total_budget * 100) if total_budget > 0 else 0
            
            # Categorize months
            underutilized = df[df['Budget_Util_Pct'] < 80]['Month'].tolist()
            well_utilized = df[(df['Budget_Util_Pct'] >= 80) & (df['Budget_Util_Pct'] <= 100)]['Month'].tolist()
            over_utilized = df[df['Budget_Util_Pct'] > 100]['Month'].tolist()
            
            # Monthly efficiency
            efficiency_by_month = dict(zip(df['Month'], df['Budget_Util_Pct']))
            
            return {
                'average_budget_utilization': float(avg_utilization),
                'total_cost': float(total_cost),
                'total_daily_budget': float(total_budget),
                'efficiency_by_month': efficiency_by_month,
                'underutilized_months': underutilized,
                'well_utilized_months': well_utilized,
                'over_utilized_months': over_utilized,
                'optimal_range_months': len(well_utilized),
                'budget_recommendations': self._generate_budget_recommendations(df)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing budget efficiency: {str(e)}")
            return self._get_empty_budget_analysis()
    
    def _generate_budget_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate specific budget recommendations"""
        recommendations = []
        
        for _, row in df.iterrows():
            month = row['Month']
            util_pct = row['Budget_Util_Pct']
            
            if util_pct < 70:
                recommendations.append(f"{month}: Increase daily budget by 20-30% to capture missed opportunities")
            elif util_pct > 120:
                recommendations.append(f"{month}: Consider 15-20% daily budget increase to reduce overspend")
        
        return recommendations
    
    # Add to analyzer.py - Insert these methods into the CampaignCRMAnalyzer class

    def get_junk_lead_notes_for_analysis(self, limit: int = 200) -> Dict[str, Any]:
        """
        PRIMARY METHOD: Extract Notes data specifically for junk leads
        Returns comprehensive notes data that Claude can analyze
        """
        try:
            if self.crm_data is None:
                self.load_data()
            
            df = self.crm_data.copy()
            
            # Define junk lead criteria based on your actual data structure
            def is_junk_lead(row):
                contact_type = str(row.get('Contact Type', '')).lower()
                status = str(row.get('Status', '')).lower()
                lead_stage = str(row.get('Lead Stage', '')).lower()
                score = row.get('Score', 0)
                
                # Multiple junk indicators
                junk_indicators = [
                    'junk' in contact_type,
                    'junk' in status,
                    'junk' in lead_stage,
                    'lost' in status,
                    'cold' in status,
                    score < 0  # Negative scores
                ]
                
                return any(junk_indicators)
            
            # Filter for junk leads
            junk_leads = df[df.apply(is_junk_lead, axis=1)].copy()
            
            if len(junk_leads) == 0:
                return {
                    'data_available': False,
                    'message': 'No junk leads found in dataset',
                    'total_crm_records': len(df),
                    'suggested_filters': self._suggest_junk_filters(df)
                }
            
            # Extract key columns including Notes
            analysis_cols = [
                'Month Year', 'Campaign', 'Term', 'Contact Type', 'Lead Source',
                'Status', 'Lead Stage', 'Notes', 'Reason', 'Score', 'Locations'
            ]
            available_cols = [col for col in analysis_cols if col in junk_leads.columns]
            
            junk_data = junk_leads[available_cols].copy()
            
            # Remove rows where Notes is empty
            notes_available = junk_data[junk_data['Notes'].notna() & (junk_data['Notes'] != '')].copy()
            
            # Limit to manage token usage, but keep it substantial
            if len(notes_available) > limit:
                # Sample strategically - get diverse months and campaigns
                sampled = notes_available.groupby(['Month Year', 'Campaign'], dropna=False).apply(
                    lambda x: x.sample(min(len(x), limit // notes_available['Campaign'].nunique()), random_state=42)
                ).reset_index(drop=True)
                notes_data = sampled.head(limit)
            else:
                notes_data = notes_available
            
            # Also get aggregated statistics
            notes_summary = self._analyze_notes_patterns(notes_data)
            
            return {
                'data_available': True,
                'CRITICAL_NOTE': f'This contains ACTUAL NOTES DATA from {len(notes_data)} junk leads (out of {len(junk_leads)} total junk leads)',
                'total_junk_leads': len(junk_leads),
                'junk_leads_with_notes': len(notes_available),
                'sample_size_provided': len(notes_data),
                'junk_rate_percent': round((len(junk_leads) / len(df)) * 100, 2),
                
                # Full notes data for Claude to analyze
                'junk_lead_records_with_notes': notes_data.to_dict('records'),
                
                # Aggregated insights
                'notes_patterns': notes_summary,
                
                # Breakdown by dimensions
                'junk_by_campaign': junk_leads['Campaign'].value_counts().head(10).to_dict() if 'Campaign' in junk_leads.columns else {},
                'junk_by_term': junk_leads['Term'].value_counts().head(15).to_dict() if 'Term' in junk_leads.columns else {},
                'junk_by_source': junk_leads['Lead Source'].value_counts().to_dict() if 'Lead Source' in junk_leads.columns else {},
                'junk_by_month': junk_leads['Month Year'].value_counts().to_dict() if 'Month Year' in junk_leads.columns else {}
            }
            
        except Exception as e:
            self.logger.error(f"Error getting junk lead notes: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'data_available': False,
                'error': str(e)
            }

    def _analyze_notes_patterns(self, notes_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze patterns in Notes text to identify common reasons
        """
        try:
            if 'Notes' not in notes_df.columns or len(notes_df) == 0:
                return {'error': 'No notes data available'}
            
            notes_text = notes_df['Notes'].astype(str)
            
            # Common junk lead keywords to look for
            junk_keywords = {
                'not_interested': ['not interested', 'no interest', 'declined'],
                'wrong_product': ['office space', 'residential', 'retail space', 'not warehouse'],
                'budget_issue': ['budget', 'too expensive', 'expensive', 'cost', 'pricing'],
                'wrong_location': ['wrong location', 'different location', 'location', 'not in area'],
                'timing_issue': ['not now', 'future', 'later', 'next quarter', 'next year'],
                'decision_maker': ['not decision maker', 'need approval', 'boss', 'manager'],
                'already_found': ['already found', 'found elsewhere', 'signed elsewhere'],
                'requirement_changed': ['requirement changed', 'no longer need', 'plans changed'],
                'competitor': ['competitor', 'checking prices', 'comparison']
            }
            
            # Count occurrences
            keyword_counts = {}
            for category, keywords in junk_keywords.items():
                count = sum(notes_text.str.contains('|'.join(keywords), case=False, na=False))
                if count > 0:
                    keyword_counts[category] = count
            
            # Sort by frequency
            sorted_reasons = dict(sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True))
            
            # Get sample notes for each category
            category_examples = {}
            for category, keywords in junk_keywords.items():
                if category in sorted_reasons:
                    mask = notes_text.str.contains('|'.join(keywords), case=False, na=False)
                    examples = notes_df[mask]['Notes'].head(3).tolist()
                    category_examples[category] = examples
            
            return {
                'total_notes_analyzed': len(notes_df),
                'reason_categories_found': sorted_reasons,
                'top_3_reasons': list(sorted_reasons.keys())[:3],
                'example_notes_by_category': category_examples
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing notes patterns: {str(e)}")
            return {'error': str(e)}

    def _suggest_junk_filters(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Suggest what filters might identify junk leads"""
        suggestions = {}
        
        if 'Contact Type' in df.columns:
            suggestions['contact_types_available'] = df['Contact Type'].value_counts().head(5).to_dict()
        
        if 'Status' in df.columns:
            suggestions['statuses_available'] = df['Status'].value_counts().head(5).to_dict()
        
        if 'Lead Stage' in df.columns:
            suggestions['lead_stages_available'] = df['Lead Stage'].value_counts().head(5).to_dict()
        
        return suggestions

    def get_claude_readable_data(self, query_context: str = "") -> Dict[str, Any]:
        """
        UPDATED: Now detects query type and provides appropriate data
        """
        try:
            if self.crm_data is None:
                self.load_data()
            
            query_lower = query_context.lower()
            
            # Detect if this is a notes/junk analysis query
            is_notes_query = any(keyword in query_lower for keyword in [
                'notes', 'junk', 'why', 'reason', 'rejected', 'lost', 'cold'
            ])
            
            # Detect if this is a budget optimization query
            is_budget_query = any(keyword in query_lower for keyword in [
                'budget', 'spend', 'optimize', 'allocation', 'cost'
            ])
            
            readable_data = {
                'IMPORTANT_CONTEXT': {
                    'total_crm_records': len(self.crm_data),
                    'total_campaign_records': len(self.campaign_data) if self.campaign_data is not None else 0,
                    'query_type_detected': 'notes_analysis' if is_notes_query else 'budget_optimization' if is_budget_query else 'general',
                    'note': 'Full dataset is available. The data below is tailored to your question.'
                },
                
                'data_availability': self._check_data_availability(),
                'data_structure': self._get_data_structure_info(),
                'quick_stats': self._get_quick_stats(),
            }
            
            # Provide query-specific data
            if is_notes_query:
                readable_data['junk_lead_notes_data'] = self.get_junk_lead_notes_for_analysis(limit=200)
                readable_data['analysis_focus'] = 'Complete junk lead notes data provided for pattern analysis'
            
            elif is_budget_query:
                # Extract contact type from query if present
                contact_type = '3pl' if '3pl' in query_lower else 'direct' if 'direct' in query_lower else '3pl'
                readable_data['campaign_performance_for_budget'] = self.get_campaign_performance_for_budget_optimization(contact_type)
                readable_data['analysis_focus'] = f'Campaign performance data for {contact_type} budget optimization'
            
            else:
                # General query - provide samples
                readable_data['sample_records'] = self._get_sample_records_for_claude()
                readable_data['analysis_focus'] = 'General dataset overview'
            
            readable_data['query_context'] = query_context
            
            return readable_data
            
        except Exception as e:
            self.logger.error(f"Error getting Claude readable data: {str(e)}")
            return {'error': str(e), 'data_available': False}
    
    # Empty data structure methods (kept for compatibility)
    def _get_empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure"""
        return {
            'total_spend': 0.0, 'total_daily_budget': 0.0, 'budget_utilization': 0.0,
            'total_impressions': 0, 'total_clicks': 0, 'total_conversions': 0,
            'overall_ctr': 0.0, 'overall_conversion_rate': 0.0, 'overall_cpc': 0.0,
            'overall_cost_per_conversion': 0.0, 'active_campaigns': 0
        }
    
    def _get_empty_budget_analysis(self) -> Dict[str, Any]:
        """Return empty budget analysis structure"""
        return {
            'average_budget_utilization': 0.0, 'total_cost': 0.0, 'total_daily_budget': 0.0,
            'efficiency_by_month': {}, 'underutilized_months': [], 'well_utilized_months': [],
            'over_utilized_months': [], 'optimal_range_months': 0, 'budget_recommendations': []
        }
    
    def _get_empty_device_analysis(self) -> Dict[str, Any]:
        """Return empty device analysis structure"""
        empty_device = {'clicks': 0, 'impressions': 0, 'cost': 0.0, 'conversions': 0,
                       'click_share': 0.0, 'cost_share': 0.0, 'conversion_share': 0.0,
                       'ctr': 0.0, 'cpc': 0.0, 'conversion_rate': 0.0, 'cost_per_conversion': 0.0}
        
        return {
            'device_metrics': {'mobile': empty_device.copy(), 'desktop': empty_device.copy(), 'tablet': empty_device.copy()},
            'best_performing_device': 'mobile', 'total_clicks': 0, 'total_cost': 0.0, 'total_conversions': 0
        }