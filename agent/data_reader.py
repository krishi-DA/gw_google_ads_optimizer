# agent/data_reader.py - Fixed for pandas 2.3.2 compatibility

import pandas as pd
import numpy as np
import logging
import os
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import gspread
from google.oauth2.service_account import Credentials

class GoogleSheetsDataReader:
    """
    Fixed Google Sheets data reader for pandas 2.3.2 compatibility
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.gc = None
        self.campaign_sheet = None
        self.crm_sheet = None
        self.campaign_data = None
        self.crm_data = None
        self.monthly_campaign_data = None
        self.monthly_crm_data = None
        
        self._initialize_google_sheets_client()
    
    def _initialize_google_sheets_client(self):
        """Initialize Google Sheets client with proper error handling"""
        try:
            # Check for service account file
            service_account_file = 'service_account.json'
            if os.path.exists(service_account_file):
                scope = ['https://spreadsheets.google.com/feeds',
                        'https://www.googleapis.com/auth/drive']
                
                credentials = Credentials.from_service_account_file(
                    service_account_file, scopes=scope)
                self.gc = gspread.authorize(credentials)
                self.logger.info("Google Sheets client initialized successfully")
            else:
                self.logger.info("No service account file found - will use sample data")
                
        except Exception as e:
            self.logger.info(f"Google Sheets initialization failed, using sample data: {str(e)}")
    
    def _safe_numeric_conversion(self, series: pd.Series, column_name: str = "") -> pd.Series:
        """Safely convert series to numeric, compatible with pandas 2.3.2"""
        try:
            # For pandas 2.3.2, use errors='coerce' replacement
            numeric_series = pd.Series(index=series.index, dtype=float)
            
            for idx, val in series.items():
                try:
                    if pd.isna(val) or val == '' or val == 'nan':
                        numeric_series[idx] = 0.0
                    else:
                        # Handle string numbers
                        clean_val = str(val).replace(',', '').replace('$', '').strip()
                        numeric_series[idx] = float(clean_val)
                except (ValueError, TypeError):
                    numeric_series[idx] = 0.0
            
            return numeric_series.fillna(0)
            
        except Exception as e:
            self.logger.error(f"Error converting {column_name} to numeric: {str(e)}")
            return pd.Series([0.0] * len(series), index=series.index)
    
    def _clean_campaign_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean campaign data with safe type conversions"""
        try:
            if df is None or len(df) == 0:
                return df
            
            df = df.copy()
            
            # Numeric columns that need safe conversion
            numeric_columns = [
                'Daily Budget', 'Monthly Budget', 'Cost', 'Impression Share',
                'Impressions', 'Clicks', 'CTR %', 'CPC', 'Conversions',
                'Conversion Rate', 'Cost Per Conversion',
                'Mobile Clicks', 'Mobile Impressions', 'Mobile Cost', 'Mobile Conversions',
                'Desktop Clicks', 'Desktop Impressions', 'Desktop Cost', 'Desktop Conversions',
                'Tablet Clicks', 'Tablet Impressions', 'Tablet Cost', 'Tablet Conversions'
            ]
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = self._safe_numeric_conversion(df[col], col)
            
            # String columns
            string_columns = ['Month', 'Date', 'Campaign ID', 'Campaign Name', 'Campaign Status']
            for col in string_columns:
                if col in df.columns:
                    df[col] = df[col].astype(str).fillna('')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error cleaning campaign data: {str(e)}")
            return df
    
    def _clean_crm_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean CRM data with safe type conversions"""
        try:
            if df is None or len(df) == 0:
                return df
            
            df = df.copy()
            
            # Numeric columns
            numeric_columns = ['Score', 'Area Requirement', 'ID']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = self._safe_numeric_conversion(df[col], col)
            
            # String columns - ensure they're strings
            string_columns = [
                'Month Year', 'Formatted Time', 'Owner', 'Account Name', 'Title',
                'Full Name', 'First Name', 'Last Name', 'Email', 'Mobile', 'Description',
                'Contact Type', 'Lead Source', 'Lead Stage', 'Location', 'Business Type',
                'Industry Type', 'Status', 'Business Model', 'Reason', 'Warehouse Assigned',
                'Notes', 'Created By', 'Modified By', 'Record Creation Source ID',
                'Locations', 'Campaign', 'Term', 'Medium', 'Content'
            ]
            
            for col in string_columns:
                if col in df.columns:
                    df[col] = df[col].astype(str).fillna('')
            
            # Boolean columns
            if 'Email Opt Out' in df.columns:
                df['Email Opt Out'] = df['Email Opt Out'].astype(bool).fillna(False)
            
            # Date columns - handle with care
            date_columns = ['Created Time', 'Modified Time', 'Last Activity Time', 'Last Activity Date']
            for col in date_columns:
                if col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    except Exception:
                        self.logger.warning(f"Could not convert {col} to datetime")
                        df[col] = df[col].astype(str)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error cleaning CRM data: {str(e)}")
            return df
    
    def read_campaign_data(self, months: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """Read campaign data from Google Sheets or return cached data"""
        try:
            if self.gc is not None:
                # Try to read from actual Google Sheets
                try:
                    sheet_id = os.getenv('GOOGLE_ADS_SHEET_ID')
                    worksheet_name = os.getenv('ADS_WORKSHEET_NAME', 'Sheet1')
                    
                    if sheet_id:
                        spreadsheet = self.gc.open_by_key(sheet_id)
                        worksheet = spreadsheet.worksheet(worksheet_name)
                        data = worksheet.get_all_records()
                        
                        if data:
                            df = pd.DataFrame(data)
                            df = self._clean_campaign_data(df)
                            
                            # Filter by months if specified
                            if months and 'Month' in df.columns:
                                df = df[df['Month'].isin(months)]
                            
                            self.campaign_data = df
                            self.logger.info(f"Successfully read {len(df)} campaign records")
                            return df
                            
                except Exception as e:
                    self.logger.error(f"Failed to read from Google Sheets: {str(e)}")
            
            # Fallback to sample data
            return self._generate_sample_campaign_data(months)
            
        except Exception as e:
            self.logger.error(f"Error reading campaign data: {str(e)}")
            return self._generate_sample_campaign_data(months)
    
    def read_crm_data(self, months: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """Read CRM data from Google Sheets or return cached data"""
        try:
            if self.gc is not None:
                # Try to read from actual Google Sheets
                try:
                    sheet_id = os.getenv('CRM_SHEET_ID')
                    worksheet_name = os.getenv('CRM_WORKSHEET_NAME', 'Sheet1')
                    
                    if sheet_id:
                        spreadsheet = self.gc.open_by_key(sheet_id)
                        worksheet = spreadsheet.worksheet(worksheet_name)
                        data = worksheet.get_all_records()
                        
                        if data:
                            df = pd.DataFrame(data)
                            df = self._clean_crm_data(df)
                            
                            # Filter by months if specified
                            if months and 'Month Year' in df.columns:
                                df = df[df['Month Year'].isin(months)]
                            
                            self.crm_data = df
                            self.logger.info(f"Successfully read {len(df)} CRM records")
                            return df
                            
                except Exception as e:
                    self.logger.error(f"Failed to read CRM from Google Sheets: {str(e)}")
            
            # Fallback to sample data
            return self._generate_sample_crm_data(months)
            
        except Exception as e:
            self.logger.error(f"Error reading CRM data: {str(e)}")
            return self._generate_sample_crm_data(months)
    
    def _generate_sample_campaign_data(self, months: Optional[List[str]] = None) -> pd.DataFrame:
        """Generate realistic sample campaign data"""
        try:
            # Generate date range
            start_date = datetime(2025, 1, 1)
            end_date = datetime(2025, 9, 27)
            date_range = pd.date_range(start_date, end_date, freq='D')
            
            # Create sample data with exact column structure
            df = pd.DataFrame({
                'Month': date_range.strftime('%B %Y'),
                'Date': date_range.strftime('%m/%d/%Y'),
                'Campaign ID': np.random.choice(['20861935863', '22058660104', '22075487942', '21174729252'], len(date_range)),
                'Campaign Name': np.random.choice(['Lead-MaximiseConv', 'Brand-Awareness', 'Retargeting-Campaign'], len(date_range)),
                'Campaign Status': 'ENABLED',
                'Daily Budget': np.random.uniform(200, 1500, len(date_range)),
                'Monthly Budget': np.random.uniform(6000, 45000, len(date_range)),
                'Cost': np.random.uniform(80, 1200, len(date_range)),
                'Impression Share': np.random.uniform(60, 95, len(date_range)),
                'Impressions': np.random.uniform(1000, 15000, len(date_range)).astype(int),
                'Clicks': np.random.uniform(30, 800, len(date_range)).astype(int),
                'CTR %': np.random.uniform(1.5, 8.0, len(date_range)),
                'CPC': np.random.uniform(2, 25, len(date_range)),
                'Conversions': np.random.uniform(1, 35, len(date_range)).astype(int),
                'Conversion Rate': np.random.uniform(0.8, 6.5, len(date_range)),
                'Cost Per Conversion': np.random.uniform(15, 150, len(date_range)),
                'Mobile Clicks': np.random.uniform(15, 400, len(date_range)).astype(int),
                'Mobile Impressions': np.random.uniform(500, 8000, len(date_range)).astype(int),
                'Mobile Cost': np.random.uniform(40, 600, len(date_range)),
                'Mobile Conversions': np.random.uniform(0, 20, len(date_range)).astype(int),
                'Desktop Clicks': np.random.uniform(10, 300, len(date_range)).astype(int),
                'Desktop Impressions': np.random.uniform(300, 5000, len(date_range)).astype(int),
                'Desktop Cost': np.random.uniform(30, 450, len(date_range)),
                'Desktop Conversions': np.random.uniform(0, 15, len(date_range)).astype(int),
                'Tablet Clicks': np.random.uniform(2, 100, len(date_range)).astype(int),
                'Tablet Impressions': np.random.uniform(100, 2000, len(date_range)).astype(int),
                'Tablet Cost': np.random.uniform(10, 150, len(date_range)),
                'Tablet Conversions': np.random.uniform(0, 5, len(date_range)).astype(int)
            })
            
            # Clean the data
            df = self._clean_campaign_data(df)
            
            # Filter by months if specified
            if months and 'Month' in df.columns:
                df = df[df['Month'].isin(months)]
            
            self.campaign_data = df
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating sample campaign data: {str(e)}")
            return pd.DataFrame()
    
    def _generate_sample_crm_data(self, months: Optional[List[str]] = None) -> pd.DataFrame:
        """Generate realistic sample CRM data matching your structure"""
        try:
            # Generate realistic CRM data
            num_leads = 5669  # Match your actual record count
            start_date = datetime(2025, 1, 1)
            end_date = datetime(2025, 9, 27)
            lead_dates = pd.date_range(start_date, end_date, periods=num_leads)
            
            # Realistic distributions
            campaigns = ['Lead-MaximiseConv', 'Brand-Awareness', 'Retargeting-Campaign', '']
            campaign_weights = [0.5, 0.2, 0.15, 0.15]
            
            terms = [
                'warehouse space mumbai', 'storage facility near me', 'logistics warehouse rent',
                'cold storage mumbai', 'industrial space lease', 'warehouse for rent',
                'storage space bangalore', 'distribution center', 'fulfillment center',
                'manufacturing space', 'industrial plot', ''
            ]
            
            locations = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Pune', 'Hyderabad', 'Kolkata', 'Ahmedabad']
            location_weights = [0.3, 0.2, 0.15, 0.1, 0.08, 0.07, 0.05, 0.05]
            
            business_types = ['3PL', 'Manufacturing', 'E-commerce', 'Retail', 'FMCG', 'Pharmaceutical', 'Automotive']
            industry_types = ['Logistics', 'Manufacturing', 'Retail', 'Healthcare', 'Technology', 'Food & Beverage']
            contact_types = ['3PL', 'Direct', 'Broker', 'Junk']
            contact_weights = [0.25, 0.35, 0.15, 0.25]  # Realistic junk rate
            
            statuses = ['Hot', 'Cold', 'Converted', 'Lost']
            status_weights = [0.1, 0.4, 0.05, 0.45]
            
            # Generate notes based on contact type and status
            def generate_notes(contact_type, status, business_type):
                if contact_type == 'Junk':
                    return np.random.choice([
                        'Not interested in warehouse space',
                        'Looking for office space instead',
                        'Budget too low for requirements',
                        'Wrong location preference',
                        'Not the decision maker',
                        'Already found space elsewhere',
                        'Requirement changed',
                        'No immediate requirement'
                    ])
                elif status == 'Hot':
                    return np.random.choice([
                        'Urgent requirement for Q4',
                        'Ready to visit and finalize',
                        'Budget approved, need immediate space',
                        'Expansion plans confirmed',
                        'Director interested, scheduling visit'
                    ])
                elif status == 'Cold':
                    return np.random.choice([
                        'Will decide after board meeting',
                        'Considering multiple options',
                        'Budget approval pending',
                        'Timeline pushed to next quarter',
                        'Evaluating cost vs alternatives'
                    ])
                else:
                    return np.random.choice([
                        'Good potential lead',
                        'Following up next week',
                        'Sent location details',
                        'Discussed pricing options',
                        'Awaiting client feedback'
                    ])
            
            # Create the DataFrame with your exact column structure
            df = pd.DataFrame({
                'Month Year': lead_dates.strftime('%B %Y'),
                'Formatted Time': lead_dates.strftime('%m/%d/%Y'),
                'Owner': np.random.choice(['Rutuja Ganekar', 'Sales Team', 'Lead Manager'], num_leads),
                'Account Name': [f'Account {i}' for i in range(1, num_leads + 1)],
                'Title': np.random.choice(['Manager', 'Director', 'VP Operations', 'Owner', 'Executive'], num_leads),
                'Full Name': [f'Lead {i}' for i in range(1, num_leads + 1)],
                'First Name': [f'First{i}' for i in range(1, num_leads + 1)],
                'Last Name': [f'Last{i}' for i in range(1, num_leads + 1)],
                'Email': [f'lead{i}@company{i%100}.com' for i in range(1, num_leads + 1)],
                'Mobile': [f'+91-9{i:09d}' for i in range(num_leads)],
                'Description': 'Warehouse space requirement',
                'Contact Type': np.random.choice(contact_types, num_leads, p=contact_weights),
                'Lead Source': np.random.choice(['Web Mail', 'Organic Search', 'Referral', 'Direct'], num_leads, p=[0.6, 0.2, 0.1, 0.1]),
                'Lead Stage': np.random.choice(['New', 'Qualified', 'Contacted', 'Junk'], num_leads, p=[0.3, 0.25, 0.25, 0.2]),
                'Location': np.random.choice(locations, num_leads, p=location_weights),
                'Business Type': np.random.choice(business_types, num_leads),
                'Industry Type': np.random.choice(industry_types, num_leads),
                'Status': np.random.choice(statuses, num_leads, p=status_weights),
                'Area Requirement': np.random.uniform(500, 50000, num_leads),
                'Business Model': np.random.choice(['B2B', 'B2C', 'B2B2C'], num_leads),
                'Reason': '',
                'Warehouse Assigned': np.random.choice(['Warehouse A', 'Warehouse B', 'Warehouse C', ''], num_leads, p=[0.2, 0.2, 0.2, 0.4]),
                'Notes': '',  # Will be filled below
                'ID': range(1, num_leads + 1),
                'Created By': 'System',
                'Created Time': lead_dates,
                'Modified By': 'System',
                'Modified Time': lead_dates,
                'Last Activity Time': lead_dates,
                'Email Opt Out': False,
                'Record Creation Source ID': 'WEB_FORM',
                'Locations': np.random.choice(locations, num_leads, p=location_weights),
                'Last Activity Date': lead_dates.strftime('%m/%d/%Y'),
                'Campaign': np.random.choice(campaigns, num_leads, p=campaign_weights),
                'Term': np.random.choice(terms, num_leads),
                'Medium': 'cpc',
                'Content': 'ad_variant_1',
                'Score': np.random.uniform(0, 100, num_leads)
            })
            
            # Generate realistic notes
            notes_list = []
            for i in range(num_leads):
                contact_type = df.iloc[i]['Contact Type']
                status = df.iloc[i]['Status']
                business_type = df.iloc[i]['Business Type']
                notes_list.append(generate_notes(contact_type, status, business_type))
            
            df['Notes'] = notes_list
            
            # Clean the data
            df = self._clean_crm_data(df)
            
            # Filter by months if specified
            if months and 'Month Year' in df.columns:
                df = df[df['Month Year'].isin(months)]
            
            self.crm_data = df
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating sample CRM data: {str(e)}")
            return pd.DataFrame()
    
    def get_monthly_aggregated_campaign_data(self) -> Optional[pd.DataFrame]:
        """Get monthly aggregated campaign data"""
        try:
            if self.campaign_data is None:
                self.read_campaign_data()
            
            if self.campaign_data is None or len(self.campaign_data) == 0:
                return None
            
            # Aggregate by month
            monthly_agg = self.campaign_data.groupby('Month').agg({
                'Cost': 'sum',
                'Daily Budget': 'sum',
                'Impressions': 'sum',
                'Clicks': 'sum',
                'Conversions': 'sum',
                'CTR %': 'mean',
                'CPC': 'mean',
                'Conversion Rate': 'mean',
                'Mobile Clicks': 'sum',
                'Mobile Cost': 'sum',
                'Mobile Conversions': 'sum',
                'Desktop Clicks': 'sum',
                'Desktop Cost': 'sum',
                'Desktop Conversions': 'sum',
                'Tablet Clicks': 'sum',
                'Tablet Cost': 'sum',
                'Tablet Conversions': 'sum'
            }).reset_index()
            
            # Calculate budget utilization safely
            monthly_agg['Budget Utilization'] = np.where(
                monthly_agg['Daily Budget'] > 0,
                monthly_agg['Cost'] / monthly_agg['Daily Budget'],
                0
            )
            
            self.monthly_campaign_data = monthly_agg
            return monthly_agg
            
        except Exception as e:
            self.logger.error(f"Error aggregating campaign data: {str(e)}")
            return None
    
    def get_monthly_aggregated_crm_data(self) -> Optional[pd.DataFrame]:
        """Get monthly aggregated CRM data"""
        try:
            if self.crm_data is None:
                self.read_crm_data()
            
            if self.crm_data is None or len(self.crm_data) == 0:
                return None
            
            # Aggregate by month
            monthly_agg = self.crm_data.groupby('Month Year').agg({
                'ID': 'count',
                'Score': 'mean'
            }).reset_index()
            
            monthly_agg.columns = ['Month Year', 'Total Leads', 'Average Score']
            
            self.monthly_crm_data = monthly_agg
            return monthly_agg
            
        except Exception as e:
            self.logger.error(f"Error aggregating CRM data: {str(e)}")
            return None
    
    def get_available_months(self) -> Dict[str, List[str]]:
        """Get available months from both datasets"""
        try:
            campaign_months = []
            crm_months = []
            
            # Get campaign months
            if self.campaign_data is None:
                self.read_campaign_data()
            
            if self.campaign_data is not None and 'Month' in self.campaign_data.columns:
                campaign_months = sorted(self.campaign_data['Month'].unique())
            
            # Get CRM months
            if self.crm_data is None:
                self.read_crm_data()
            
            if self.crm_data is not None and 'Month Year' in self.crm_data.columns:
                crm_months = sorted(self.crm_data['Month Year'].unique())
            
            return {
                'campaign_months': campaign_months,
                'crm_months': crm_months
            }
            
        except Exception as e:
            self.logger.error(f"Error getting available months: {str(e)}")
            return {'campaign_months': [], 'crm_months': []}
    
    def validate_data_structure(self) -> Dict[str, Any]:
        """Validate data structure and quality"""
        validation = {
            'campaign_data': {'valid': False, 'issues': []},
            'crm_data': {'valid': False, 'issues': []},
            'overall_quality': 'Poor'
        }
        
        try:
            # Validate campaign data
            if self.campaign_data is not None:
                required_cols = ['Month', 'Campaign ID', 'Cost', 'Daily Budget', 'Impressions', 'Clicks', 'Conversions']
                missing_cols = [col for col in required_cols if col not in self.campaign_data.columns]
                
                if not missing_cols:
                    validation['campaign_data']['valid'] = True
                else:
                    validation['campaign_data']['issues'].append(f"Missing columns: {missing_cols}")
            else:
                validation['campaign_data']['issues'].append("No campaign data loaded")
            
            # Validate CRM data
            if self.crm_data is not None:
                required_cols = ['Month Year', 'ID', 'Email', 'Lead Source', 'Contact Type']
                missing_cols = [col for col in required_cols if col not in self.crm_data.columns]
                
                if not missing_cols:
                    validation['crm_data']['valid'] = True
                else:
                    validation['crm_data']['issues'].append(f"Missing columns: {missing_cols}")
            else:
                validation['crm_data']['issues'].append("No CRM data loaded")
            
            # Overall quality assessment
            if validation['campaign_data']['valid'] and validation['crm_data']['valid']:
                validation['overall_quality'] = 'Good'
            elif validation['campaign_data']['valid'] or validation['crm_data']['valid']:
                validation['overall_quality'] = 'Fair'
            
            return validation
            
        except Exception as e:
            self.logger.error(f"Error validating data structure: {str(e)}")
            validation['campaign_data']['issues'].append(f"Validation error: {str(e)}")
            validation['crm_data']['issues'].append(f"Validation error: {str(e)}")
            return validation