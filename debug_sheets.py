#!/usr/bin/env python3
"""
Debug script to check Google Sheets access and worksheet names
Run this to diagnose sheet access issues: python debug_sheets.py
"""

import gspread
from google.oauth2.service_account import Credentials
import sys

def debug_google_sheets():
    """Debug Google Sheets connection and list available worksheets"""
    
    # Sheet IDs from your playbook
    CAMPAIGN_SHEET_ID = "1T3SINF2MOrHmpOx00yxYVGb7-ScSTjBvi1VVh0t_6cg"
    CRM_SHEET_ID = "1Mtpi0rNRourJbg8VFnWP933nirrFvv0jo9Wk9qh3OCU"
    
    try:
        print("🔧 Debugging Google Sheets Access...")
        print("=" * 50)
        
        # Connect to Google Sheets
        gc = gspread.service_account(filename='service_account.json')
        print("✅ Successfully connected to Google Sheets")
        
        # Check Campaign Data Sheet
        print(f"\n📊 CAMPAIGN DATA SHEET")
        print(f"Sheet ID: {CAMPAIGN_SHEET_ID}")
        
        try:
            campaign_sheet = gc.open_by_key(CAMPAIGN_SHEET_ID)
            print(f"✅ Successfully opened campaign sheet: '{campaign_sheet.title}'")
            
            print("\n📋 Available Worksheets:")
            for i, worksheet in enumerate(campaign_sheet.worksheets(), 1):
                print(f"  {i}. '{worksheet.title}' ({worksheet.row_count} rows, {worksheet.col_count} cols)")
                
                # Try to read first few rows to check access
                try:
                    sample_data = worksheet.get_all_values()[:3]  # First 3 rows
                    print(f"     ✅ Can read data - Sample: {len(sample_data)} rows")
                    if sample_data and len(sample_data) > 0:
                        print(f"     📝 Headers: {sample_data[0][:5]}...")  # First 5 columns
                except Exception as e:
                    print(f"     ❌ Cannot read data: {str(e)}")
            
        except Exception as e:
            print(f"❌ Cannot access campaign sheet: {str(e)}")
            print("💡 Make sure you've shared the sheet with your service account email")
        
        # Check CRM Data Sheet
        print(f"\n📊 CRM DATA SHEET")
        print(f"Sheet ID: {CRM_SHEET_ID}")
        
        try:
            crm_sheet = gc.open_by_key(CRM_SHEET_ID)
            print(f"✅ Successfully opened CRM sheet: '{crm_sheet.title}'")
            
            print("\n📋 Available Worksheets:")
            for i, worksheet in enumerate(crm_sheet.worksheets(), 1):
                print(f"  {i}. '{worksheet.title}' ({worksheet.row_count} rows, {worksheet.col_count} cols)")
                
                # Try to read first few rows
                try:
                    sample_data = worksheet.get_all_values()[:3]  # First 3 rows
                    print(f"     ✅ Can read data - Sample: {len(sample_data)} rows")
                    if sample_data and len(sample_data) > 0:
                        print(f"     📝 Headers: {sample_data[0][:5]}...")  # First 5 columns
                except Exception as e:
                    print(f"     ❌ Cannot read data: {str(e)}")
            
        except Exception as e:
            print(f"❌ Cannot access CRM sheet: {str(e)}")
            print("💡 Make sure you've shared the sheet with your service account email")
        
        # Get service account email for sharing instructions
        try:
            # Read service account file to get email
            import json
            with open('service_account.json', 'r') as f:
                sa_data = json.load(f)
                sa_email = sa_data.get('client_email', 'Unknown')
                print(f"\n📧 Service Account Email: {sa_email}")
                print("💡 Make sure BOTH sheets are shared with this email address!")
        except Exception as e:
            print(f"\n❌ Cannot read service account email: {str(e)}")
        
        print("\n" + "=" * 50)
        print("🎯 RECOMMENDED ACTIONS:")
        print("1. Share both Google Sheets with your service account email")
        print("2. Give 'Editor' or 'Viewer' permissions")
        print("3. Update worksheet names in the app if they don't match exactly")
        print("4. Add your Anthropic API key to .env file")
        
    except Exception as e:
        print(f"💥 Failed to connect to Google Sheets: {str(e)}")
        print("\n🔧 TROUBLESHOOTING STEPS:")
        print("1. Make sure 'service_account.json' file exists in project root")
        print("2. Check if the service account has proper permissions")
        print("3. Verify Google Sheets API is enabled in your Google Cloud project")
        return False
    
    return True

if __name__ == "__main__":
    success = debug_google_sheets()
    if success:
        print("\n✅ Debug completed! Check the output above for next steps.")
    else:
        print("\n❌ Debug failed! Please fix the connection issues first.")
        sys.exit(1)