import sys
import os
from agent import data_reader
from agent import tools
sys.path.append('agent')

def quick_debug():
    print("QUICK DEBUG - Finding Your Data Issue")
    print("="*60)
    
    try:
        print("Step 1: Testing data loading...")
        from agent.data_reader import load_and_preprocess_data
        
        campaign_df, crm_df = load_and_preprocess_data()
        
        print(f"Campaign data: {len(campaign_df)} rows")
        print(f"CRM data: {len(crm_df)} rows")
        
        if crm_df.empty:
            print("PROBLEM: CRM data is empty!")
            return
        
        print("\nStep 2: Checking Lead Sources...")
        if 'Lead Source' in crm_df.columns:
            sources = crm_df['Lead Source'].value_counts()
            print("Lead Sources found:")
            for source, count in sources.head(10).items():
                print(f"   {source}: {count}")
                
            web_mail_count = (crm_df['Lead Source'] == 'Web Mail').sum()
            print(f"\nExact 'Web Mail': {web_mail_count}")
            
            if web_mail_count == 0:
                print("PROBLEM: No 'Web Mail' leads found!")
                return
                
        else:
            print("PROBLEM: No 'Lead Source' column found!")
            print("Available columns:")
            for col in crm_df.columns:
                print(f"   - {col}")
            return
        
        print("\nStep 3: Checking Lead Quality...")
        if 'Lead Quality' in crm_df.columns:
            quality = crm_df['Lead Quality'].value_counts()
            for q, count in quality.items():
                print(f"   {q}: {count}")
            
            threpl_count = (crm_df['Lead Quality'] == '3PL Lead').sum()
            if threpl_count == 0:
                print("PROBLEM: No 3PL leads found!")
                return
            print(f"SUCCESS: Found {threpl_count} 3PL leads")
        
        print("\nStep 4: Testing final pipeline...")
        from agent.tools import get_campaign_performance_data, get_campaign_summary_stats        
        performance_df = get_campaign_performance_data(campaign_df, crm_df)
        if performance_df.empty:
            print("PROBLEM: Performance analysis failed")
            return
        
        summary_stats = get_campaign_summary_stats(performance_df)
        
        print("\nFINAL RESULTS:")
        print(f"   Total 3PL Leads: {summary_stats.get('total_3pl_leads', 0)}")
        print(f"   Total Cost: Rs{summary_stats.get('total_cost', 0):,.0f}")
        
        if summary_stats.get('total_3pl_leads', 0) > 0:
            print("SUCCESS! Pipeline working!")
        else:
            print("PROBLEM: No 3PL leads in final results")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    quick_debug()