# app.py - Enhanced with Learning System and Fixed Claude Data Access

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import os
import sys
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add agent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'agent'))

# Import modules with error handling
try:
    from agent.data_reader import GoogleSheetsDataReader
    from agent.analyzer import CampaignCRMAnalyzer
    from agent.response_gen import ResponseGenerator
    from agent.learner import EnhancedLearningSystem  # NEW: Learning system
    from agent.tools import (
        AnthropicClient, DataValidator, MetricsCalculator, 
        VisualizationHelper, CacheManager, ConfigManager, 
        Logger, BusinessContext, SmartContextBuilder,
        format_currency, format_percentage, calculate_budget_utilization, 
        get_utilization_status, detect_lead_quality_crisis,
        extract_campaign_name_from_question, get_performance_grade,
        sort_months_chronologically
    )
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info("Using fallback implementations")
    
    # [Fallback implementations remain the same as your original code]
    # ... (keeping all your fallback classes)

class GoogleAdsAnalysisApp:
    """
    Enhanced Streamlit application with integrated learning system
    FIXED: Proper data flow to Claude for filtered queries
    """
    
    def __init__(self):
        self.setup_logging()
        self.setup_streamlit_config()
        self.initialize_components()
        self.init_session_state()
    
    def setup_logging(self):
        """Setup application logging"""
        try:
            Logger.setup_logging(
                log_level=os.getenv('LOG_LEVEL', 'INFO'),
                log_file='data/app.log'
            )
            self.logger = logging.getLogger(__name__)
            self.logger.info("Google Ads Analysis Application started")
        except Exception as e:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
            self.logger.warning(f"Logging setup issue: {e}")
    
    def setup_streamlit_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="Smart Google Ads & CRM Analyzer",
            page_icon="🎯",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': None,
                'Report a bug': None,
                'About': "Smart Google Ads Campaign & CRM Analysis System v2.0 - Powered by Claude with Learning System"
            }
        )
    
    def initialize_components(self):
        """Initialize system components with error handling"""
        try:
            self.config = ConfigManager()
            self.data_reader = GoogleSheetsDataReader()
            self.analyzer = CampaignCRMAnalyzer()
            self.response_generator = ResponseGenerator()
            self.anthropic_client = AnthropicClient()
            self.validator = DataValidator()
            self.viz_helper = VisualizationHelper()
            self.cache_manager = CacheManager()
            self.business_context = BusinessContext()
            self.context_builder = SmartContextBuilder()
            
            # NEW: Initialize learning system
            self.learner = EnhancedLearningSystem()
            self.logger.info("Learning system initialized successfully")
            
            self.logger.info("All components initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing components: {str(e)}")
            st.error(f"System initialization failed: {str(e)}")
    
    def init_session_state(self):
        """Initialize session state"""
        defaults = {
            'analysis_cache': {},
            'conversation_history': [],
            'selected_months': [],
            'data_validation_results': {},
            'last_refresh': datetime.now(),
            'system_alerts': [],
            'pending_ratings': {},  # NEW: Track conversations pending rating
            'show_quality_report': False  # NEW: Toggle for quality report
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    def run(self):
        """Main application runner"""
        # Enhanced CSS with rating styles
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #4285f4;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: bold;
        }
        .critical-alert {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            border-left: 4px solid #dc3545;
        }
        .insight-highlight {
            background: #d1ecf1;
            border: 1px solid #bee5eb;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            border-left: 4px solid #17a2b8;
        }
        .rating-box {
            background: #f8f9fa;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
        .quality-metric {
            background: #e7f3ff;
            border-left: 4px solid #2196F3;
            padding: 0.5rem;
            margin: 0.5rem 0;
            border-radius: 4px;
        }
        .performance-good {
            color: #28a745;
            font-weight: bold;
        }
        .performance-warning {
            color: #ffc107;
            font-weight: bold;
        }
        .performance-critical {
            color: #dc3545;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.markdown('<h1 class="main-header">🧠 Smart Google Ads & CRM Analyzer</h1>', 
                   unsafe_allow_html=True)
        st.markdown("*Powered by Claude AI with Continuous Learning System*")
        
        # System status
        self.show_enhanced_system_status()
        
        # Sidebar
        self.render_enhanced_sidebar()
        
        # Main content tabs
        dashboard_tab, analysis_tab, claude_chat_tab, insights_tab, learning_tab = st.tabs([
            "📊 Smart Dashboard", "🔍 Deep Analysis", "🧠 Claude AI Chat", "💡 AI Insights", "🎓 Learning System"
        ])
        
        with dashboard_tab:
            self.render_smart_dashboard()
        
        with analysis_tab:
            self.render_enhanced_deep_analysis()
        
        with claude_chat_tab:
            self.render_enhanced_chat_interface()
        
        with insights_tab:
            self.render_ai_insights_interface()
        
        with learning_tab:
            self.render_learning_dashboard()  # NEW: Learning system dashboard
    
    def show_enhanced_system_status(self):
        """Show system status with data quality alerts"""
        status_col1, status_col2, status_col3, status_col4, status_col5 = st.columns(5)
        
        with status_col1:
            try:
                available_months = self.data_reader.get_available_months()
                campaign_months = len(available_months.get('campaign_months', []))
                if campaign_months > 0:
                    st.success(f"✅ Campaign Data: {campaign_months} months")
                else:
                    st.error("❌ Campaign Data: Not connected")
            except Exception as e:
                st.error(f"❌ Campaign Data: {str(e)}")
        
        with status_col2:
            try:
                available_months = self.data_reader.get_available_months()
                crm_months = len(available_months.get('crm_months', []))
                if crm_months > 0:
                    st.success(f"✅ CRM Data: {crm_months} months")
                    if hasattr(self.analyzer, 'crm_data') and self.analyzer.crm_data is not None:
                        record_count = len(self.analyzer.crm_data)
                        st.info(f"📊 {record_count:,} lead records")
                else:
                    st.error("❌ CRM Data: Not connected")
            except Exception as e:
                st.error(f"❌ CRM Data: {str(e)}")
        
        with status_col3:
            try:
                analysis_data = self.get_comprehensive_analysis()
                quality_data = analysis_data.get('crm_analysis', {}).get('lead_quality_patterns', {})
                quality_pct = quality_data.get('quality_distribution', {}).get('quality_percentage', 0)
                
                quality_status = detect_lead_quality_crisis(quality_pct)
                if quality_status == "CRITICAL CRISIS":
                    st.error(f"🚨 Quality: {quality_pct:.2f}% - CRISIS")
                elif "NEEDS" in quality_status:
                    st.warning(f"⚠️ Quality: {quality_pct:.1f}% - NEEDS WORK")
                else:
                    st.success(f"✅ Quality: {quality_pct:.1f}%")
            except Exception:
                st.info("📊 Quality: Analyzing...")
        
        with status_col4:
            try:
                if os.getenv('ANTHROPIC_API_KEY'):
                    st.success("🧠 Claude AI: Ready")
                else:
                    st.error("❌ Claude AI: No API key")
                    st.info("Add ANTHROPIC_API_KEY to .env file")
            except Exception:
                st.warning("⚠️ Claude AI: Check config")
        
        # NEW: Learning system status
        with status_col5:
            try:
                quality_report = self.learner.get_quality_report()
                rated_count = quality_report.get('rated_conversations', 0)
                avg_rating = quality_report.get('average_rating', 0)
                
                if rated_count > 0:
                    st.success(f"🎓 Learning: {rated_count} rated")
                    st.info(f"⭐ Avg: {avg_rating:.1f}/10")
                else:
                    st.info("🎓 Learning: No ratings yet")
            except Exception:
                st.info("🎓 Learning: Active")
        
        self.show_critical_alerts()
    
    def show_critical_alerts(self):
        """Show critical system alerts"""
        try:
            analysis_data = self.get_comprehensive_analysis()
            
            anomalies = []
            campaign_anomalies = analysis_data.get('campaign_analysis', {}).get('data_anomalies', [])
            anomalies.extend(campaign_anomalies)
            
            quality_data = analysis_data.get('crm_analysis', {}).get('lead_quality_patterns', {})
            quality_dist = quality_data.get('quality_distribution', {})
            
            if quality_dist:
                quality_pct = quality_dist.get('quality_percentage', 0)
                junk_count = quality_dist.get('junk_count', 0)
                total_leads = quality_dist.get('total_leads', 1)
                junk_pct = junk_count / total_leads * 100 if total_leads > 0 else 0
                
                if quality_pct < 1:
                    anomalies.append(f"CRITICAL: Lead quality crisis - only {quality_pct:.2f}% high quality leads")
                
                if junk_pct > 60:
                    anomalies.append(f"CRITICAL: {junk_pct:.1f}% junk rate - lead generation failing")
            
            critical_alerts = [a for a in anomalies if 'CRITICAL' in a]
            if critical_alerts:
                st.markdown("### 🚨 Critical Alerts")
                for alert in critical_alerts[:2]:
                    st.markdown(f'<div class="critical-alert"><strong>{alert}</strong></div>', 
                               unsafe_allow_html=True)
        
        except Exception as e:
            self.logger.error(f"Error showing critical alerts: {str(e)}")
    
    def render_enhanced_sidebar(self):
        """Render enhanced sidebar with learning metrics"""
        st.sidebar.header("🎛️ Smart Analysis Controls")
        
        # NEW: Quick quality metrics
        with st.sidebar.expander("🎓 Learning Quality Metrics", expanded=False):
            quality_report = self.learner.get_quality_report()
            
            if quality_report.get('rated_conversations', 0) > 0:
                st.metric("Average Rating", f"{quality_report['average_rating']:.1f}/10")
                st.metric("Rated Responses", quality_report['rated_conversations'])
                st.metric("High Quality (8+)", quality_report['high_quality_count'])
                
                if st.button("📊 View Full Report"):
                    st.session_state.show_quality_report = True
            else:
                st.info("Rate responses to see quality metrics")
        
        # Data connection status
        self.show_detailed_connection_status()
        
        # Load sample data
        if st.sidebar.button("🔄 Load Sample Data", type="primary"):
            with st.spinner("Loading sample data..."):
                try:
                    self.data_reader.load_sample_data()
                    st.sidebar.success("✅ Sample data loaded!")
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Error loading data: {str(e)}")
        
        # Month selection
        available_months = self.get_available_months()
        if available_months.get('campaign_months'):
            st.sidebar.subheader("📅 Analysis Period")
            
            col1, col2 = st.sidebar.columns(2)
            with col1:
                if st.button("📊 Recent (3M)", key="recent_3m"):
                    st.session_state.selected_months = available_months['campaign_months'][-3:]
            with col2:
                if st.button("📈 All Data", key="all_data_btn"):
                    st.session_state.selected_months = available_months['campaign_months']
            
            selected_months = st.sidebar.multiselect(
                "Custom Selection",
                options=available_months['campaign_months'],
                default=st.session_state.selected_months or available_months['campaign_months'][-3:],
                help="Select specific months for analysis"
            )
            st.session_state.selected_months = selected_months
        
        # Actions
        st.sidebar.subheader("🚀 Actions")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("🔄 Refresh", type="primary"):
                self.refresh_all_data()
        with col2:
            if st.button("🧹 Clear Cache"):
                self.clear_all_caches()
    
    def render_enhanced_chat_interface(self):
        """Render enhanced chat interface with rating mechanism"""
        st.header("🧠 Claude AI Analysis Assistant")
        st.markdown("*Ask Claude anything about your Google Ads and CRM data*")
        
        # Smart suggestions
        self.show_smart_question_suggestions()
        
        # Chat input
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_question = st.text_input(
                "Ask Claude about your data:",
                placeholder="e.g., 'List keywords creating junk leads for Contact Type = 3pl, month-by-month, campaign-wise'",
                key="enhanced_chat_input"
            )
        
        with col2:
            analyze_button = st.button("🧠 Ask Claude", type="primary")
        
        # Quick action buttons
        quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)
        
        with quick_col1:
            if st.button("🚨 What's my biggest problem?"):
                user_question = "What's my biggest problem right now based on the data?"
                analyze_button = True
        
        with quick_col2:
            if st.button("💡 Give me optimization ideas"):
                user_question = "Give me specific optimization recommendations based on my data"
                analyze_button = True
        
        with quick_col3:
            if st.button("🎯 Analyze campaign performance"):
                user_question = "Analyze my campaign performance and tell me what needs immediate attention"
                analyze_button = True
        
        with quick_col4:
            if st.button("📝 Why so many junk leads?"):
                user_question = "Can you analyze the notes column from the CRM sheets and find out why there are so many junk leads?"
                analyze_button = True
        # Process question
        if (analyze_button and user_question.strip()) or st.session_state.get('quick_question'):
            if st.session_state.get('quick_question'):
                user_question = st.session_state.quick_question
                del st.session_state.quick_question
            
            self.process_enhanced_ai_question(user_question)
        
        # Display conversation history with rating interface
        if st.session_state.conversation_history:
            st.subheader("💭 Recent AI Conversations")
            
            for i, conversation in enumerate(reversed(st.session_state.conversation_history[-5:])):
                with st.expander(
                    f"Q: {conversation['question'][:60]}..." if len(conversation['question']) > 60 else f"Q: {conversation['question']}", 
                    expanded=(i==0)
                ):
                    # Question
                    st.markdown(f"**Your Question:** {conversation['question']}")
                    
                    # Claude's Response
                    st.markdown(f"**Claude's Analysis:** {conversation['answer']}")
                    
                    # Context used
                    if conversation.get('context_type'):
                        st.markdown(f"*Analysis Type: {conversation['context_type']}*")
                    
                    # Timestamp
                    st.markdown(f"*{conversation['timestamp']}*")
                    
                    # NEW: Rating Interface
                    if conversation.get('conv_id'):
                        self.render_rating_interface(conversation)
    
    def render_rating_interface(self, conversation: Dict[str, Any]):
        """Render rating interface for a conversation"""
        conv_id = conversation['conv_id']
        
        # Check if already rated
        is_rated = any(
            c.id == conv_id and c.rating is not None 
            for c in self.learner.qa_conversations
        )
        
        if is_rated:
            # Show existing rating
            rated_conv = next(c for c in self.learner.qa_conversations if c.id == conv_id)
            st.success(f"✅ Rated: {rated_conv.rating}/10")
            if rated_conv.feedback_notes:
                st.info(f"Feedback: {rated_conv.feedback_notes}")
        else:
            # Show rating form
            st.markdown('<div class="rating-box">', unsafe_allow_html=True)
            st.markdown("#### ⭐ Rate this response to help improve future answers")
            
            rating_col1, rating_col2 = st.columns([3, 1])
            
            with rating_col1:
                rating = st.slider(
                    "Quality Rating",
                    min_value=1,
                    max_value=10,
                    value=7,
                    help="1=Poor, 5=Okay, 8=Great, 10=Perfect",
                    key=f"rating_slider_{conv_id}"
                )
                
                feedback_notes = st.text_area(
                    "Optional: What was good or bad about this response?",
                    placeholder="E.g., 'Very specific and actionable' or 'Too generic, missing data citations'",
                    key=f"feedback_text_{conv_id}",
                    height=80
                )
            
            with rating_col2:
                st.markdown("**Guide:**")
                st.markdown("10: Perfect")
                st.markdown("8-9: Great")
                st.markdown("6-7: Good")
                st.markdown("4-5: Okay")
                st.markdown("1-3: Poor")
            
            if st.button("💾 Submit Rating", key=f"submit_rating_{conv_id}", type="primary"):
                success = self.learner.update_qa_rating(
                    conv_id,
                    rating,
                    feedback_notes
                )
                
                if success:
                    st.success(f"✅ Thank you! Rating saved: {rating}/10")
                    st.balloons()
                    
                    # Show improvement suggestions if rating is low
                    if rating <= 5:
                        st.info("💡 Low ratings help us improve! Your feedback is valuable.")
                    
                    st.rerun()
                else:
                    st.error("Failed to save rating. Please try again.")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # FIXED: Key method that handles Claude data access
    def process_enhanced_ai_question(self, question: str):
        """
        FIXED: Process user question with enhanced Claude integration and proper data flow
        This is the critical method that ensures filtered data reaches Claude
        """
        with st.spinner("🧠 Claude is analyzing your data and generating insights..."):
            try:
                # Classify question type for learning
                question_type = self.learner.classify_question_type(question)
                
                # NEW: Detect junk notes queries
                is_junk_notes_query = any(keyword in question.lower() for keyword in [
                    'notes', 'why junk', 'junk reason', 'rejected', 'analyze notes',
                    'notes column', 'why are there', 'why so many junk'
                ])
                
                # NEW: If junk notes query, load junk notes data first
                if is_junk_notes_query:
                    with st.spinner("📝 Extracting junk lead notes..."):
                        junk_notes_data = self.analyzer.get_junk_lead_notes_for_analysis(limit=200)
                        
                        if junk_notes_data.get('data_available'):
                            st.info(f"✅ Found {junk_notes_data.get('junk_leads_with_notes', 0)} junk leads with notes")
                        else:
                            st.warning("⚠️ No junk notes data found. Proceeding with general analysis.")
                
                # Get few-shot learning context from high-quality past responses
                few_shot_context = self.learner.build_few_shot_context(question, question_type)
                
                # CRITICAL FIX: Use the new Claude context generator
                claude_context = self.response_generator.generate_claude_context(question)
                
                # NEW: Add junk notes data to context if available
                if is_junk_notes_query and junk_notes_data.get('data_available'):
                    claude_context['context_type'] = 'junk_notes_analysis'
                    claude_context['junk_notes_data'] = junk_notes_data
                
                # Check if we got an error or no data
                if claude_context.get('context_type') == 'error':
                    st.error(f"Data analysis error: {claude_context.get('error_message', 'Unknown error')}")
                    return
                
                # Create prompt based on context type
                if claude_context.get('context_type') == 'junk_notes_analysis':
                    # NEW: Junk notes specific query
                    focused_prompt = self.create_junk_notes_prompt(question, claude_context, few_shot_context)
                elif claude_context.get('context_type') == 'filtered_data_analysis':
                    # Filtered data query - use actual data table
                    focused_prompt = self.create_filtered_data_prompt(question, claude_context, few_shot_context)
                elif claude_context.get('context_type') == 'junk_keywords_analysis':
                    # Junk keywords specific query
                    focused_prompt = self.create_junk_keywords_prompt(question, claude_context, few_shot_context)
                else:
                    # Enhanced analysis query
                    focused_prompt = self.create_enhanced_analysis_prompt(question, claude_context, few_shot_context)
                
                # Get Claude's response with proper context
                claude_response = self.anthropic_client.generate_response(
                    focused_prompt,
                    context=claude_context
                )
                
                # Determine data scope for learning
                data_scope = self.get_data_scope_from_claude_context(claude_context)
                
                # Record interaction in learning system
                conv_id = self.learner.record_qa_interaction(
                    question=question,
                    response=claude_response,
                    question_type=question_type,
                    data_scope=data_scope
                )
                
                # Store conversation with conv_id
                conversation = {
                    'question': question,
                    'answer': claude_response,
                    'context_type': claude_context.get('context_type', 'general'),
                    'data_available': claude_context.get('data_available', False),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'conv_id': conv_id,
                    'question_type': question_type
                }
                
                st.session_state.conversation_history.append(conversation)
                
                # Display response
                st.success("🧠 Claude's Analysis Complete!")
                
                st.markdown("### Claude's Response:")
                st.markdown(claude_response)
                
                # Show context type information
                context_type = claude_context.get('context_type', 'general')
                if context_type == 'junk_notes_analysis':
                    # NEW: Show junk notes analysis info
                    junk_notes = claude_context.get('junk_notes_data', {})
                    st.info(f"📝 Junk Notes Analysis: {junk_notes.get('total_junk_leads', 0)} junk leads, {junk_notes.get('junk_leads_with_notes', 0)} with notes")
                elif context_type == 'filtered_data_analysis':
                    data_summary = claude_context.get('data_summary', {})
                    st.info(f"📊 Filtered data analysis: {data_summary.get('total_rows', 0)} data rows analyzed")
                elif context_type == 'junk_keywords_analysis':
                    junk_summary = claude_context.get('junk_summary', {})
                    st.info(f"🚨 Junk analysis: {junk_summary.get('total_junk_leads', 0)} junk leads analyzed")
                elif context_type == 'enhanced_analysis':
                    st.info("🧠 Enhanced analysis using comprehensive data context")
                
                # Show if few-shot learning was used
                if few_shot_context:
                    st.info("🎓 This response was enhanced using similar high-quality examples from the learning system")
                
                # Show context transparency
                with st.expander("🔍 Analysis Context (What Claude Analyzed)"):
                    st.write(f"**Question Type:** {question_type}")
                    st.write(f"**Context Type:** {context_type}")
                    st.write(f"**Data Available:** {claude_context.get('data_available', False)}")
                    st.write(f"**Data Scope:** {data_scope}")
                    
                    # NEW: Show junk notes context details
                    if context_type == 'junk_notes_analysis':
                        junk_notes = claude_context.get('junk_notes_data', {})
                        st.write(f"**Total Junk Leads:** {junk_notes.get('total_junk_leads', 0)}")
                        st.write(f"**Leads with Notes:** {junk_notes.get('junk_leads_with_notes', 0)}")
                        st.write(f"**Sample Provided:** {junk_notes.get('sample_size_provided', 0)}")
                        st.write(f"**Junk Rate:** {junk_notes.get('junk_rate_percent', 0):.1f}%")
                        
                        # Show pattern categories if available
                        patterns = junk_notes.get('notes_patterns', {})
                        if patterns.get('reason_categories_found'):
                            st.write(f"**Reason Categories:** {list(patterns['reason_categories_found'].keys())}")
                        
                        # Show top campaigns and terms
                        junk_by_campaign = junk_notes.get('junk_by_campaign', {})
                        if junk_by_campaign:
                            top_campaigns = list(junk_by_campaign.keys())[:5]
                            st.write(f"**Top Junk Campaigns:** {top_campaigns}")
                        
                        junk_by_term = junk_notes.get('junk_by_term', {})
                        if junk_by_term:
                            top_terms = list(junk_by_term.keys())[:5]
                            st.write(f"**Top Junk Terms:** {top_terms}")
                    
                    # Show specific context details based on type
                    elif context_type == 'filtered_data_analysis':
                        data_summary = claude_context.get('data_summary', {})
                        st.write(f"**Filtered Rows:** {data_summary.get('total_rows', 0)}")
                        st.write(f"**Original Lead Count:** {data_summary.get('original_lead_count', 0)}")
                        st.write(f"**Filtered Lead Count:** {data_summary.get('filtered_lead_count', 0)}")
                        st.write(f"**Unique Campaigns:** {data_summary.get('unique_campaigns', 0)}")
                        st.write(f"**Unique Terms:** {data_summary.get('unique_terms', 0)}")
                        st.write(f"**Date Range:** {data_summary.get('date_range', 'Unknown')}")
                        st.write(f"**Filters Applied:** {claude_context.get('filters_applied', {})}")
                    
                    elif context_type == 'junk_keywords_analysis':
                        junk_summary = claude_context.get('junk_summary', {})
                        st.write(f"**Total Junk Leads:** {junk_summary.get('total_junk_leads', 0)}")
                        st.write(f"**Contact Type Analyzed:** {junk_summary.get('contact_type_analyzed', 'Unknown')}")
                        st.write(f"**Junk Rate:** {junk_summary.get('junk_rate_percent', 0):.1f}%")
                        top_terms = claude_context.get('top_junk_terms', {})
                        if top_terms:
                            st.write(f"**Top Junk Terms:** {list(top_terms.keys())[:5]}")
                    
                    elif context_type == 'enhanced_analysis':
                        analysis_data = claude_context.get('analysis_data', {})
                        data_overview = claude_context.get('data_overview', {})
                        st.write(f"**Total Leads:** {data_overview.get('total_leads', 0)}")
                        st.write(f"**Google Ads Leads:** {data_overview.get('google_ads_leads', 0)}")
                        st.write(f"**Unique Campaigns:** {data_overview.get('unique_campaigns', 0)}")
                        st.write(f"**Average Score:** {data_overview.get('avg_score', 0):.1f}")
                    
                    st.write(f"**Used Learning Examples:** {'Yes' if few_shot_context else 'No'}")
                
                # Prompt for rating
                st.info("👇 Please rate this response below to help improve future answers!")
                
            except Exception as e:
                st.error(f"Claude analysis error: {str(e)}")
                self.logger.error(f"Enhanced AI question processing error: {str(e)}")
                import traceback
                traceback.print_exc()
    
    def create_filtered_data_prompt(self, question: str, claude_context: Dict, few_shot_context: str) -> str:
        """Create prompt for filtered data analysis"""
        prompt_parts = []
        
        # Add few-shot examples if available
        if few_shot_context:
            prompt_parts.append(few_shot_context)
            prompt_parts.append("\n---\n")
        
        # Add current question
        prompt_parts.append(f"Question: {question}\n")
        
        # Add critical instructions for filtered data
        prompt_parts.append("**CRITICAL INSTRUCTIONS:**")
        prompt_parts.append("- The data_table below contains ACTUAL FILTERED DATA ROWS")
        prompt_parts.append("- Each row shows real aggregated leads for Campaign + Term + Month combinations")
        prompt_parts.append("- DO NOT say 'data not available' - the data is right here")
        prompt_parts.append("- Analyze this table directly to provide the specific breakdown requested")
        prompt_parts.append("- Show ALL rows/combinations, don't just sample or summarize")
        prompt_parts.append("")
        
        # Add data table
        data_table = claude_context.get('data_table', [])
        if data_table:
            prompt_parts.append("**FILTERED DATA TABLE:**")
            for i, row in enumerate(data_table):
                row_str = f"Row {i+1}: {dict(row)}"
                prompt_parts.append(row_str)
            prompt_parts.append("")
        
        # Add data summary
        data_summary = claude_context.get('data_summary', {})
        prompt_parts.append(f"Data Summary: {data_summary.get('total_rows', 0)} rows, {data_summary.get('unique_campaigns', 0)} campaigns, {data_summary.get('unique_terms', 0)} terms")
        prompt_parts.append(f"Period: {data_summary.get('date_range', 'Unknown')}")
        prompt_parts.append(f"Filters: {claude_context.get('filters_applied', {})}")
        
        return "\n".join(prompt_parts)
    
    def create_junk_keywords_prompt(self, question: str, claude_context: Dict, few_shot_context: str) -> str:
        """Create prompt for comprehensive junk keywords analysis"""
        prompt_parts = []
        
        if few_shot_context:
            prompt_parts.append(few_shot_context)
            prompt_parts.append("\n---\n")
        
        prompt_parts.append(f"Question: {question}\n")
        
        # Get data counts
        junk_data = claude_context.get('junk_keywords_data', [])
        junk_summary = claude_context.get('junk_summary', {})
        top_terms = claude_context.get('top_junk_terms', {})
        
        prompt_parts.append("=" * 80)
        prompt_parts.append("COMPREHENSIVE JUNK KEYWORDS ANALYSIS - SHOW ALL DATA")
        prompt_parts.append("=" * 80)
        prompt_parts.append("")
        
        prompt_parts.append("**CRITICAL INSTRUCTIONS:**")
        prompt_parts.append(f"✓ You have been provided with {len(junk_data)} complete data records")
        prompt_parts.append(f"✓ Total junk leads to analyze: {junk_summary.get('total_junk_leads', 0)}")
        prompt_parts.append(f"✓ Contact type: {junk_summary.get('contact_type_analyzed', 'Unknown')}")
        prompt_parts.append("✓ Your task: Present ALL records in month-by-month, campaign-wise format")
        prompt_parts.append("✓ DO NOT sample, summarize, or truncate - show EVERY keyword")
        prompt_parts.append("✓ User explicitly wants comprehensive, holistic analysis")
        prompt_parts.append("✓ Length is NOT a concern - completeness IS the priority")
        prompt_parts.append("")
        
        prompt_parts.append("**OUTPUT STRUCTURE REQUIRED:**")
        prompt_parts.append("For each month:")
        prompt_parts.append("  1. Header: **[Month Year] ([Total junk leads for that month])**")
        prompt_parts.append("  2. List each campaign that had junk leads")
        prompt_parts.append("  3. Under each campaign, list ALL terms with counts")
        prompt_parts.append("  4. Format: '* 'term': X junk leads'")
        prompt_parts.append("  5. Ensure sum of all terms = monthly total")
        prompt_parts.append("")
        
        prompt_parts.append("**VERIFICATION CHECKLIST:**")
        prompt_parts.append(f"  □ Did you include all {len(junk_data)} data records?")
        prompt_parts.append("  □ Did you show every month that appears in the data?")
        prompt_parts.append("  □ Did you list every campaign for each month?")
        prompt_parts.append("  □ Did you include every keyword/term with its count?")
        prompt_parts.append("  □ Do your monthly totals match the data?")
        prompt_parts.append("")
        
        # Show complete data is in context
        prompt_parts.append("**DATA PROVIDED IN CONTEXT:**")
        prompt_parts.append(f"All {len(junk_data)} keyword records are in your context above")
        prompt_parts.append(f"All {len(top_terms)} unique terms are listed")
        prompt_parts.append("Use this complete data to generate your comprehensive response")
        prompt_parts.append("")
        
        prompt_parts.append("=" * 80)
        prompt_parts.append("BEGIN YOUR COMPREHENSIVE ANALYSIS BELOW")
        prompt_parts.append("=" * 80)
        
        return "\n".join(prompt_parts)
    
    def create_enhanced_analysis_prompt(self, question: str, claude_context: Dict, few_shot_context: str) -> str:
        """Create prompt for enhanced analysis"""
        prompt_parts = []
        
        # Add few-shot examples if available
        if few_shot_context:
            prompt_parts.append(few_shot_context)
            prompt_parts.append("\n---\n")
        
        # Add current question
        prompt_parts.append(f"Question: {question}\n")
        
        # Add data overview
        data_overview = claude_context.get('data_overview', {})
        if data_overview:
            prompt_parts.append("**DATA OVERVIEW:**")
            prompt_parts.append(f"Total Leads: {data_overview.get('total_leads', 0):,}")
            prompt_parts.append(f"Google Ads Leads: {data_overview.get('google_ads_leads', 0):,}")
            prompt_parts.append(f"Junk Rate: {data_overview.get('junk_rate_percent', 0):.1f}%")
            prompt_parts.append(f"Average Score: {data_overview.get('avg_score', 0):.1f}")
            prompt_parts.append(f"Unique Campaigns: {data_overview.get('unique_campaigns', 0)}")
            prompt_parts.append("")
        
        # Add sample records
        sample_records = claude_context.get('sample_records', [])
        if sample_records:
            prompt_parts.append("**SAMPLE DATA RECORDS:**")
            for i, sample in enumerate(sample_records[:3]):
                prompt_parts.append(f"Sample {i+1} ({sample.get('type', 'Unknown')}):")
                prompt_parts.append(f"  {sample.get('data', {})}")
            prompt_parts.append("")
        
        # Add analysis data
        analysis_data = claude_context.get('analysis_data', {})
        if analysis_data:
            prompt_parts.append("**ANALYSIS CONTEXT:**")
            prompt_parts.append(f"Analysis Focus: {claude_context.get('analysis_focus', 'comprehensive')}")
            
            # Add relevant analysis data based on focus
            for key, value in analysis_data.items():
                if isinstance(value, dict) and len(str(value)) < 500:  # Avoid too much detail
                    prompt_parts.append(f"{key}: {value}")
        
        return "\n".join(prompt_parts)
    
    def create_junk_notes_prompt(self, question: str, claude_context: Dict, few_shot_context: str) -> str:
        """Create prompt for junk notes analysis"""
        prompt_parts = []
        
        if few_shot_context:
            prompt_parts.append(few_shot_context)
            prompt_parts.append("\n---\n")
        
        prompt_parts.append(f"Question: {question}\n")
        
        # Get junk notes data
        junk_notes_data = claude_context.get('junk_notes_data', {})
        
        prompt_parts.append("=" * 80)
        prompt_parts.append("JUNK LEAD NOTES ANALYSIS")
        prompt_parts.append("=" * 80)
        prompt_parts.append("")
        
        prompt_parts.append("**YOUR TASK:**")
        prompt_parts.append("Analyze the junk lead notes data provided in the context above to:")
        prompt_parts.append("1. Identify the main categories of junk reasons")
        prompt_parts.append("2. Find the top 3-5 specific reasons for junk leads")
        prompt_parts.append("3. Analyze patterns by campaign, keyword, and month")
        prompt_parts.append("4. Provide actionable recommendations to reduce junk leads")
        prompt_parts.append("")
        
        prompt_parts.append("**DATA SUMMARY:**")
        prompt_parts.append(f"Total Junk Leads: {junk_notes_data.get('total_junk_leads', 0)}")
        prompt_parts.append(f"Leads with Notes: {junk_notes_data.get('junk_leads_with_notes', 0)}")
        prompt_parts.append(f"Sample Provided: {junk_notes_data.get('sample_size_provided', 0)} records")
        prompt_parts.append(f"Junk Rate: {junk_notes_data.get('junk_rate_percent', 0):.1f}%")
        prompt_parts.append("")
        
        # Show pattern analysis if available
        patterns = junk_notes_data.get('notes_patterns', {})
        if patterns and patterns.get('reason_categories_found'):
            prompt_parts.append("**PATTERN ANALYSIS PROVIDED:**")
            prompt_parts.append("The context includes pre-analyzed junk reason categories.")
            prompt_parts.append("Use these to structure your analysis.")
            prompt_parts.append("")
        
        prompt_parts.append("**CRITICAL INSTRUCTIONS:**")
        prompt_parts.append("✓ The actual notes data is in your context above")
        prompt_parts.append("✓ Reference specific examples from the notes")
        prompt_parts.append("✓ Cite actual campaign names and keywords")
        prompt_parts.append("✓ Use the pattern analysis to categorize reasons")
        prompt_parts.append("✓ Be specific and actionable in recommendations")
        prompt_parts.append("✓ DO NOT invent reasons not in the data")
        prompt_parts.append("")
        
        prompt_parts.append("=" * 80)
        prompt_parts.append("BEGIN YOUR ANALYSIS")
        prompt_parts.append("=" * 80)
        
        return "\n".join(prompt_parts)
    
    def get_data_scope_from_claude_context(self, claude_context: Dict) -> str:
        """Get data scope description from Claude context"""
        scope_parts = []
        
        context_type = claude_context.get('context_type', 'general')
        
        if context_type == 'filtered_data_analysis':
            data_summary = claude_context.get('data_summary', {})
            scope_parts.append(f"Filtered: {data_summary.get('total_rows', 0)} rows")
            scope_parts.append(f"Campaigns: {data_summary.get('unique_campaigns', 0)}")
            scope_parts.append(f"Terms: {data_summary.get('unique_terms', 0)}")
            
        elif context_type == 'junk_keywords_analysis':
            junk_summary = claude_context.get('junk_summary', {})
            scope_parts.append(f"Junk: {junk_summary.get('total_junk_leads', 0)} leads")
            scope_parts.append(f"Contact Type: {junk_summary.get('contact_type_analyzed', 'Unknown')}")
            
        else:
            data_overview = claude_context.get('data_overview', {})
            scope_parts.append(f"Total: {data_overview.get('total_leads', 0)} leads")
            scope_parts.append(f"Campaigns: {data_overview.get('unique_campaigns', 0)}")
        
        months = st.session_state.selected_months
        if months:
            scope_parts.append(f"Period: {months[0]} to {months[-1]}")
        
        return " | ".join(scope_parts) if scope_parts else "Full dataset"
    
    def debug_analysis_data_structure(self, analysis_data: Dict[str, Any]):
        """Debug helper to see what data structure we actually have"""
        st.write("### 🔍 Analysis Data Structure Debug")
        
        # Campaign Analysis Structure
        st.write("**Campaign Analysis Keys:**")
        campaign_keys = list(analysis_data.get('campaign_analysis', {}).keys())
        st.write(campaign_keys)
        
        if 'monthly_trends' in campaign_keys:
            monthly_trends = analysis_data['campaign_analysis']['monthly_trends']
            st.write("**Monthly Trends Structure:**")
            st.write(type(monthly_trends))
            st.write(list(monthly_trends.keys()) if isinstance(monthly_trends, dict) else "Not a dict")
        
        # CRM Analysis Structure  
        st.write("**CRM Analysis Keys:**")
        crm_keys = list(analysis_data.get('crm_analysis', {}).keys())
        st.write(crm_keys)
        
        if 'lead_quality_patterns' in crm_keys:
            quality_patterns = analysis_data['crm_analysis']['lead_quality_patterns']
            st.write("**Lead Quality Patterns Structure:**")
            st.write(type(quality_patterns))
            st.write(list(quality_patterns.keys()) if isinstance(quality_patterns, dict) else "Not a dict")
    
    def render_learning_dashboard(self):
        """NEW: Render learning system dashboard"""
        st.header("🎓 Learning System Dashboard")
        st.markdown("*Track response quality and continuous improvement*")
        
        # Quality Overview
        quality_report = self.learner.get_quality_report()
        
        if quality_report.get('message'):
            st.info(quality_report['message'])
            st.markdown("""
            ### How to Use the Learning System
            
            1. **Ask Claude questions** about your Google Ads and CRM data
            2. **Rate responses** using the slider (1-10 scale)
            3. **Add feedback notes** to explain what was good or bad
            4. **System learns** from highly-rated responses (8+)
            5. **Future responses improve** using patterns from best answers
            
            Start rating responses in the Claude AI Chat tab!
            """)
            return
        
        # Quality Metrics
        st.subheader("📊 Quality Metrics Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Conversations",
                quality_report['total_conversations']
            )
        
        with col2:
            st.metric(
                "Average Rating",
                f"{quality_report['average_rating']:.1f}/10"
            )
        
        with col3:
            st.metric(
                "High Quality (8+)",
                quality_report['high_quality_count']
            )
        
        with col4:
            st.metric(
                "Needs Improvement",
                quality_report['needs_improvement_count']
            )
        
        # Rating Distribution
        st.subheader("📈 Rating Distribution")
        
        rating_dist = quality_report['rating_distribution']
        if rating_dist:
            fig = go.Figure(data=[
                go.Bar(
                    x=list(rating_dist.keys()),
                    y=list(rating_dist.values()),
                    marker_color=['#dc3545' if int(k) <= 5 else '#ffc107' if int(k) <= 7 else '#28a745' 
                                  for k in rating_dist.keys()]
                )
            ])
            
            fig.update_layout(
                title="Distribution of Response Ratings",
                xaxis_title="Rating (1-10)",
                yaxis_title="Number of Responses",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Quality by Question Type
        if quality_report.get('quality_by_type'):
            st.subheader("🎯 Quality by Question Type")
            
            quality_by_type = quality_report['quality_by_type']
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(quality_by_type.keys()),
                    y=list(quality_by_type.values()),
                    marker_color='#4285f4'
                )
            ])
            
            fig.update_layout(
                title="Average Rating by Question Type",
                xaxis_title="Question Type",
                yaxis_title="Average Rating",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Improvement Suggestions
        st.subheader("💡 Improvement Suggestions")
        
        suggestions = self.learner.get_improvement_suggestions()
        for suggestion in suggestions:
            if "CRISIS" in suggestion or "CRITICAL" in suggestion:
                st.error(suggestion)
            elif "NEEDS" in suggestion or "%" in suggestion:
                st.warning(suggestion)
            else:
                st.success(suggestion)
        
        # Quality Indicators
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="quality-metric">', unsafe_allow_html=True)
            st.metric(
                "Hallucination Rate",
                f"{quality_report['hallucination_rate']:.1f}%",
                delta=f"{'Good' if quality_report['hallucination_rate'] < 10 else 'Needs Work'}",
                delta_color="inverse"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="quality-metric">', unsafe_allow_html=True)
            st.metric(
                "Data Attribution Rate",
                f"{quality_report['attribution_rate']:.1f}%",
                delta=f"{'Good' if quality_report['attribution_rate'] > 80 else 'Needs Work'}"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        # High Quality Examples
        st.subheader("⭐ High Quality Examples (8+ ratings)")
        
        high_quality_examples = self.learner.get_high_quality_examples(min_rating=8, limit=5)
        
        if high_quality_examples:
            for i, example in enumerate(high_quality_examples, 1):
                with st.expander(f"Example {i}: {example['question'][:60]}... (Rated {example['rating']}/10)"):
                    st.markdown(f"**Question:** {example['question']}")
                    st.markdown(f"**Question Type:** {example['question_type']}")
                    st.markdown(f"**Data Scope:** {example['data_scope']}")
                    st.markdown(f"**Response Preview:** {example['response'][:300]}...")
        else:
            st.info("No high-quality examples yet. Rate more responses with 8+ to see examples here.")
    
    # Keep all your existing methods for other tabs
    def render_smart_dashboard(self):
        """Render smart dashboard"""
        st.header("📊 Smart Performance Dashboard")
        st.markdown("*AI-powered insights for your marketing campaigns*")
        
        with st.spinner("🧠 AI analyzing your data..."):
            analysis_data = self.get_comprehensive_analysis()
        
        if not analysis_data or analysis_data.get('error'):
            st.error("Unable to load analysis data. Please check your data connections.")
            return
        
        st.subheader("🎯 Key Performance Indicators")
        self.render_smart_kpi_cards(analysis_data)

        # Add this right after self.render_smart_kpi_cards(analysis_data) in render_smart_dashboard
        if st.button("🔍 Debug Analysis Data Structure"):
            self.debug_analysis_data_structure(analysis_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Performance Trends")
            self.render_enhanced_monthly_trends(analysis_data)
        
        with col2:
            st.subheader("🎯 Lead Quality Analysis")
            self.render_lead_quality_dashboard(analysis_data)
    
    # Add this method to app.py or replace the existing render_smart_kpi_cards method

    def render_smart_kpi_cards(self, analysis_data: Dict[str, Any]):
        """Render smart KPI cards - FIXED for actual data format"""    
        # DIAGNOSTIC CODE - Add this at the very start
        st.write("### Raw Data Check")
        
        if hasattr(self.analyzer, 'campaign_data') and self.analyzer.campaign_data is not None:
            st.write(f"campaign_data shape: {self.analyzer.campaign_data.shape}")
            st.write(f"campaign_data columns: {list(self.analyzer.campaign_data.columns)}")
            st.write(f"Cost column type: {self.analyzer.campaign_data['Cost'].dtype}")
            st.write(f"Sample Cost values: {self.analyzer.campaign_data['Cost'].head(3).tolist()}")
            st.write(f"Total Cost SUM: ${self.analyzer.campaign_data['Cost'].sum():,.2f}")
        else:
            st.error("campaign_data is None!")
        
        if hasattr(self.analyzer, 'monthly_campaign_data') and self.analyzer.monthly_campaign_data is not None:
            st.write(f"monthly_campaign_data shape: {self.analyzer.monthly_campaign_data.shape}")
            st.write(f"monthly_campaign_data columns: {list(self.analyzer.monthly_campaign_data.columns)}")
        else:
            st.error("monthly_campaign_data is None!")
        
        st.write("### Analysis Data Structure")
        st.write(f"campaign_analysis keys: {list(analysis_data.get('campaign_analysis', {}).keys())}")
        st.write(f"overview keys: {list(analysis_data.get('campaign_analysis', {}).get('overview', {}).keys())}")
        
        campaign_metrics = analysis_data.get('campaign_analysis', {}).get('overview', {})
        st.write(f"total_spend from analysis: {campaign_metrics.get('total_spend', 'KEY NOT FOUND')}")
        
        st.write("---")
        
        # Get campaign metrics (these might be zero if no campaign data or column mismatches)
        campaign_metrics = analysis_data.get('campaign_analysis', {}).get('overview', {})
        
        # FIXED: Get CRM data directly for more reliable metrics
        try:
            # Load raw CRM data for direct calculation
            if hasattr(self.analyzer, 'crm_data') and self.analyzer.crm_data is not None:
                df = self.analyzer.crm_data
                
                # FIXED: Calculate metrics directly from raw data
                total_leads = len(df)
                
                # FIXED: Handle array-type Contact_Type for junk detection
                def is_junk_lead(contact_type):
                    if pd.isna(contact_type):
                        return False
                    contact_str = str(contact_type).lower()
                    return 'junk' in contact_str
                
                junk_count = df.apply(lambda row: is_junk_lead(row.get('Contact Type')), axis=1).sum()
                google_ads_count = len(df[df['Lead Source'] == 'Web Mail']) if 'Lead Source' in df.columns else 0
                
                # FIXED: Realistic quality calculation
                if 'Score' in df.columns:
                    scored_leads = df[df['Score'] > 0]  # Any score > 0
                    quality_count = len(scored_leads)
                    avg_score = float(df['Score'].mean())
                    quality_pct = (quality_count / total_leads * 100) if total_leads > 0 else 0
                else:
                    quality_count = 0
                    avg_score = 0
                    quality_pct = 0
                    
                # Calculate junk rate
                junk_pct = (junk_count / total_leads * 100) if total_leads > 0 else 0
                
            else:
                # Fallback to analysis data
                crm_overview = analysis_data.get('crm_analysis', {}).get('overview', {})
                total_leads = crm_overview.get('total_leads', 0)
                quality_count = crm_overview.get('high_quality_leads', 0)
                quality_pct = crm_overview.get('conversion_rate', 0)
                junk_count = crm_overview.get('junk_leads', 0)
                junk_pct = (junk_count / total_leads * 100) if total_leads > 0 else 0
                avg_score = crm_overview.get('avg_score', 0)
                google_ads_count = crm_overview.get('google_ads_leads', 0)
                
        except Exception as e:
            st.error(f"Error calculating KPIs: {str(e)}")
            # Fallback values
            total_leads = 0
            quality_count = 0
            quality_pct = 0
            junk_count = 0
            junk_pct = 0
            avg_score = 0
            google_ads_count = 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # FIXED: Handle missing campaign spend data
            total_spend = campaign_metrics.get('total_spend', 0)
            utilization = campaign_metrics.get('budget_utilization', 0)
            
            if total_spend == 0:
                # Show data source info instead
                st.metric(
                    "Data Loaded", 
                    "✅ Connected",
                    delta=f"{google_ads_count:,} Google Ads leads"
                )
                st.markdown('<div class="performance-good">Status: Data Available</div>', 
                        unsafe_allow_html=True)
            else:
                st.metric(
                    "Campaign Spend", 
                    format_currency(total_spend),
                    delta=f"{utilization:.1f}% utilization"
                )
                status_class = "performance-good" if 80 <= utilization <= 100 else "performance-critical"
                st.markdown(f'<div class="{status_class}">Status: {get_utilization_status(utilization)}</div>', 
                        unsafe_allow_html=True)
        
        with col2:
            # FIXED: Show actual lead counts and realistic quality metrics
            crisis_level = detect_lead_quality_crisis(quality_pct)
            status_class = "performance-crtical" if "CRISIS" in crisis_level else "performance-good"
            
            st.metric(
                "Total Leads",
                f"{total_leads:,}",
                delta=f"{quality_pct:.1f}% with scores >0"
            )
            st.markdown(f'<div class="{status_class}">Quality: {crisis_level}</div>', 
                    unsafe_allow_html=True)
        
        with col3:
            # FIXED: Show Google Ads specific metrics
            conversions = campaign_metrics.get('total_conversions', 0)
            conv_rate = campaign_metrics.get('overall_conversion_rate', 0)
            
            if conversions == 0:
                # Show Google Ads lead data instead
                st.metric(
                    "Google Ads Leads",
                    f"{google_ads_count:,}",
                    delta=f"{(google_ads_count/total_leads*100):.1f}% of total" if total_leads > 0 else "0% of total"
                )
                status_class = "performance-good" if google_ads_count > 0 else "performance-critical"
                st.markdown(f'<div class="{status_class}">Source: {"Active" if google_ads_count > 0 else "No Data"}</div>', 
                        unsafe_allow_html=True)
            else:
                st.metric(
                    "Conversions",
                    f"{conversions:,}",
                    delta=f"{conv_rate:.2f}% rate"
                )
                status_class = "performance-good" if conv_rate > 3 else "performance-critical"
                st.markdown(f'<div class="{status_class}">Performance: {get_performance_grade(conv_rate)}</div>', 
                        unsafe_allow_html=True)
        
        with col4:
            # FIXED: Show actual junk metrics
            status_class = "performance-good" if junk_pct < 30 else "performance-critical"
            
            st.metric(
                "Junk Leads",
                f"{junk_count:,}",
                delta=f"{junk_pct:.1f}% junk rate"
            )
            st.markdown(f'<div class="{status_class}">Targeting: {"Good" if junk_pct < 30 else "Poor"}</div>', 
                    unsafe_allow_html=True)
        
        # FIXED: Add debug info
        with st.expander("🔍 Data Debug Info", expanded=False):
            st.write(f"**Total CRM Records:** {total_leads:,}")
            st.write(f"**Google Ads Leads:** {google_ads_count:,}")
            st.write(f"**Leads with Scores >0:** {quality_count:,}")
            st.write(f"**Average Score:** {avg_score:.2f}")
            st.write(f"**Junk Leads:** {junk_count:,}")
            
            if hasattr(self.analyzer, 'crm_data') and self.analyzer.crm_data is not None:
                sample_contacts = self.analyzer.crm_data['Contact Type'].head(3).tolist()
                st.write(f"**Sample Contact Types:** {sample_contacts}")

    def render_enhanced_monthly_trends(self, analysis_data: Dict[str, Any]):
        """Render monthly trends chart with better error handling"""
        try:
            # DIAGNOSTIC: See what we have
            st.write("**Debug: Checking monthly_trends data...**")
            
            campaign_analysis = analysis_data.get('campaign_analysis', {})
            st.write(f"Campaign analysis keys: {list(campaign_analysis.keys())}")
            
            # Try multiple possible data structures
            monthly_data = None
            
            # Option 1: nested in monthly_trends
            if 'monthly_trends' in campaign_analysis:
                monthly_trends = campaign_analysis['monthly_trends']
                st.write(f"Found monthly_trends, type: {type(monthly_trends)}")
                
                if isinstance(monthly_trends, dict) and 'monthly_data' in monthly_trends:
                    monthly_data = monthly_trends['monthly_data']
                elif isinstance(monthly_trends, dict):
                    monthly_data = monthly_trends
            
            # Option 2: directly in campaign_analysis as monthly_data
            elif 'monthly_data' in campaign_analysis:
                monthly_data = campaign_analysis['monthly_data']
            
            # Option 3: Try to build from monthly_campaign_data
            elif hasattr(self.analyzer, 'monthly_campaign_data') and self.analyzer.monthly_campaign_data is not None:
                st.info("Building monthly data from raw monthly_campaign_data...")
                df = self.analyzer.monthly_campaign_data
                
                if 'Month' in df.columns:
                    monthly_data = {}
                    for month in df['Month'].unique():
                        month_df = df[df['Month'] == month]
                        monthly_data[month] = {
                            'cost': month_df['Cost'].sum() if 'Cost' in df.columns else 0,
                            'conversions': month_df['Conversions'].sum() if 'Conversions' in df.columns else 0,
                            'clicks': month_df['Clicks'].sum() if 'Clicks' in df.columns else 0,
                            'impressions': month_df['Impressions'].sum() if 'Impressions' in df.columns else 0
                        }
            
            if not monthly_data or len(monthly_data) == 0:
                st.warning("No monthly trend data available")
                st.info("Check if analyzer.analyze_campaign_performance() is returning monthly_trends properly")
                return
            
            st.write(f"**Monthly data found:** {len(monthly_data)} months")
            st.write(f"**Months:** {list(monthly_data.keys())}")
            
            # Sort months chronologically
            from datetime import datetime
            def month_to_date(month_str):
                try:
                    return datetime.strptime(month_str, '%B %Y')
                except:
                    return datetime.now()
            
            months = sorted(monthly_data.keys(), key=month_to_date)
            
            # Extract data
            costs = [monthly_data[m].get('cost', 0) for m in months]
            conversions = [monthly_data[m].get('conversions', 0) for m in months]
            
            st.write(f"**Costs:** {costs}")
            st.write(f"**Conversions:** {conversions}")
            
            # Create chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=months,
                y=costs,
                mode='lines+markers',
                name='Monthly Spend',
                line=dict(color='#4285f4', width=3),
                marker=dict(size=8)
            ))
            
            fig.add_trace(go.Scatter(
                x=months,
                y=conversions,
                mode='lines+markers',
                name='Conversions',
                yaxis='y2',
                line=dict(color='#34a853', width=3),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title="Monthly Performance Trends",
                xaxis_title="Month",
                yaxis_title="Cost ($)",
                yaxis2=dict(title="Conversions", overlaying='y', side='right'),
                template="plotly_white",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error rendering monthly trends: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    def render_lead_quality_dashboard(self, analysis_data: Dict[str, Any]):
        """Render lead quality dashboard with better error handling"""
        try:
            st.write("**Debug: Checking lead quality data...**")
            
            crm_analysis = analysis_data.get('crm_analysis', {})
            st.write(f"CRM analysis keys: {list(crm_analysis.keys())}")
            
            # Try to get quality data
            quality_data = None
            
            # Option 1: From lead_quality_patterns
            if 'lead_quality_patterns' in crm_analysis:
                patterns = crm_analysis['lead_quality_patterns']
                st.write(f"Found lead_quality_patterns, type: {type(patterns)}")
                
                if isinstance(patterns, dict) and 'quality_distribution' in patterns:
                    quality_data = patterns['quality_distribution']
                elif isinstance(patterns, dict):
                    quality_data = patterns
            
            # Option 2: Build from raw CRM data
            elif hasattr(self.analyzer, 'crm_data') and self.analyzer.crm_data is not None:
                st.info("Building quality data from raw crm_data...")
                df = self.analyzer.crm_data
                
                total_leads = len(df)
                
                # Count junk leads
                def is_junk(contact_type):
                    if pd.isna(contact_type):
                        return False
                    return 'junk' in str(contact_type).lower()
                
                junk_count = df.apply(lambda row: is_junk(row.get('Contact Type')), axis=1).sum()
                
                # Count high quality (score > 70)
                high_quality = 0
                if 'Score' in df.columns:
                    high_quality = len(df[df['Score'] > 70])
                
                quality_data = {
                    'total_leads': total_leads,
                    'junk_count': junk_count,
                    'high_quality_count': high_quality,
                    'other_count': total_leads - junk_count - high_quality
                }
            
            if not quality_data:
                st.warning("No lead quality data available")
                st.info("Check if analyzer.analyze_crm_performance() is returning lead_quality_patterns properly")
                return
            
            st.write(f"**Quality data:** {quality_data}")
            
            # Create pie chart
            total_leads = quality_data.get('total_leads', 0)
            junk_count = quality_data.get('junk_count', 0)
            high_quality = quality_data.get('high_quality_count', 0)
            other = total_leads - junk_count - high_quality
            
            labels = ['High Quality', 'Junk Leads', 'Other']
            values = [high_quality, junk_count, other]
            
            # Filter out zero values
            filtered_data = [(l, v) for l, v in zip(labels, values) if v > 0]
            if filtered_data:
                labels, values = zip(*filtered_data)
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.3,
                marker_colors=['#34a853', '#ea4335', '#fbbc04'],
                textinfo='label+percent',
                textposition='outside'
            )])
            
            fig.update_layout(
                title="Lead Quality Distribution",
                template="plotly_white",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add metrics below chart
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("High Quality", f"{high_quality:,}", 
                        f"{(high_quality/total_leads*100):.1f}%" if total_leads > 0 else "0%")
            with col2:
                st.metric("Junk Leads", f"{junk_count:,}",
                        f"{(junk_count/total_leads*100):.1f}%" if total_leads > 0 else "0%")
            with col3:
                st.metric("Other", f"{other:,}",
                        f"{(other/total_leads*100):.1f}%" if total_leads > 0 else "0%")
            
        except Exception as e:
            st.error(f"Error rendering lead quality dashboard: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    def render_enhanced_deep_analysis(self):
        """Render enhanced deep analysis"""
        st.header("🔍 Deep Performance Analysis")
        
        analysis_data = self.get_comprehensive_analysis()
        
        if not analysis_data:
            st.error("No analysis data available")
            return
        
        st.subheader("📊 Campaign Performance Deep Dive")
        campaign_analysis = analysis_data.get('campaign_analysis', {})
        
        if campaign_analysis and not campaign_analysis.get('error'):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Overall Metrics")
                metrics = campaign_analysis.get('overall_metrics', {})
                st.json({
                    'Total Spend': format_currency(metrics.get('total_spend', 0)),
                    'Total Clicks': f"{metrics.get('total_clicks', 0):,}",
                    'Total Conversions': f"{metrics.get('total_conversions', 0):,}",
                    'Overall CTR': format_percentage(metrics.get('overall_ctr', 0)),
                    'Overall CPC': format_currency(metrics.get('overall_cpc', 0)),
                    'Conversion Rate': format_percentage(metrics.get('overall_conversion_rate', 0))
                })
            
            with col2:
                st.markdown("#### Data Insights")
                anomalies = campaign_analysis.get('data_anomalies', [])
                patterns = campaign_analysis.get('performance_patterns', [])
                
                if anomalies:
                    st.markdown("**Critical Issues:**")
                    for anomaly in anomalies[:3]:
                        st.markdown(f"- {anomaly}")
                
                if patterns:
                    st.markdown("**Key Patterns:**")
                    for pattern in patterns[:3]:
                        st.markdown(f"- {pattern}")
        
        st.subheader("🎯 Lead Quality Deep Dive")
        crm_analysis = analysis_data.get('crm_analysis', {})
        
        if crm_analysis and not crm_analysis.get('error'):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Lead Quality Metrics")
                quality_data = crm_analysis.get('lead_quality_patterns', {}).get('quality_distribution', {})
                
                if quality_data:
                    st.json({
                        'Total Leads': f"{quality_data.get('total_leads', 0):,}",
                        'Quality Percentage': f"{quality_data.get('quality_percentage', 0):.2f}%",
                        'Junk Count': f"{quality_data.get('junk_count', 0):,}",
                        'Crisis Level': detect_lead_quality_crisis(quality_data.get('quality_percentage', 0))
                    })
            
            with col2:
                st.markdown("#### Google Ads Attribution")
                attribution = crm_analysis.get('google_ads_attribution', {})
                
                if attribution:
                    st.json({
                        'Google Ads Leads': f"{attribution.get('total_google_ads_leads', 0):,}",
                        'GA Percentage': f"{attribution.get('google_ads_percentage', 0):.1f}%"
                    })
    
    def render_ai_insights_interface(self):
        """Render AI insights interface"""
        st.header("💡 AI-Powered Insights")
        st.markdown("*Proactive insights and recommendations from Claude*")
        
        if st.button("🧠 Generate AI Insights", type="primary"):
            self.generate_proactive_ai_insights()
        
        if hasattr(st.session_state, 'latest_ai_insights'):
            insights = st.session_state.latest_ai_insights
            st.markdown("### Latest AI Insights")
            st.markdown(insights['content'])
            st.markdown(f"*Generated: {insights['generated_at'].strftime('%Y-%m-%d %H:%M:%S')}*")
    
    def generate_proactive_ai_insights(self):
        """Generate proactive AI insights"""
        with st.spinner("🧠 Claude is analyzing your data for insights..."):
            try:
                insights_question = "Generate comprehensive insights and recommendations for my Google Ads and CRM performance"
                
                # Use the same context system as the chat
                claude_context = self.response_generator.generate_claude_context(insights_question)
                
                insights_prompt = self.create_enhanced_analysis_prompt(
                    insights_question,
                    claude_context,
                    ""  # No few-shot for insights
                )
                
                claude_insights = self.anthropic_client.generate_response(
                    insights_prompt,
                    context=claude_context
                )
                
                st.success("🧠 AI Insights Generated!")
                st.markdown(claude_insights)
                
                st.session_state.latest_ai_insights = {
                    'content': claude_insights,
                    'generated_at': datetime.now(),
                    'context_quality': len(claude_context.get('data_overview', {}))
                }
                
            except Exception as e:
                st.error(f"Error generating AI insights: {str(e)}")
    
    def show_smart_question_suggestions(self):
        """Show smart question suggestions"""
        try:
            analysis_data = self.get_comprehensive_analysis()
            
            suggestions = []
            
            # Check for junk lead issues
            quality_data = analysis_data.get('crm_analysis', {}).get('lead_quality_patterns', {})
            quality_dist = quality_data.get('quality_distribution', {})
            junk_count = quality_dist.get('junk_count', 0)
            total_leads = quality_dist.get('total_leads', 1)
            junk_pct = (junk_count / total_leads * 100) if total_leads > 0 else 0
            
            # NEW: Add junk notes suggestion if high junk rate
            if junk_pct > 20:
                suggestions.append("📝 Can you analyze the notes column and find out why there are so many junk leads?")
            
            quality_pct = quality_dist.get('quality_percentage', 0)
            if quality_pct < 1:
                suggestions.append("🚨 Why is my lead quality in crisis mode?")
            
            campaign_metrics = analysis_data.get('campaign_analysis', {}).get('overall_metrics', {})
            utilization = campaign_metrics.get('budget_utilization', 0)
            
            if utilization < 80:
                suggestions.append("💰 How should I optimize my underutilized budget?")
            
            # Add filtered data suggestions
            suggestions.append("📊 List keywords creating junk leads for Contact Type = 3pl, month-by-month, campaign-wise")
            
            if suggestions:
                st.markdown("#### 💡 Smart Suggestions Based on Your Data:")
                cols = st.columns(min(len(suggestions), 3))
                for i, suggestion in enumerate(suggestions[:3]):
                    with cols[i]:
                        if st.button(suggestion, key=f"smart_suggest_{i}"):
                            st.session_state.quick_question = suggestion
                            st.rerun()
        
        except Exception as e:
            self.logger.error(f"Error generating smart suggestions: {str(e)}")
    
    def get_comprehensive_analysis(self) -> Dict[str, Any]:
        """Get comprehensive analysis with caching"""
        cache_key = f"enhanced_analysis_{hash(str(st.session_state.selected_months))}"
        
        cached_result = self.cache_manager.get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        try:
            self.analyzer.load_data(st.session_state.selected_months or None)
            
            analysis_data = {}
            
            try:
                campaign_analysis = self.analyzer.analyze_campaign_performance(
                    st.session_state.selected_months or None
                )
                analysis_data['campaign_analysis'] = campaign_analysis
            except Exception as e:
                self.logger.warning(f"Campaign analysis failed: {str(e)}")
                analysis_data['campaign_analysis'] = {'error': str(e)}
            
            try:
                crm_analysis = self.analyzer.analyze_crm_performance(
                    st.session_state.selected_months or None
                )
                analysis_data['crm_analysis'] = crm_analysis
            except Exception as e:
                self.logger.warning(f"CRM analysis failed: {str(e)}")
                analysis_data['crm_analysis'] = {'error': str(e)}
            
            try:
                cross_channel = self.analyzer.cross_channel_analysis()
                analysis_data['cross_channel_analysis'] = cross_channel
            except Exception as e:
                self.logger.warning(f"Cross-channel analysis failed: {str(e)}")
            
            self.cache_manager.cache_analysis_result(cache_key, analysis_data, ttl_minutes=15)
            
            return analysis_data
            
        except Exception as e:
            self.logger.error(f"Error getting comprehensive analysis: {str(e)}")
            return {'error': str(e)}
    
    def refresh_all_data(self):
        """Enhanced data refresh"""
        with st.spinner("🔄 Refreshing all data..."):
            try:
                st.session_state.analysis_cache = {}
                self.cache_manager.clear_expired_cache()
                self.analyzer.load_data(st.session_state.selected_months or None)
                st.session_state.conversation_history = []
                st.session_state.last_refresh = datetime.now()
                
                st.success("✅ All data refreshed successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error refreshing data: {str(e)}")
    
    def clear_all_caches(self):
        """Clear all caches"""
        try:
            st.session_state.analysis_cache = {}
            self.cache_manager.clear_expired_cache()
            st.success("✅ All caches cleared!")
        except Exception as e:
            st.error(f"Error clearing caches: {str(e)}")
    
    def get_available_months(self) -> Dict[str, List[str]]:
        """Get available months with error handling"""
        try:
            return self.data_reader.get_available_months()
        except Exception as e:
            self.logger.error(f"Error getting available months: {str(e)}")
            return {'campaign_months': [], 'crm_months': []}
    
    def show_detailed_connection_status(self):
        """Show detailed connection status in sidebar"""
        st.sidebar.subheader("🔗 Data Connections")
        
        try:
            validation_results = self.data_reader.validate_data_structure()
            
            campaign_status = validation_results.get('campaign_data', {})
            if campaign_status.get('valid', False):
                st.sidebar.success("✅ Campaign Structure: Valid")
            else:
                st.sidebar.error("❌ Campaign Structure: Issues")
            
            crm_status = validation_results.get('crm_data', {})
            if crm_status.get('valid', False):
                st.sidebar.success("✅ CRM Structure: Valid")
            else:
                st.sidebar.error("❌ CRM Structure: Issues")
            
        except Exception as e:
            st.sidebar.error(f"❌ Connection test failed: {str(e)}")

def main():
    """Main application entry point"""
    try:
        app = GoogleAdsAnalysisApp()
        app.run()
    except Exception as e:
        st.error(f"Application startup failed: {str(e)}")
        st.info("Please check your configuration and try again.")

if __name__ == "__main__":
    main()