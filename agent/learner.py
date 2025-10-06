# learner.py - Consolidated Learning System
# Q&A Feedback with Ratings + ML Pattern Learning for Google Ads & CRM Analysis

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import json
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import pickle
from collections import defaultdict, Counter
import hashlib
from dataclasses import dataclass, asdict
import re

@dataclass
class ConversationRecord:
    """Q&A interaction with quality rating"""
    id: str
    timestamp: str
    question: str
    question_type: str
    response: str
    rating: Optional[int]
    data_scope: str
    feedback_notes: str
    response_length: int
    had_hallucination: bool
    was_question_specific: bool
    included_data_attribution: bool

class EnhancedLearningSystem:
    """
    Dual-purpose learning system for Google Ads & CRM Analysis:
    1. Q&A feedback with ratings (1-10) for continuous improvement
    2. ML pattern learning from campaign/CRM data for insights
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.qa_feedback_path = 'data/qa_feedback.json'
        self.models_path = 'data/models/'
        self.patterns_path = 'data/patterns/'
        self.qa_conversations = []
        self._ensure_directories()
        self._load_qa_conversations()
    
    def _ensure_directories(self):
        """Ensure required directories exist"""
        os.makedirs(os.path.dirname(self.qa_feedback_path), exist_ok=True)
        os.makedirs(self.models_path, exist_ok=True)
        os.makedirs(self.patterns_path, exist_ok=True)
    
    # ==================== Q&A FEEDBACK SYSTEM ====================
    
    def record_qa_interaction(
        self,
        question: str,
        response: str,
        question_type: str,
        data_scope: str,
        rating: Optional[int] = None,
        feedback_notes: str = ""
    ) -> str:
        """
        Record Q&A interaction with optional rating
        
        Args:
            question: User's question
            response: System's response
            question_type: Type of question (performance, budget, trends, etc.)
            data_scope: Data used (e.g., "August 2025 campaigns", "All CRM data")
            rating: Optional rating 1-10
            feedback_notes: Optional feedback text
            
        Returns:
            Conversation ID
        """
        try:
            conv_id = hashlib.md5(
                f"{question}{datetime.now().isoformat()}".encode()
            ).hexdigest()[:12]
            
            quality_metrics = self._analyze_response_quality(response, question)
            
            record = ConversationRecord(
                id=conv_id,
                timestamp=datetime.now().isoformat(),
                question=question,
                question_type=question_type,
                response=response,
                rating=rating,
                data_scope=data_scope,
                feedback_notes=feedback_notes,
                response_length=len(response),
                had_hallucination=quality_metrics['has_hallucination'],
                was_question_specific=quality_metrics['is_question_specific'],
                included_data_attribution=quality_metrics['has_data_attribution']
            )
            
            self.qa_conversations.append(record)
            self._save_qa_conversations()
            
            self.logger.info(f"Recorded conversation {conv_id}")
            return conv_id
            
        except Exception as e:
            self.logger.error(f"Error recording Q&A: {str(e)}")
            return ""
    
    def update_qa_rating(self, conversation_id: str, rating: int, feedback_notes: str = "") -> bool:
        """
        Update rating for a Q&A interaction
        
        Args:
            conversation_id: ID of conversation to update
            rating: Rating 1-10 (1=poor, 10=excellent)
            feedback_notes: Optional detailed feedback
            
        Returns:
            Success status
        """
        try:
            if not 1 <= rating <= 10:
                raise ValueError("Rating must be between 1 and 10")
            
            for conv in self.qa_conversations:
                if conv.id == conversation_id:
                    conv.rating = rating
                    if feedback_notes:
                        conv.feedback_notes = feedback_notes
                    self._save_qa_conversations()
                    self.logger.info(f"Updated rating for {conversation_id}: {rating}/10")
                    return True
            
            self.logger.warning(f"Conversation {conversation_id} not found")
            return False
            
        except Exception as e:
            self.logger.error(f"Error updating rating: {str(e)}")
            return False
    
    def get_high_quality_examples(
        self,
        question_type: Optional[str] = None,
        min_rating: int = 8,
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Get high-quality examples for few-shot learning
        
        Args:
            question_type: Filter by question type
            min_rating: Minimum rating threshold
            limit: Maximum number of examples
            
        Returns:
            List of high-quality Q&A examples
        """
        try:
            filtered = [
                conv for conv in self.qa_conversations
                if conv.rating is not None 
                and conv.rating >= min_rating
                and not conv.had_hallucination
                and conv.included_data_attribution
                and (question_type is None or conv.question_type == question_type)
            ]
            
            filtered.sort(key=lambda x: x.rating, reverse=True)
            
            return [
                {
                    'question': conv.question,
                    'response': conv.response,
                    'rating': conv.rating,
                    'question_type': conv.question_type,
                    'data_scope': conv.data_scope
                }
                for conv in filtered[:limit]
            ]
            
        except Exception as e:
            self.logger.error(f"Error retrieving examples: {str(e)}")
            return []
    
    def build_few_shot_context(self, question: str, question_type: str) -> str:
        """
        Build few-shot learning context from high-quality examples
        This context can be injected into LLM prompts
        
        Args:
            question: Current user question
            question_type: Type of question
            
        Returns:
            Formatted context string for LLM
        """
        try:
            examples = self.get_high_quality_examples(
                question_type=question_type,
                min_rating=8,
                limit=2
            )
            
            if not examples:
                return ""
            
            context_parts = ["EXAMPLE HIGH-QUALITY RESPONSES:\n"]
            
            for i, example in enumerate(examples, 1):
                response_preview = example['response'][:400] + "..." if len(example['response']) > 400 else example['response']
                
                context_parts.append(f"\nExample {i} (Rated {example['rating']}/10):")
                context_parts.append(f"Q: {example['question']}")
                context_parts.append(f"A: {response_preview}")
                context_parts.append(f"Data Used: {example['data_scope']}\n")
            
            context_parts.append("Follow this format and quality level for your response.\n")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            self.logger.error(f"Error building few-shot context: {str(e)}")
            return ""
    
    def get_quality_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive quality report from rated interactions
        
        Returns:
            Dictionary with quality metrics and insights
        """
        try:
            rated = [c for c in self.qa_conversations if c.rating is not None]
            
            if not rated:
                return {
                    'total_conversations': len(self.qa_conversations),
                    'rated_conversations': 0,
                    'message': 'No rated conversations yet. Start rating responses to track quality!'
                }
            
            ratings = [c.rating for c in rated]
            
            return {
                'total_conversations': len(self.qa_conversations),
                'rated_conversations': len(rated),
                'average_rating': round(float(np.mean(ratings)), 2),
                'median_rating': float(np.median(ratings)),
                'rating_distribution': {
                    str(i): len([r for r in ratings if r == i])
                    for i in range(1, 11)
                },
                'high_quality_count': len([r for r in ratings if r >= 8]),
                'needs_improvement_count': len([r for r in ratings if r <= 5]),
                'quality_by_type': self._calculate_quality_by_type(rated),
                'hallucination_rate': round(sum(1 for c in rated if c.had_hallucination) / len(rated) * 100, 2),
                'attribution_rate': round(sum(1 for c in rated if c.included_data_attribution) / len(rated) * 100, 2),
                'avg_response_length': round(np.mean([c.response_length for c in rated]), 0)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating quality report: {str(e)}")
            return {'error': str(e)}
    
    def get_improvement_suggestions(self) -> List[str]:
        """
        Get actionable suggestions to improve response quality
        
        Returns:
            List of improvement suggestions
        """
        try:
            rated = [c for c in self.qa_conversations if c.rating is not None]
            
            if not rated or len(rated) < 5:
                return ["Need at least 5 rated conversations to generate suggestions"]
            
            suggestions = []
            
            # Check hallucination rate
            hallucination_rate = sum(1 for c in rated if c.had_hallucination) / len(rated)
            if hallucination_rate > 0.2:
                suggestions.append("⚠️ High hallucination rate detected. Ensure responses are based on actual data, not assumptions.")
            
            # Check attribution rate
            attribution_rate = sum(1 for c in rated if c.included_data_attribution) / len(rated)
            if attribution_rate < 0.7:
                suggestions.append("📊 Improve data attribution. Always cite which data source/timeframe was analyzed.")
            
            # Check question specificity
            specificity_rate = sum(1 for c in rated if c.was_question_specific) / len(rated)
            if specificity_rate < 0.8:
                suggestions.append("🎯 Responses should directly address the specific question asked.")
            
            # Check low-rated responses
            low_rated = [c for c in rated if c.rating <= 5]
            if len(low_rated) > len(rated) * 0.3:
                suggestions.append("❗ 30%+ responses rated poorly. Review feedback notes to identify common issues.")
            
            # Question type specific suggestions
            quality_by_type = self._calculate_quality_by_type(rated)
            for qtype, avg_rating in quality_by_type.items():
                if avg_rating < 6:
                    suggestions.append(f"📉 '{qtype}' questions performing poorly (avg: {avg_rating:.1f}/10). Review these responses.")
            
            if not suggestions:
                suggestions.append("✅ Quality metrics look good! Continue current approach.")
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error generating suggestions: {str(e)}")
            return []
    
    def _analyze_response_quality(self, response: str, question: str) -> Dict[str, bool]:
        """Automatically analyze response quality indicators"""
        response_lower = response.lower()
        question_lower = question.lower()
        
        # Hallucination indicators (fabricated monthly patterns)
        hallucination_indicators = [
            'april: ', 'may: ', 'june: ', 'july: ', 'august: ',
            'month-by-month pattern:', 'peak in june', 'spike in may',
            'here\'s the breakdown by month' if 'month' not in question_lower else ''
        ]
        has_hallucination = any(indicator and indicator in response_lower for indicator in hallucination_indicators)
        
        # Question-specific check
        question_keywords = set(question_lower.split()[:15])
        response_start = set(response_lower[:600].split())
        is_question_specific = len(question_keywords & response_start) > 3
        
        # Data attribution check
        attribution_phrases = [
            'based on analysis of', 'analyzed', 'from the data',
            'records show', 'data indicates', 'according to'
        ]
        has_data_attribution = any(phrase in response_lower[:400] for phrase in attribution_phrases)
        
        return {
            'has_hallucination': has_hallucination,
            'is_question_specific': is_question_specific,
            'has_data_attribution': has_data_attribution
        }
    
    def _calculate_quality_by_type(self, conversations: List[ConversationRecord]) -> Dict[str, float]:
        """Calculate average quality by question type"""
        quality_by_type = defaultdict(list)
        
        for conv in conversations:
            if conv.rating:
                quality_by_type[conv.question_type].append(conv.rating)
        
        return {
            qtype: round(float(np.mean(ratings)), 2)
            for qtype, ratings in quality_by_type.items()
        }
    
    def _load_qa_conversations(self):
        """Load Q&A conversation records"""
        try:
            if os.path.exists(self.qa_feedback_path):
                with open(self.qa_feedback_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.qa_conversations = [ConversationRecord(**record) for record in data]
                self.logger.info(f"Loaded {len(self.qa_conversations)} Q&A records")
        except Exception as e:
            self.logger.error(f"Error loading Q&A: {str(e)}")
            self.qa_conversations = []
    
    def _save_qa_conversations(self):
        """Save Q&A conversation records"""
        try:
            with open(self.qa_feedback_path, 'w', encoding='utf-8') as f:
                records = [asdict(conv) for conv in self.qa_conversations]
                json.dump(records, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Error saving Q&A: {str(e)}")
    
    # ==================== ML PATTERN LEARNING ====================
    
    def learn_from_campaign_data(self, campaign_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Learn patterns from campaign data for better insights
        
        Args:
            campaign_data: DataFrame with campaign performance data
            
        Returns:
            Dictionary with learned patterns
        """
        try:
            patterns = {
                'seasonal_patterns': self._detect_seasonal_patterns(campaign_data),
                'performance_clusters': self._cluster_campaign_performance(campaign_data),
                'device_behavior': self._analyze_device_patterns(campaign_data),
                'budget_insights': self._learn_budget_patterns(campaign_data)
            }
            
            self._cache_patterns('campaign', patterns)
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error learning from campaign data: {str(e)}")
            return {}
    
    def learn_from_crm_data(self, crm_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Learn patterns from CRM data for better lead insights
        
        Args:
            crm_data: DataFrame with CRM/leads data
            
        Returns:
            Dictionary with learned CRM patterns
        """
        try:
            patterns = {
                'lead_quality_patterns': self._analyze_lead_quality_patterns(crm_data),
                'geographic_insights': self._learn_geographic_patterns(crm_data),
                'source_performance': self._analyze_source_trends(crm_data),
                'temporal_patterns': self._analyze_temporal_lead_patterns(crm_data)
            }
            
            self._cache_patterns('crm', patterns)
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error learning from CRM data: {str(e)}")
            return {}
    
    def _detect_seasonal_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect seasonal patterns in campaign data"""
        if 'Date' not in data.columns or len(data) < 30:
            return {'detected': False, 'reason': 'Insufficient data'}
        
        data['Date'] = pd.to_datetime(data['Date'])
        data['Month'] = data['Date'].dt.month
        data['DayOfWeek'] = data['Date'].dt.dayofweek
        
        monthly_performance = data.groupby('Month').agg({
            'Conversion Rate': 'mean',
            'Cost': 'sum'
        })
        
        return {
            'detected': True,
            'best_months': monthly_performance.nlargest(3, 'Conversion Rate').index.tolist(),
            'highest_spend_months': monthly_performance.nlargest(3, 'Cost').index.tolist()
        }
    
    def _cluster_campaign_performance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Cluster campaigns by performance"""
        if len(data) < 10:
            return {'clusters': 0}
        
        features = ['CTR %', 'Conversion Rate', 'CPC']
        available = [f for f in features if f in data.columns]
        
        if len(available) < 2:
            return {'clusters': 0}
        
        cluster_data = data[available].fillna(0)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(cluster_data)
        
        n_clusters = min(3, len(data) // 10)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(scaled)
        
        return {'clusters': n_clusters, 'model_saved': True}
    
    def _analyze_device_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze device-specific performance"""
        device_patterns = {}
        
        devices = ['Mobile', 'Desktop', 'Tablet']
        for device in devices:
            cols = [f'{device} Clicks', f'{device} Cost', f'{device} Conversions']
            if all(col in data.columns for col in cols):
                clicks = data[f'{device} Clicks'].sum()
                cost = data[f'{device} Cost'].sum()
                conversions = data[f'{device} Conversions'].sum()
                
                if clicks > 0:
                    device_patterns[device.lower()] = {
                        'avg_cpc': float(cost / clicks),
                        'conversion_rate': float(conversions / clicks) if clicks > 0 else 0
                    }
        
        return device_patterns
    
    def _learn_budget_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Learn budget optimization patterns"""
        if 'Monthly Budget' not in data.columns or 'Cost' not in data.columns:
            return {}
        
        data['Budget_Utilization'] = data['Cost'] / data['Monthly Budget']
        
        return {
            'avg_utilization': float(data['Budget_Utilization'].mean()),
            'optimal_range': '80-100%'
        }
    
    def _analyze_lead_quality_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze lead quality patterns"""
        if 'Score' not in data.columns:
            return {}
        
        return {
            'avg_score': float(data['Score'].mean()),
            'score_range': [float(data['Score'].min()), float(data['Score'].max())]
        }
    
    def _learn_geographic_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Learn geographic patterns"""
        if 'Location' not in data.columns:
            return {}
        
        location_counts = data['Location'].value_counts()
        return {
            'top_locations': location_counts.head(10).to_dict()
        }
    
    def _analyze_source_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze lead source trends"""
        if 'Lead Source' not in data.columns:
            return {}
        
        source_counts = data['Lead Source'].value_counts()
        return {
            'top_sources': source_counts.head(10).to_dict()
        }
    
    def _analyze_temporal_lead_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal lead patterns"""
        if 'Created Time' not in data.columns:
            return {}
        
        data['Created Time'] = pd.to_datetime(data['Created Time'])
        data['Hour'] = data['Created Time'].dt.hour
        
        hourly_leads = data['Hour'].value_counts().sort_index()
        return {
            'peak_hours': hourly_leads.nlargest(3).index.tolist()
        }
    
    def _cache_patterns(self, pattern_type: str, patterns: Dict[str, Any]) -> None:
        """Cache learned patterns"""
        cache_path = f'{self.patterns_path}patterns_{pattern_type}.json'
        try:
            with open(cache_path, 'w') as f:
                json.dump(patterns, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Error caching patterns: {str(e)}")
    
    def classify_question_type(self, question: str) -> str:
        """
        Classify user question into categories for targeted learning
        
        Args:
            question: User's question
            
        Returns:
            Question type string
        """
        question_lower = question.lower()
        
        if any(term in question_lower for term in ['budget', 'spend', 'cost', 'allocation']):
            return 'budget'
        elif any(term in question_lower for term in ['conversion', 'performance', 'roi', 'efficiency']):
            return 'performance'
        elif any(term in question_lower for term in ['lead', 'crm', 'quality', 'source']):
            return 'leads'
        elif any(term in question_lower for term in ['device', 'mobile', 'desktop', 'tablet']):
            return 'device'
        elif any(term in question_lower for term in ['trend', 'month', 'time', 'season']):
            return 'trends'
        elif any(term in question_lower for term in ['recommend', 'suggest', 'optimize', 'improve']):
            return 'recommendations'
        else:
            return 'general'