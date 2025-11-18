"""
MIT Executive Pipeline Report - Data Collection Module
======================================================

This module collects and processes data for the executive print report.
The report provides a simplified, print-optimized view of the MIT pipeline
with performance insights and critical window mentors.

Author: AI Assistant
Date: December 2024
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
from config import *
from utils import (
    merge_candidate_sources,
    categorize_candidates_by_week,
    fetch_open_positions_data,
    get_active_training_mentors,
    calculate_avg_onboarding_completion,
    calculate_avg_lessons_completion,
    calculate_avg_qbr_score,
    calculate_avg_perf_score,
    log_error,
    fetch_meeting_insights
)

logger = logging.getLogger(__name__)

# Dashboard base URL for hyperlinks
DASHBOARD_BASE_URL = "https://mit-training-dashboard-dd693bfc9f5a.herokuapp.com"

# =============================================================================
# CRITICAL WINDOW MENTORS
# =============================================================================

def get_critical_window_mentors(candidates: List[Dict], active_mentors: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Get mentors who have MITs in Week 6-7 (critical placement window)
    
    Args:
        candidates: List of all candidate dictionaries
        active_mentors: Dictionary from get_active_training_mentors()
    
    Returns:
        List of mentors with MITs in critical window, sorted by number of critical MITs
    """
    # Get all mentors with their trainees
    all_mentors = {}
    mentors_list = active_mentors.get('strong_mentors', []) + \
                   active_mentors.get('monitoring_mentors', []) + \
                   active_mentors.get('needs_help_mentors', [])
    
    for mentor in mentors_list:
        mentor_name = mentor.get('name', 'Unknown')
        all_mentors[mentor_name] = {
            'name': mentor_name,
            'critical_mits': [],
            'critical_count': 0
        }
    
    # Find MITs in Week 6-7 and group by mentor
    for candidate in candidates:
        week = candidate.get('week', 0)
        mentor_name = candidate.get('mentor_name', '')
        status = str(candidate.get('status', '')).lower()
        
        if 'pending' in status:
            continue
            
        if 6 <= week <= 7 and mentor_name and mentor_name in all_mentors:
            all_mentors[mentor_name]['critical_mits'].append({
                'name': candidate.get('name', 'Unknown'),
                'week': week,
                'location': candidate.get('training_site', 'TBD')
            })
            all_mentors[mentor_name]['critical_count'] += 1
    
    # Filter to only mentors with critical MITs and sort
    critical_mentors = [m for m in all_mentors.values() if m['critical_count'] > 0]
    critical_mentors.sort(key=lambda x: x['critical_count'], reverse=True)
    
    return critical_mentors

# =============================================================================
# TRAINING INDICATORS FILTERING
# =============================================================================

def filter_training_indicators_by_assessments(candidates: List[Dict]) -> Dict[str, Any]:
    """
    Calculate training indicators only for MITs who completed assessments
    
    Args:
        candidates: List of all candidate dictionaries
    
    Returns:
        Dictionary with filtered training indicators
    """
    # Filter to only candidates with assessment scores
    candidates_with_assessments = [
        c for c in candidates 
        if c.get('scores', {}).get('assessment_score', 0) and 
           c.get('scores', {}).get('assessment_score', 0) > 0
    ]
    
    if not candidates_with_assessments:
        return {
            'onboarding_completion': 0.0,
            'lessons_completion': 0.0,
            'avg_qbr_score': None,
            'avg_perf_score': None,
            'candidates_count': 0
        }
    
    onboarding_completion = calculate_avg_onboarding_completion(candidates_with_assessments)
    lessons_completion = calculate_avg_lessons_completion(candidates_with_assessments)
    avg_qbr = calculate_avg_qbr_score(candidates_with_assessments)
    avg_perf = calculate_avg_perf_score(candidates_with_assessments)
    
    return {
        'onboarding_completion': onboarding_completion,
        'lessons_completion': lessons_completion,
        'avg_qbr_score': avg_qbr,
        'avg_perf_score': avg_perf,
        'candidates_count': len(candidates_with_assessments)
    }

# =============================================================================
# PERFORMANCE INSIGHTS
# =============================================================================

def generate_performance_insights(candidates: List[Dict], training_indicators: Dict[str, Any]) -> List[str]:
    """
    Generate 3-5 performance insights based on rule-based logic
    
    Args:
        candidates: List of all candidate dictionaries
        training_indicators: Dictionary from filter_training_indicators_by_assessments()
    
    Returns:
        List of insight strings (3-5 bullets)
    """
    insights = []
    
    # 1. Critical Window MITs (Week 6-7)
    critical_window = [c for c in candidates if 6 <= c.get('week', 0) <= 7]
    critical_count = len([c for c in critical_window if 'pending' not in str(c.get('status', '')).lower()])
    if critical_count > 0:
        insights.append(f"<strong>{critical_count} MIT(s)</strong> in critical placement window (Weeks 6-7) requiring immediate placement attention.")
    
    # 2. Lesson Completion Gaps
    lessons_pct = training_indicators.get('lessons_completion', 0)
    if lessons_pct < 50:
        insights.append(f"Business lessons completion at <strong>{lessons_pct:.1f}%</strong> — below target threshold. Accelerated lesson completion needed.")
    elif lessons_pct < 70:
        insights.append(f"Business lessons completion at <strong>{lessons_pct:.1f}%</strong> — approaching target but needs improvement.")
    
    # 3. Strong QBR Performers
    avg_qbr = training_indicators.get('avg_qbr_score')
    strong_qbr = [c for c in candidates 
                  if c.get('scores', {}).get('mock_qbr_score', 0) and 
                     c.get('scores', {}).get('mock_qbr_score', 0) >= 3.0]
    strong_qbr_count = len(strong_qbr)
    if strong_qbr_count > 0:
        insights.append(f"<strong>{strong_qbr_count} MIT(s)</strong> scoring 3.0+ on Mock QBR — strong performers ready for placement priority.")
    
    if avg_qbr and avg_qbr >= 3.0:
        insights.append(f"Average Mock QBR score at <strong>{avg_qbr:.2f}/4.0</strong> — above target, indicating strong program performance.")
    
    # 4. Extended Week MITs (Week 8+)
    extended = [c for c in candidates if c.get('week', 0) >= 8]
    extended_count = len([c for c in extended if 'pending' not in str(c.get('status', '')).lower()])
    if extended_count > 0:
        insights.append(f"<strong>{extended_count} MIT(s)</strong> at Week 8+ — placement ready and awaiting assignment.")
    
    # 5. Onboarding Trends
    onboarding_pct = training_indicators.get('onboarding_completion', 0)
    if onboarding_pct >= 80:
        insights.append(f"Onboarding completion at <strong>{onboarding_pct:.1f}%</strong> — strong onboarding execution.")
    elif onboarding_pct < 60:
        insights.append(f"Onboarding completion at <strong>{onboarding_pct:.1f}%</strong> — below target, requires immediate action.")
    
    # Ensure we have 3-5 insights
    if len(insights) < 3:
        # Add general insights if needed
        total_active = len([c for c in candidates if 'pending' not in str(c.get('status', '')).lower()])
        insights.append(f"Total active pipeline: <strong>{total_active} MIT(s)</strong> in training across all week bands.")
    
    # Return top 5 insights
    return insights[:5]

# =============================================================================
# MAIN DATA COLLECTION
# =============================================================================

def collect_report_data() -> Dict[str, Any]:
    """
    Collect all data needed for the executive report
    
    Returns:
        Dictionary containing all report data
    """
    logger.info("Collecting data for executive report...")
    
    # Get all candidates
    candidates = merge_candidate_sources()
    
    # Categorize by week
    categorized = categorize_candidates_by_week(candidates)
    
    # Get open positions
    open_positions = fetch_open_positions_data()
    
    # Get mentor data
    active_mentors = get_active_training_mentors()
    
    # Calculate metrics
    total_mits = len(candidates)
    active_mits = len([c for c in candidates if 'pending' not in str(c.get('status', '')).lower()])
    ready_mits = len(categorized.get('weeks_8_plus', []))
    
    # Calculate readiness (next 30 days) - weeks 5-7
    ready_in_30_days = []
    for candidate in candidates:
        week = candidate.get('week', 0)
        status = str(candidate.get('status', '')).lower()
        if 5 <= week <= 7 and 'pending' not in status:
            ready_in_30_days.append({
                'name': candidate.get('name', 'Unknown'),
                'week': week,
                'location': candidate.get('training_site', 'TBD')
            })
    
    # Group MITs by week bands
    mits_by_week_band = {
        'weeks_0_3': [],
        'weeks_4_6': [],
        'week_7': [],
        'weeks_8_plus': []
    }
    
    for candidate in candidates:
        week = candidate.get('week', 0)
        status = str(candidate.get('status', '')).lower()
        
        if 'pending' in status:
            continue
            
        mit_data = {
            'name': candidate.get('name', 'Unknown'),
            'location': candidate.get('training_site', 'TBD'),
            'vertical': candidate.get('operation_details', {}).get('vertical', 'TBD'),
            'week': week,
            'mentor': candidate.get('mentor_name', 'TBD'),
            'status': candidate.get('status', 'TBD')
        }
        
        if 0 <= week <= 3:
            mits_by_week_band['weeks_0_3'].append(mit_data)
        elif 4 <= week <= 6:
            mits_by_week_band['weeks_4_6'].append(mit_data)
        elif week == 7:
            mits_by_week_band['week_7'].append(mit_data)
        elif week >= 8:
            mits_by_week_band['weeks_8_plus'].append(mit_data)
    
    # Sort each band
    for band in mits_by_week_band.values():
        band.sort(key=lambda x: (x['week'], x['name']))
    
    # Get critical window mentors (Week 6-7)
    critical_window_mentors = get_critical_window_mentors(candidates, active_mentors)
    
    # Filter training indicators to only MITs with assessments
    training_indicators = filter_training_indicators_by_assessments(candidates)
    
    # Generate performance insights
    performance_insights = generate_performance_insights(candidates, training_indicators)
    
    # Get weekly meeting insights
    meeting_insights = fetch_meeting_insights()
    
    # Build all MITs list for appendix
    all_mits = []
    for candidate in candidates:
        status = str(candidate.get('status', '')).lower()
        if 'pending' not in status:
            all_mits.append({
                'name': candidate.get('name', 'Unknown'),
                'vertical': candidate.get('operation_details', {}).get('vertical', 'TBD'),
                'week': candidate.get('week', 0),
                'location': candidate.get('training_site', 'TBD'),
                'mentor': candidate.get('mentor_name', 'TBD'),
                'status': candidate.get('status', 'TBD')
            })
    
    all_mits.sort(key=lambda x: (x['week'], x['name']))
    
    # Build dashboard URLs for hyperlinks
    dashboard_urls = {
        'active_mits': f"{DASHBOARD_BASE_URL}/#/in-training",
        'placement_ready': f"{DASHBOARD_BASE_URL}/#/candidates/weeks-8-plus",
        'at_risk': f"{DASHBOARD_BASE_URL}/#/candidates/week-7-priority",
        'open_roles': f"{DASHBOARD_BASE_URL}/#/open-positions"
    }
    
    return {
        'timestamp': datetime.now().strftime('%B %d, %Y at %I:%M %p'),
        'dashboard_url': DASHBOARD_BASE_URL,
        'dashboard_urls': dashboard_urls,
        'total_mits': total_mits,
        'active_mits': active_mits,
        'ready_mits': ready_mits,
        'open_positions_count': len(open_positions),
        'ready_in_30_days': ready_in_30_days,
        'mits_by_week_band': mits_by_week_band,
        'critical_window_mentors': critical_window_mentors,
        'training_indicators': training_indicators,
        'performance_insights': performance_insights,
        'meeting_insights': meeting_insights,
        'all_mits': all_mits
    }

