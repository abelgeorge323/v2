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
    group_positions_by_region,
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

def generate_performance_insights(candidates: List[Dict], training_indicators: Dict[str, Any], mits_by_week_band: Dict = None) -> List[str]:
    """
    Generate 3-5 performance insights based on rule-based logic
    
    Args:
        candidates: List of all candidate dictionaries
        training_indicators: Dictionary from filter_training_indicators_by_assessments()
        mits_by_week_band: Optional pre-computed week bands
    
    Returns:
        List of insight strings (3-5 bullets)
    """
    insights = []
    
    # 1. Critical Window MITs (Week 6-7)
    critical_window = [c for c in candidates if 6 <= c.get('week', 0) <= 7]
    critical_count = len([c for c in critical_window if 'pending' not in str(c.get('status', '')).lower()])
    if critical_count > 0:
        insights.append(f"🚨 <strong>{critical_count} MIT(s)</strong> in critical placement window (Weeks 6-7) — immediate placement attention required.")
    
    # 2. Extended Week MITs (Week 8+) - URGENT
    extended = [c for c in candidates if c.get('week', 0) >= 8]
    extended_count = len([c for c in extended if 'pending' not in str(c.get('status', '')).lower()])
    if extended_count > 0:
        max_week = max([c.get('week', 0) for c in extended])
        if max_week >= 12:
            insights.append(f"⚠️ <strong>{extended_count} MIT(s)</strong> at Week 8+ (longest: Week {max_week}) — escalate placement urgency.")
        else:
            insights.append(f"✅ <strong>{extended_count} MIT(s)</strong> at Week 8+ — placement ready and awaiting assignment.")
    
    # 3. New Starts needing first survey
    new_starts = [c for c in candidates if 0 <= c.get('week', 0) <= 3 and 'pending' not in str(c.get('status', '')).lower()]
    if new_starts:
        insights.append(f"🌱 <strong>{len(new_starts)} MIT(s)</strong> in onboarding phase (Weeks 0-3) — monitor first mentor surveys.")
    
    # 4. Strong QBR Performers
    strong_qbr = [c for c in candidates 
                  if c.get('scores', {}).get('mock_qbr_score', 0) and 
                     c.get('scores', {}).get('mock_qbr_score', 0) >= 3.0]
    strong_qbr_count = len(strong_qbr)
    if strong_qbr_count > 0:
        insights.append(f"⭐ <strong>{strong_qbr_count} MIT(s)</strong> scoring 3.0+ on Mock QBR — top performers ready for priority placement.")
    
    # 5. Onboarding Trends
    onboarding_pct = training_indicators.get('onboarding_completion', 0)
    if onboarding_pct >= 80:
        insights.append(f"📋 Onboarding completion at <strong>{onboarding_pct:.1f}%</strong> — strong execution.")
    elif onboarding_pct < 60:
        insights.append(f"📋 Onboarding completion at <strong>{onboarding_pct:.1f}%</strong> — needs improvement.")
    
    # Ensure we have 3-5 insights
    if len(insights) < 3:
        total_active = len([c for c in candidates if 'pending' not in str(c.get('status', '')).lower()])
        insights.append(f"📊 Total active pipeline: <strong>{total_active} MIT(s)</strong> in training.")
    
    return insights[:5]

# =============================================================================
# MENTOR ASSESSMENT PROCESSING
# =============================================================================

# MITs to exclude from assessment calculations
EXCLUDED_MITS = [
    'kathryn keillor',  # Exclude unfairly judged assessments
    'ivves mullen',     # Placed/removed from program
    'shaquille thompson',  # Placed/removed from program
    'agrein turner'  # Removed from program for ethics-related issues
]

# Threshold for showing bottom 2 MITs (only show if scores are below this)
POOR_SCORE_THRESHOLD = 3.5

def load_mentor_assessments() -> Dict[str, Any]:
    """
    Load mentor_assessments.json safely using json.load
    
    Returns:
        Dictionary keyed by MIT name
    """
    try:
        import json
        import os
        
        # Get the path to mentor_assessments.json (same approach as bios.json)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        assessments_path = os.path.join(current_dir, 'data', 'mentor_assessments.json')
        
        # If file doesn't exist, try alternative path (handles nested folder structure)
        if not os.path.exists(assessments_path):
            assessments_path = os.path.join(os.path.dirname(current_dir), 'data', 'mentor_assessments.json')
        
        logger.info(f"Loading mentor assessments from: {assessments_path}")
        
        with open(assessments_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.info(f"Successfully loaded {len(data)} MIT assessments")
            return data
    except FileNotFoundError as e:
        logger.error(f"Mentor assessments file not found. Tried: {assessments_path}")
        logger.error(f"Error: {str(e)}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in mentor assessments file: {str(e)}")
        return {}
    except Exception as e:
        logger.error(f"Error loading mentor assessments: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {}

def compute_section_averages(assessment: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute section averages and overall score from an assessment
    
    Args:
        assessment: Single assessment dictionary
        
    Returns:
        Dictionary with section averages and overall score
    """
    try:
        # Section 1: Core Competencies
        section_1_questions = assessment.get('section_1_core_competencies', {}).get('questions', {})
        section_1_values = [v for v in section_1_questions.values() if isinstance(v, (int, float))]
        section_1_avg = sum(section_1_values) / len(section_1_values) if section_1_values else 0.0
        
        # Section 2: Soft Skills & Leadership
        section_2_questions = assessment.get('section_2_soft_skills_leadership', {}).get('questions', {})
        section_2_values = [v for v in section_2_questions.values() if isinstance(v, (int, float))]
        section_2_avg = sum(section_2_values) / len(section_2_values) if section_2_values else 0.0
        
        # Section 3: Engagement & Aptitude
        section_3_questions = assessment.get('section_3_engagement_aptitude', {}).get('questions', {})
        section_3_values = [v for v in section_3_questions.values() if isinstance(v, (int, float))]
        section_3_avg = sum(section_3_values) / len(section_3_values) if section_3_values else 0.0
        
        # Section 4: Readiness & Confidence (use confidence_level)
        section_4_data = assessment.get('section_4_readiness_confidence', {})
        section_4_avg = float(section_4_data.get('confidence_level', 0))
        
        # Overall = average of all four sections
        overall = (section_1_avg + section_2_avg + section_3_avg + section_4_avg) / 4.0
        
        return {
            'section_1': round(section_1_avg, 2),
            'section_2': round(section_2_avg, 2),
            'section_3': round(section_3_avg, 2),
            'section_4': round(section_4_avg, 2),
            'overall': round(overall, 2)
        }
    except Exception as e:
        logger.error(f"Error computing section averages: {str(e)}")
        return {
            'section_1': 0.0,
            'section_2': 0.0,
            'section_3': 0.0,
            'section_4': 0.0,
            'overall': 0.0
        }

def build_mit_assessment_summary() -> List[Dict[str, Any]]:
    """
    Build summary of all MIT assessments, excluding specified MITs
    
    Returns:
        List of MIT assessment summaries, sorted by overall score descending
    """
    assessments_data = load_mentor_assessments()
    logger.info(f"Loaded {len(assessments_data)} MITs from assessments file")
    
    summary_list = []
    
    excluded_normalized = [e.lower().strip() for e in EXCLUDED_MITS]
    
    for mit_name_raw, mit_data in assessments_data.items():
        mit_name = mit_name_raw.lower().strip()
        
        # Skip excluded MITs
        if mit_name in excluded_normalized:
            logger.info(f"Excluding {mit_name_raw} from assessment calculations")
            continue
        
        assessments = mit_data.get('assessments', [])
        if not assessments:
            logger.debug(f"Skipping {mit_name_raw} - no assessments")
            continue
        
        # Use the most recent assessment (last one in the list)
        latest_assessment = assessments[-1]
        
        # Compute section averages
        scores = compute_section_averages(latest_assessment)
        
        # Get observations
        observations = latest_assessment.get('section_4_readiness_confidence', {}).get('observations', 'No observations provided.')
        
        summary_list.append({
            'name': mit_name_raw.title(),  # Capitalize properly
            'section_1': scores['section_1'],
            'section_2': scores['section_2'],
            'section_3': scores['section_3'],
            'section_4': scores['section_4'],
            'overall': scores['overall'],
            'observations': observations
        })
        logger.debug(f"Added assessment for {mit_name_raw}: overall={scores['overall']}")
    
    # Sort by overall score descending
    summary_list.sort(key=lambda x: x['overall'], reverse=True)
    
    logger.info(f"Built assessment summary for {len(summary_list)} MITs")
    return summary_list

def generate_assessment_executive_summary(top_2: List[Dict], bottom_2: List[Dict]) -> str:
    """
    Generate executive summary text from top and bottom MITs
    
    Args:
        top_2: List of top 2 MIT assessment summaries
        bottom_2: List of bottom 2 MIT assessment summaries
        
    Returns:
        Formatted executive summary string
    """
    if not top_2 or not bottom_2:
        return "Insufficient assessment data available for executive summary."
    
    # Calculate ranges
    top_scores = [mit['overall'] for mit in top_2]
    bottom_scores = [mit['overall'] for mit in bottom_2]
    
    top_range_low = min(top_scores)
    top_range_high = max(top_scores)
    bottom_range_low = min(bottom_scores)
    bottom_range_high = max(bottom_scores)
    
    # Find highest-scoring section among top 2
    top_section_1_avg = sum([mit['section_1'] for mit in top_2]) / len(top_2)
    top_section_2_avg = sum([mit['section_2'] for mit in top_2]) / len(top_2)
    top_section_3_avg = sum([mit['section_3'] for mit in top_2]) / len(top_2)
    top_section_4_avg = sum([mit['section_4'] for mit in top_2]) / len(top_2)
    
    top_sections = {
        'Core Competencies': top_section_1_avg,
        'Soft Skills and Leadership': top_section_2_avg,
        'Engagement and Aptitude': top_section_3_avg,
        'Readiness and Confidence': top_section_4_avg
    }
    
    top_best_section_name = max(top_sections, key=top_sections.get)
    top_best_section_score = top_sections[top_best_section_name]
    
    # Find lowest-scoring section among bottom 2
    bottom_section_1_avg = sum([mit['section_1'] for mit in bottom_2]) / len(bottom_2)
    bottom_section_2_avg = sum([mit['section_2'] for mit in bottom_2]) / len(bottom_2)
    bottom_section_3_avg = sum([mit['section_3'] for mit in bottom_2]) / len(bottom_2)
    bottom_section_4_avg = sum([mit['section_4'] for mit in bottom_2]) / len(bottom_2)
    
    bottom_sections = {
        'Core Competencies': bottom_section_1_avg,
        'Soft Skills and Leadership': bottom_section_2_avg,
        'Engagement and Aptitude': bottom_section_3_avg,
        'Readiness and Confidence': bottom_section_4_avg
    }
    
    bottom_low_section_name = min(bottom_sections, key=bottom_sections.get)
    bottom_low_section_score = bottom_sections[bottom_low_section_name]
    
    # Format ranges
    top_best_range = f"{top_best_section_score:.1f}"
    bottom_low_range = f"{bottom_low_section_score:.1f}"
    
    # Build summary text
    summary = f"Top MITs averaged {top_range_low:.1f}–{top_range_high:.1f} overall, with highest scores in {top_best_section_name} ({top_best_range}). Lower-scoring MITs averaged {bottom_range_low:.1f}–{bottom_range_high:.1f}, with the lowest ratings concentrated in {bottom_low_section_name} ({bottom_low_range})."
    
    return summary

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
    
    # Get open positions and group by region
    open_positions = fetch_open_positions_data()
    positions_by_region = group_positions_by_region(open_positions)
    
    # Calculate job openings summary statistics
    total_positions = len(open_positions)
    salaries = [p.get('salary', 0) for p in open_positions if isinstance(p.get('salary'), (int, float)) and p.get('salary', 0) > 0]
    avg_salary = sum(salaries) / len(salaries) if salaries else 0
    
    # Calculate vertical distribution
    vertical_counts = {}
    for pos in open_positions:
        vertical = pos.get('vertical', 'Unknown').strip()
        if vertical:
            vertical_counts[vertical] = vertical_counts.get(vertical, 0) + 1
    
    # Get top 3 verticals
    top_verticals = sorted(vertical_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    
    # Sort positions by salary (highest first) within each region
    for region in positions_by_region:
        positions_by_region[region].sort(key=lambda p: p.get('salary', 0) if isinstance(p.get('salary'), (int, float)) else 0, reverse=True)
    
    # Get mentor data
    active_mentors = get_active_training_mentors()
    
    # Calculate metrics
    # Note: active_mits will be calculated after week band categorization to ensure consistency
    total_mits = len(candidates)
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
    
    # Load mentor assessments for integration
    mentor_assessments = load_mentor_assessments()
    
    # Helper to get latest mentor score for a candidate
    def get_mentor_score(name):
        normalized = name.lower().strip().replace("'", "").replace("-", " ")
        for key in mentor_assessments.keys():
            if key.lower().strip() == normalized or normalized in key.lower():
                assessments = mentor_assessments[key].get('assessments', [])
                if assessments:
                    latest = assessments[-1]  # Get latest
                    scores = compute_section_averages(latest)
                    return scores.get('overall', 0)
        return None
    
    # Helper to get mentor observations
    def get_mentor_observations(name):
        normalized = name.lower().strip().replace("'", "").replace("-", " ")
        for key in mentor_assessments.keys():
            if key.lower().strip() == normalized or normalized in key.lower():
                assessments = mentor_assessments[key].get('assessments', [])
                if assessments:
                    latest = assessments[-1]
                    return latest.get('section_4_readiness_confidence', {}).get('observations', '')
        return ''
    
    # Group MITs by NEW week bands: 0-3 Onboarding, 4-5 Mid, 6-7 Critical, 8+ Ready, Pending
    mits_by_week_band = {
        'weeks_0_3': [],      # Onboarding / New Starts
        'weeks_4_5': [],      # Mid-Training
        'weeks_6_7': [],      # Critical Window (placement priority)
        'weeks_8_plus': [],   # Placement Ready
        'pending_mits': []    # Incoming MITs (offers accepted, not started)
    }
    
    for candidate in candidates:
        week = candidate.get('week', 0)
        status = str(candidate.get('status', '')).lower()
        
        # Check if incoming MIT who hasn't started yet (week = 0 with pending/offer status)
        is_pending_status = 'pending' in status or 'offer' in status
        has_not_started = week == 0 or week is None or (isinstance(week, str) and week.lower() in ['n/a', 'na', ''])
        
        if is_pending_status and has_not_started:
            # Add to pending MITs section
            pending_data = {
                'name': candidate.get('name', 'Unknown'),
                'location': candidate.get('training_site', 'TBD'),
                'vertical': candidate.get('operation_details', {}).get('vertical', 'TBD'),
                'mentor': candidate.get('mentor_name', 'TBD'),
                'status': candidate.get('status', 'TBD'),
                'expected_start': candidate.get('operation_details', {}).get('training_start_date', 'TBD')
            }
            mits_by_week_band['pending_mits'].append(pending_data)
            continue  # Skip the rest of the processing for pending MITs
        
        # Get mentor score and determine status indicator
        mentor_score = get_mentor_score(candidate.get('name', ''))
        mentor_obs = get_mentor_observations(candidate.get('name', ''))
        
        # Determine readiness status based on mentor score
        if mentor_score:
            if mentor_score >= 4.5:
                readiness_status = 'Strong'
            elif mentor_score >= 3.5:
                readiness_status = 'On Track'
            else:
                readiness_status = 'Monitor'
        else:
            readiness_status = 'Pending'
            
        # Get all Performance Snapshot scores
        scores = candidate.get('scores', {})
        skill_ranking = scores.get('skill_ranking', 'TBD')
        assessment_score = scores.get('assessment_score', 0)
        confidence_score = scores.get('confidence_score', 0)
        perf_eval_score = scores.get('perf_eval_score', 0)
        
        # Determine primary score: Skill Ranking if available, otherwise Mentor Score
        primary_score_type = 'Skill Ranking' if skill_ranking and skill_ranking != 'TBD' and str(skill_ranking).strip().lower() not in ['nan', 'none', ''] else 'Mentor Score'
        primary_score_value = skill_ranking if primary_score_type == 'Skill Ranking' else mentor_score
        
        mit_data = {
            'name': candidate.get('name', 'Unknown'),
            'location': candidate.get('training_site', 'TBD'),
            'vertical': candidate.get('operation_details', {}).get('vertical', 'TBD'),
            'week': week,
            'mentor': candidate.get('mentor_name', 'TBD'),
            'status': candidate.get('status', 'TBD'),
            'mock_qbr_date': candidate.get('mock_qbr_date', ''),
            'mock_qbr_score': scores.get('mock_qbr_score', ''),
            'mentor_score': mentor_score,
            'skill_ranking': skill_ranking,
            'assessment_score': assessment_score,
            'confidence_score': confidence_score,
            'perf_eval_score': perf_eval_score,
            'primary_score_type': primary_score_type,
            'primary_score_value': primary_score_value,
            'mentor_observations': mentor_obs[:150] + '...' if len(mentor_obs) > 150 else mentor_obs,
            'readiness_status': readiness_status,
            'oig_status': candidate.get('oig_completion', {}).get('status', 'Unknown'),
            'first_survey_submitted': mentor_score is not None
        }
        
        if 0 <= week <= 3:
            mits_by_week_band['weeks_0_3'].append(mit_data)
        elif 4 <= week <= 5:
            mits_by_week_band['weeks_4_5'].append(mit_data)
        elif 6 <= week <= 7:
            mits_by_week_band['weeks_6_7'].append(mit_data)
        elif week >= 8:
            mits_by_week_band['weeks_8_plus'].append(mit_data)
    
    # Sort each band by week then name (except pending_mits which don't have week field)
    for band_name, band_list in mits_by_week_band.items():
        if band_name == 'pending_mits':
            # Sort pending MITs by name only
            band_list.sort(key=lambda x: x['name'])
        else:
            # Sort by week then name for active MITs
            band_list.sort(key=lambda x: (x['week'], x['name']))
    
    # Get critical window mentors (Week 6-7)
    critical_window_mentors = get_critical_window_mentors(candidates, active_mentors)
    
    # Build upcoming Mock QBR schedule (only FUTURE dates)
    upcoming_mock_qbrs = []
    today = datetime.now().date()
    
    for candidate in candidates:
        mock_qbr_date = candidate.get('mock_qbr_date', '')
        if mock_qbr_date and str(mock_qbr_date).strip() and str(mock_qbr_date).lower() not in ['nan', 'none', '']:
            # Try to parse the date and check if it's in the future
            try:
                import pandas as pd
                parsed_date = pd.to_datetime(str(mock_qbr_date), errors='coerce')
                if pd.notna(parsed_date) and parsed_date.date() >= today:
                    upcoming_mock_qbrs.append({
                        'name': candidate.get('name', 'Unknown'),
                        'date': parsed_date.strftime('%m/%d/%Y'),
                        'week': candidate.get('week', 0)
                    })
            except:
                pass  # Skip if date can't be parsed
    
    # Sort by date
    upcoming_mock_qbrs.sort(key=lambda x: x['date'])
    
    # Filter training indicators to only MITs with assessments
    training_indicators = filter_training_indicators_by_assessments(candidates)
    
    # Generate performance insights with week band data
    performance_insights = generate_performance_insights(candidates, training_indicators, mits_by_week_band)
    
    # Get weekly meeting insights
    meeting_insights = fetch_meeting_insights()
    
    # Build mentor assessment summary
    assessment_data = build_mit_assessment_summary()
    
    # Get top 2 and bottom 2 MITs
    top_2_mits = assessment_data[:2] if len(assessment_data) >= 2 else assessment_data
    
    # Only include MITs that are below the threshold
    bottom_2_mits = []
    if len(assessment_data) >= 2:
        bottom_2 = assessment_data[-2:]
        # Only show MITs that have a score below threshold
        bottom_2_mits = [mit for mit in bottom_2 if mit['overall'] < POOR_SCORE_THRESHOLD]
    
    # Manually add Evan Tichenor and Lloyd Harrison-Hine to Development Focus
    # (Evan: Mock QBR redo, scored low; Lloyd: scoring low on everything)
    manual_development_focus = ['evan tichenor', 'lloyd harrison hine']
    
    for mit_name in manual_development_focus:
        # Find in assessment_data
        found_mit = None
        for mit in assessment_data:
            if mit['name'].lower().strip().replace("'", "").replace("-", " ") == mit_name.lower().strip().replace("'", "").replace("-", " "):
                found_mit = mit.copy()
                break
        
        # If not found in assessment_data, create from candidate data
        if not found_mit:
            for candidate in candidates:
                candidate_name_normalized = candidate.get('name', '').lower().strip().replace("'", "").replace("-", " ")
                if candidate_name_normalized == mit_name.lower().strip().replace("'", "").replace("-", " "):
                    # Create a basic assessment entry from candidate data
                    mentor_score = get_mentor_score(candidate.get('name', ''))
                    mentor_obs = get_mentor_observations(candidate.get('name', ''))
                    # Always add them even if no mentor score (they're manually flagged)
                    found_mit = {
                        'name': candidate.get('name', 'Unknown'),
                        'section_1': 0.0,
                        'section_2': 0.0,
                        'section_3': 0.0,
                        'section_4': mentor_score if mentor_score else 0.0,
                        'overall': mentor_score if mentor_score else 0.0,
                        'observations': mentor_obs if mentor_obs else ''
                    }
                    break
        
        # Add to bottom_2_mits if found and not already included
        if found_mit:
            # Check if already in list
            already_included = any(
                mit['name'].lower().strip().replace("'", "").replace("-", " ") == found_mit['name'].lower().strip().replace("'", "").replace("-", " ")
                for mit in bottom_2_mits
            )
            if not already_included:
                bottom_2_mits.append(found_mit)
    
    # Enrich bottom_2_mits with Performance Snapshot scores from candidate data
    # Create a lookup dictionary for candidate scores by normalized name
    candidate_scores_lookup = {}
    for candidate in candidates:
        candidate_name = candidate.get('name', '')
        normalized_name = candidate_name.lower().strip().replace("'", "").replace("-", " ")
        candidate_scores_lookup[normalized_name] = {
            'mock_qbr_score': candidate.get('scores', {}).get('mock_qbr_score', 0),
            'assessment_score': candidate.get('scores', {}).get('assessment_score', 0),
            'confidence_score': candidate.get('scores', {}).get('confidence_score', 0),
            'perf_eval_score': candidate.get('scores', {}).get('perf_eval_score', 0),
            'skill_ranking': candidate.get('scores', {}).get('skill_ranking', 'TBD')
        }
    
    # Add Performance Snapshot scores to bottom_2_mits
    for mit in bottom_2_mits:
        mit_name_normalized = mit['name'].lower().strip().replace("'", "").replace("-", " ")
        if mit_name_normalized in candidate_scores_lookup:
            mit.update(candidate_scores_lookup[mit_name_normalized])
        else:
            # Default values if candidate not found
            mit['mock_qbr_score'] = 0
            mit['assessment_score'] = 0
            mit['confidence_score'] = 0
            mit['perf_eval_score'] = 0
            mit['skill_ranking'] = 'TBD'
    
    # Generate assessment executive summary
    if len(top_2_mits) >= 2 and len(bottom_2_mits) >= 1:
        assessment_summary = generate_assessment_executive_summary(top_2_mits, bottom_2_mits)
    elif len(top_2_mits) >= 2:
        assessment_summary = f"Top MITs averaged {top_2_mits[0]['overall']:.1f}–{top_2_mits[1]['overall']:.1f} overall. All remaining MITs scored above {POOR_SCORE_THRESHOLD}, indicating strong overall performance."
    else:
        assessment_summary = "Insufficient assessment data available."
    
    # Build all MITs list for appendix
    # Include all candidates except those who haven't started (week 0/N/A with pending/offer status)
    all_mits = []
    for candidate in candidates:
        status = str(candidate.get('status', '')).lower()
        week = candidate.get('week', 0)
        
        # Only exclude incoming MITs who haven't started (same logic as week categorization)
        is_pending_status = 'pending' in status or 'offer' in status
        has_not_started = week == 0 or week is None or (isinstance(week, str) and week.lower() in ['n/a', 'na', ''])
        
        # Include all candidates who have started (week >= 1) or don't match the incoming MIT criteria
        if not (is_pending_status and has_not_started):
            # Get scores with priority: Mock QBR score > Mentor score > Assessment score > Mock QBR date
            mock_qbr_score = candidate.get('scores', {}).get('mock_qbr_score', 0)
            assessment_score = candidate.get('scores', {}).get('assessment_score', 0)
            mentor_score = get_mentor_score(candidate.get('name', ''))
            mock_qbr_date = candidate.get('mock_qbr_date', '')
            
            # Determine what to show: Mock QBR score, then mentor score, then assessment score, then date
            score_display = None
            if mock_qbr_score and mock_qbr_score > 0:
                score_display = f"QBR: {mock_qbr_score}/4"
            elif mentor_score and mentor_score > 0:
                score_display = f"Mentor: {mentor_score:.1f}/5"
            elif assessment_score and assessment_score > 0:
                score_display = f"Assessment: {assessment_score}/100"
            elif mock_qbr_date and str(mock_qbr_date).strip() and str(mock_qbr_date).lower() not in ['nan', 'none', '']:
                # Format the date nicely
                try:
                    import pandas as pd
                    parsed_date = pd.to_datetime(str(mock_qbr_date), errors='coerce')
                    if pd.notna(parsed_date):
                        score_display = parsed_date.strftime('%m/%d/%Y')
                    else:
                        score_display = str(mock_qbr_date)
                except:
                    score_display = str(mock_qbr_date)
            
            all_mits.append({
                'name': candidate.get('name', 'Unknown'),
                'vertical': candidate.get('operation_details', {}).get('vertical', 'TBD'),
                'week': candidate.get('week', 0),
                'location': candidate.get('training_site', 'TBD'),
                'mentor': candidate.get('mentor_name', 'TBD'),
                'status': candidate.get('status', 'TBD'),
                'scores': score_display or 'TBD'
            })
    
    all_mits.sort(key=lambda x: (x['week'], x['name']))
    
    # Build dashboard URLs for hyperlinks
    dashboard_urls = {
        'active_mits': f"{DASHBOARD_BASE_URL}/#/in-training",
        'placement_ready': f"{DASHBOARD_BASE_URL}/#/candidates/weeks-8-plus",
        'at_risk': f"{DASHBOARD_BASE_URL}/#/candidates/week-7-priority",
        'open_roles': f"{DASHBOARD_BASE_URL}/#/open-positions"
    }
    
    # Calculate band counts for KPI boxes
    band_counts = {
        'new_starts': len(mits_by_week_band['weeks_0_3']),
        'mid_training': len(mits_by_week_band['weeks_4_5']),
        'critical_window': len(mits_by_week_band['weeks_6_7']),
        'placement_ready': len(mits_by_week_band['weeks_8_plus']),
        'pending_mits': len(mits_by_week_band['pending_mits'])
    }
    
    # Calculate active_mits as sum of all week bands (ensures consistency)
    active_mits = sum(band_counts.values())
    
    # Fetch client MIT requests
    from utils import fetch_client_mit_requests
    client_requests = fetch_client_mit_requests()
    
    # Calculate client request status breakdown
    client_request_stats = {
        'total': len(client_requests),
        'posted': 0,
        'offer_extended': 0,
        'pending_approval': 0,
        'by_client': {},
        'by_manager': {}
    }
    
    for req in client_requests:
        status_lower = req.get('status', '').lower()
        if 'posted' in status_lower:
            client_request_stats['posted'] += 1
        elif 'offer' in status_lower or 'extended' in status_lower:
            client_request_stats['offer_extended'] += 1
        elif 'pending' in status_lower:
            client_request_stats['pending_approval'] += 1
        
        # Count by client
        client = req.get('client', 'Unknown')
        client_request_stats['by_client'][client] = client_request_stats['by_client'].get(client, 0) + 1
        
        # Count by hiring manager
        manager = req.get('hiring_manager', 'Unknown')
        client_request_stats['by_manager'][manager] = client_request_stats['by_manager'].get(manager, 0) + 1
    
    # Get top clients and managers
    top_clients = sorted(client_request_stats['by_client'].items(), key=lambda x: x[1], reverse=True)[:3]
    top_managers = sorted(client_request_stats['by_manager'].items(), key=lambda x: x[1], reverse=True)[:3]
    
    return {
        'timestamp': datetime.now().strftime('%B %d, %Y at %I:%M %p'),
        'dashboard_url': DASHBOARD_BASE_URL,
        'dashboard_urls': dashboard_urls,
        'total_mits': total_mits,
        'active_mits': active_mits,
        'ready_mits': ready_mits,
        'open_positions_count': len(open_positions),
        'positions_by_region': positions_by_region,
        'job_openings_stats': {
            'total_positions': total_positions,
            'avg_salary': avg_salary,
            'top_verticals': top_verticals,
            'total_regions': len(positions_by_region)
        },
        'ready_in_30_days': ready_in_30_days,
        'mits_by_week_band': mits_by_week_band,
        'band_counts': band_counts,
        'critical_window_mentors': critical_window_mentors,
        'upcoming_mock_qbrs': upcoming_mock_qbrs[:5],  # Next 5 upcoming
        'training_indicators': training_indicators,
        'performance_insights': performance_insights,
        'meeting_insights': meeting_insights,
        'assessment_summary': assessment_summary,
        'top_2_mits': top_2_mits,
        'bottom_2_mits': bottom_2_mits,
        'all_mits': all_mits,
        'assessment_data': assessment_data,
        'client_requests': client_requests,
        'client_request_stats': client_request_stats,
        'top_clients': top_clients,
        'top_managers': top_managers
    }

