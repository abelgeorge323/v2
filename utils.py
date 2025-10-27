"""
Utility functions for MIT Dashboard API Server
==============================================

This module contains utility functions used throughout the dashboard system.
These functions handle data processing, validation, and common operations.

Author: AI Assistant
Date: October 2025
"""

import pandas as pd
import requests
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from config import *

logger = logging.getLogger(__name__)

# =============================================================================
# DATA PROCESSING UTILITIES
# =============================================================================

def format_company_start_date(date_value: Any) -> str:
    """
    Format Company Start Date for display
    Converts pandas Timestamp or datetime to readable string format
    """
    if pd.isna(date_value) or date_value == '':
        return '—'
    
    try:
        # If it's a pandas Timestamp or datetime, format it
        if isinstance(date_value, pd.Timestamp):
            return date_value.strftime('%m/%d/%Y')
        elif isinstance(date_value, datetime):
            return date_value.strftime('%m/%d/%Y')
        else:
            return str(date_value)
    except (ValueError, TypeError):
        return '—'

def parse_salary(salary_value: Any) -> float:
    """
    Parse salary value and convert to float
    
    This function handles various salary formats including:
    - Plain numbers: 50000
    - Currency formatted: $50,000
    - String values: "50000"
    - Empty/null values
    
    Args:
        salary_value: Raw salary value from spreadsheet
        
    Returns:
        float: Parsed salary value, 0.0 if invalid
        
    Example:
        >>> parse_salary("$50,000")
        50000.0
        >>> parse_salary("")
        0.0
    """
    if pd.isna(salary_value) or salary_value == '':
        return 0.0
    
    try:
        # Remove common currency symbols and commas
        salary_str = str(salary_value).replace('$', '').replace(',', '').strip()
        return float(salary_str)
    except (ValueError, TypeError):
        logger.warning(f"Could not parse salary value: {salary_value}")
        return 0.0

def calculate_week_from_start_date(start_date: Any) -> int:
    """
    Calculate current week based on training start date
    
    This function calculates how many weeks have passed since the training
    start date. It's used to determine the current week of training.
    
    Args:
        start_date: Training start date (can be string, datetime, or pandas timestamp)
        
    Returns:
        int: Current week number, 0 if invalid or before start date
        
    Example:
        >>> calculate_week_from_start_date("2025-09-29")
        3  # If today is 3 weeks after start date
    """
    if pd.isna(start_date) or start_date == '':
        return 0
    
    try:
        start_dt = pd.to_datetime(start_date)
        current_date = datetime.now()
        week_diff = (current_date - start_dt).days // 7
        return max(0, week_diff)
    except (ValueError, TypeError):
        logger.warning(f"Could not parse start date: {start_date}")
        return 0

def convert_numpy_types(value: Any) -> Any:
    """
    Convert numpy types to native Python types for JSON serialization
    
    Flask's JSON encoder cannot handle numpy types directly, so this function
    converts them to native Python types that can be serialized.
    
    Args:
        value: Value that may contain numpy types
        
    Returns:
        Converted value with native Python types
        
    Example:
        >>> convert_numpy_types(pd.Timestamp('2025-10-20'))
        '2025-10-20 00:00:00'
    """
    if pd.isna(value):
        return None
    elif isinstance(value, (pd.Timestamp, datetime)):
        return str(value)
    elif hasattr(value, 'item'):  # numpy scalar
        return value.item()
    else:
        return value

# =============================================================================
# PROGRESS CALCULATION UTILITIES
# =============================================================================

def calculate_business_lessons_progress(row: pd.Series) -> Dict[str, Any]:
    """
    Calculate business lessons completion based on dates
    
    This function checks each business lesson column for completion status.
    A lesson is considered completed if:
    1. The date in the column has passed (date <= today)
    2. The value is a completion keyword (yes, completed, x, etc.)
    
    Args:
        row: Pandas Series containing candidate data
        
    Returns:
        Dict containing completion statistics:
        {
            'completed': int,  # Number of completed lessons
            'total': int,      # Total number of lessons
            'percentage': int  # Completion percentage
        }
    """
    completed = 0
    total = len(BUSINESS_LESSON_COLUMNS)
    current_date = datetime.now()
    
    if DEBUG_MODE:
        candidate_name = row.get('MIT Name', 'Unknown')
        logger.info(f"=== BUSINESS LESSONS DEBUG FOR {candidate_name} ===")
        logger.info(f"Today's Date: {current_date.strftime('%m/%d/%Y')}")
    
    for lesson in BUSINESS_LESSON_COLUMNS:
        value = row.get(lesson, '')
        is_completed = False
        
        if pd.notna(value) and str(value).strip():
            try:
                # Try to parse as date
                lesson_date = pd.to_datetime(value)
                
                if DEBUG_MODE:
                    logger.info(f"{lesson}: {repr(value)}")
                    logger.info(f"  Parsed Date: {lesson_date.strftime('%m/%d/%Y')}")
                
                # If date has passed, lesson is completed
                if lesson_date <= current_date:
                    is_completed = True
                    if DEBUG_MODE:
                        logger.info(f"  -> ✅ COMPLETED (date has passed)")
                else:
                    if DEBUG_MODE:
                        logger.info(f"  -> ⏳ SCHEDULED (future date)")
                        
            except Exception:
                # If not a date, check for completion keywords
                if str(value).strip().lower() in COMPLETION_KEYWORDS:
                    is_completed = True
                    if DEBUG_MODE:
                        logger.info(f"  -> ✅ COMPLETED (keyword)")
                else:
                    if DEBUG_MODE:
                        logger.info(f"  -> ❌ NOT COMPLETED (not a date or keyword)")
        else:
            if DEBUG_MODE:
                logger.info(f"  -> ❌ NOT COMPLETED (empty/null)")
        
        if is_completed:
            completed += 1
    
    if DEBUG_MODE:
        logger.info(f"Total completed: {completed}/{total}")
        logger.info("=" * 50)
    
    percentage = int((completed / total) * 100) if total > 0 else 0
    return {
        'completed': completed,
        'total': total,
        'percentage': percentage
    }

def calculate_onboarding_progress(row: pd.Series) -> Dict[str, Any]:
    """
    Calculate onboarding progress based on completion status
    
    This function checks each onboarding task for completion status.
    A task is considered completed if the value is a completion keyword.
    
    Args:
        row: Pandas Series containing candidate data
        
    Returns:
        Dict containing onboarding progress statistics
    """
    completed = 0
    total = len(ONBOARDING_TASKS)
    
    for task in ONBOARDING_TASKS:
        value = row.get(task, '')
        if pd.notna(value) and str(value).strip().lower() in COMPLETION_KEYWORDS:
            completed += 1
    
    percentage = int((completed / total) * 100) if total > 0 else 0
    return {
        'completed': completed,
        'total': total,
        'percentage': percentage
    }

# =============================================================================
# SCORE PROCESSING UTILITIES
# =============================================================================

def extract_real_scores(row: pd.Series) -> Dict[str, Any]:
    """
    Extract and process real scores from candidate data
    
    This function processes various score columns and converts them to
    appropriate numeric values for the dashboard.
    
    Args:
        row: Pandas Series containing candidate data
        
    Returns:
        Dict containing processed scores
    """
    scores = {}
    
    # Debug logging for score extraction
    candidate_name = row.get('MIT Name', 'Unknown')
    logger.info(f"=== SCORE EXTRACTION DEBUG FOR {candidate_name} ===")
    
    for score_key, column_name in SCORE_COLUMNS.items():
        score_value = row.get(column_name, 0)
        logger.info(f"  {score_key} -> Column '{column_name}': '{score_value}' (type: {type(score_value)})")
        
        if pd.notna(score_value) and str(score_value).strip():
            try:
                # Clean up invalid spreadsheet entries
                val_str = str(score_value).strip().replace('#REF!', '').replace('#N/A', '')
                val_str = val_str.replace('=', '').replace('"', '')
                
                # Special handling for skill_ranking (text field)
                if score_key == 'skill_ranking':
                    scores[score_key] = val_str if val_str else '—'
                    logger.info(f"    -> SUCCESS: {scores[score_key]} (text field from '{score_value}')")
                else:
                    # Numeric fields
                    scores[score_key] = float(val_str) if val_str else 0.0
                    logger.info(f"    -> SUCCESS: {scores[score_key]} (cleaned from '{score_value}')")
            except (ValueError, TypeError) as e:
                if score_key == 'skill_ranking':
                    scores[score_key] = '—'
                    logger.info(f"    -> ERROR: {e}, defaulting to '—'")
                else:
                    scores[score_key] = 0.0
                    logger.info(f"    -> ERROR: {e}, defaulting to 0.0")
        else:
            if score_key == 'skill_ranking':
                scores[score_key] = '—'
                logger.info(f"    -> EMPTY/NULL, defaulting to '—'")
            else:
                scores[score_key] = 0.0
                logger.info(f"    -> EMPTY/NULL, defaulting to 0.0")
    
    logger.info(f"Final scores: {scores}")
    logger.info("=" * 50)
    
    return scores

# =============================================================================
# DATA FETCHING UTILITIES
# =============================================================================

def derive_status(row: pd.Series) -> str:
    """
    Derive candidate status based on available data
    
    Args:
        row: Pandas Series containing candidate data
        
    Returns:
        str: Candidate status ('training', 'ready', 'offer_pending', etc.)
    """
    week = row.get('Week', 0)
    completion_status = row.get('Completion Status', '')
    
    # Handle "#ref!" or invalid status values
    if pd.isna(completion_status) or completion_status == '' or completion_status == '#ref!':
        # Derive from week calculation
        if isinstance(week, (int, float)) and week > 6:
            return "ready"
        else:
            return "training"
    else:
        # Use completion status if available and valid
        status_str = str(completion_status).lower().strip()
        if 'offer pending' in status_str:
            return "offer_pending"
        elif 'offer accepted' in status_str or 'completed' in status_str:
            return "ready"
        else:
            return "training"

def fetch_open_positions_data() -> List[Dict[str, Any]]:
    """
    Fetch open positions data from Google Sheets
    
    Returns:
        List of dictionaries containing open position data
    """
    try:
        logger.info("Fetching open positions data from Google Sheets")
        
        # Fetch data from Google Sheets with skiprows=5 to skip the first 5 rows
        df = pd.read_csv(OPEN_POSITIONS_URL, skiprows=5)
        
        # Clean up the data - remove any completely empty rows
        df = df.dropna(how='all')
        
        # Filter out rows where Job Title is empty or NaN
        df = df[df['Job Title'].notna() & (df['Job Title'] != '')]
        
        positions = []
        for _, row in df.iterrows():
            position = {
                'id': int(row.get('JV ID', 0)) if pd.notna(row.get('JV ID')) else 0,
                'title': str(row.get('Job Title', '')),
                'jv_id': str(row.get('JV ID', '')),
                'jv_link': str(row.get('JV Link', '')),
                'vertical': str(row.get('VERT', '')),
                'account': str(row.get('Account', '')),
                'city': str(row.get('City', '')),
                'state': str(row.get('State', '')),
                'salary': str(row.get('Salary', ''))
            }
            positions.append(position)
        
        logger.info(f"Successfully fetched {len(positions)} open positions")
        return positions
        
    except Exception as e:
        logger.error(f"Error fetching open positions data: {str(e)}")
        # Return empty list on error
        return []

def fetch_google_sheets_data() -> pd.DataFrame:
    """
    Fetch data from Google Sheets with error handling
    
    This function fetches data from the configured Google Sheets URL,
    processes it, and returns a clean DataFrame.
    
    Returns:
        pd.DataFrame: Processed data from Google Sheets
        
    Raises:
        Exception: If data cannot be fetched or processed
    """
    try:
        logger.info("Fetching data from Google Sheets")
        
        # Fetch data from Google Sheets
        response = requests.get(GOOGLE_SHEETS_URL, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        
        # Read CSV data
        df = pd.read_csv(GOOGLE_SHEETS_URL, dtype=str)
        
        # Filter for target programs
        df = df[df['Training Program'].isin(TARGET_PROGRAMS)]
        
        # Apply column mapping
        df = df.rename(columns=COLUMN_MAPPING)
        
        # Calculate Week from Company Start Date
        if "Company Start Date" in df.columns:
            # Save original string values before conversion
            df["Company Start Date Original"] = df["Company Start Date"].copy()
            df["Company Start Date"] = pd.to_datetime(df["Company Start Date"], errors="coerce")
            today = pd.Timestamp.now()
            
            # Calculate weeks with proper handling of NaN values
            days_diff = (today - df["Company Start Date"]).dt.days
            weeks = (days_diff / 7).round(0)
            
            # Handle NaN and infinite values
            weeks = weeks.fillna(0)
            weeks = weeks.replace([float('inf'), float('-inf')], 0)
            
            # Convert to int safely
            df["Week"] = weeks.astype(int)
            
            # For future dates (negative weeks), set to 0
            df["Week"] = df["Week"].clip(lower=0)
        else:
            df["Week"] = 0
        
        # Derive Status column
        df["Status"] = df.apply(derive_status, axis=1)
        
        logger.info(f"Successfully fetched and processed {len(df)} records")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching Google Sheets data: {str(e)}")
        raise

# =============================================================================
# CANDIDATE DATA PROCESSING
# =============================================================================

def process_candidate_data(row: pd.Series) -> Dict[str, Any]:
    """
    Process a single candidate's data into dashboard format
    
    This function takes a pandas Series (row) from the Google Sheets data
    and converts it into the format expected by the dashboard frontend.
    
    Args:
        row: Pandas Series containing candidate data
        
    Returns:
        Dict containing processed candidate data in dashboard format
    """
    # Calculate week from Company Start Date
    week_value = calculate_week_from_start_date(row.get('Company Start Date'))
    
    # Parse salary
    salary_value = parse_salary(row.get('Salary', 0))
    
    # Extract real scores
    real_scores = extract_real_scores(row)
    
    # Calculate progress metrics
    onboarding_progress = calculate_onboarding_progress(row)
    business_lessons_progress = calculate_business_lessons_progress(row)
    
    # Generate local headshot path from candidate name
    candidate_name = str(row.get('MIT Name', 'Unknown'))
    # Remove spaces and convert to lowercase for filename matching
    image_filename = candidate_name.replace(' ', '').lower() + '.png'
    profile_image_path = f'/headshots/{image_filename}'  # Fixed: back to /headshots/
    
    logger.info(f"Generated profile image path for {candidate_name}: {profile_image_path}")
    
    # Debug graduation week extraction
    graduation_week_raw = row.get('Expected Graduation Week', 'NOT_FOUND')
    logger.info(f"=== GRADUATION WEEK DEBUG FOR {candidate_name} ===")
    logger.info(f"Raw graduation week value: '{graduation_week_raw}' (type: {type(graduation_week_raw)})")
    
    # Build candidate data dictionary
    candidate_data = {
        'name': candidate_name,
        'training_site': str(row.get('Training Site', row.get('Ops Account- Location', '—'))),
        'location': str(row.get('Location', '—')),
        'week': week_value,
        'expected_graduation_week': str(row.get('Expected Graduation Week', '—')),
        'salary': salary_value,
        'status': str(row.get('Status', '—')),
        'training_program': str(row.get('Training Program', '—')),
        'mentor_name': str(row.get('Mentor Name', '—')),
        'mentor_title': str(row.get('Title of Mentor', '—')),
        'scores': {k: convert_numpy_types(v) for k, v in real_scores.items()},
        'onboarding_progress': {k: convert_numpy_types(v) for k, v in onboarding_progress.items()},
        'business_lessons_progress': {k: convert_numpy_types(v) for k, v in business_lessons_progress.items()},
        'operation_details': {
            'company_start_date': str(row.get('Company Start Date Original', '—')),  # Display original string
            'training_start_date': str(row.get('Training Start Date', '—')),  # Display only
            'title': str(row.get('Title', '—')),
            'operation_location': str(row.get('Ops Account- Location', '—')),
            'vertical': str(row.get('Vertical', '—'))
        },
        # Local file paths
        'resume_link': str(row.get('Resume', '')),
        'profile_image': profile_image_path
    }
    
    return candidate_data

# =============================================================================
# LOGGING UTILITIES
# =============================================================================

def log_debug(message: str) -> None:
    """
    Log debug message if debug mode is enabled
    
    Args:
        message: Debug message to log
    """
    if DEBUG_MODE:
        logger.info(f"DEBUG: {message}")

def log_error(message: str, exception: Exception = None) -> None:
    """
    Log error message with optional exception details
    
    Args:
        message: Error message to log
        exception: Optional exception object
    """
    if exception:
        logger.error(f"{message}: {str(exception)}")
    else:
        logger.error(message)

# =============================================================================
# JOB MATCHING UTILITIES
# =============================================================================

def calculate_match_score(candidate: Dict[str, Any], job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate match score between a candidate and a job position
    
    This function implements the scoring system based on 5 criteria:
    1. Vertical Alignment (30 pts + 10 bonus)
    2. Salary Trajectory (25 pts)
    3. Geographic Fit (20 pts)
    4. Confidence (15 pts)
    5. Readiness (10 pts)
    
    Args:
        candidate: Candidate data dictionary
        job: Job position data dictionary
        
    Returns:
        Dict containing total score and breakdown by category
    """
    total_score = 0
    score_breakdown = {}
    
    # 1. Vertical Alignment (30 pts + 10 bonus)
    vertical_score = 0
    candidate_vertical = str(candidate.get('operation_details', {}).get('vertical', '')).lower()
    job_vertical = str(job.get('vertical', '')).lower()
    
    if candidate_vertical and job_vertical:
        if candidate_vertical == job_vertical:
            vertical_score = 30
        else:
            vertical_score = 0
    
    # Bonus points for Amazon or Aviation
    training_location = str(candidate.get('operation_details', {}).get('operation_location', '')).lower()
    if 'amazon' in training_location or candidate_vertical == 'aviation':
        vertical_score += 10
    
    score_breakdown['vertical_alignment'] = vertical_score
    total_score += vertical_score
    
    # 2. Salary Trajectory (25 pts)
    salary_score = 0
    candidate_salary = candidate.get('salary', 0)
    job_salary = parse_salary(job.get('salary', 0))
    
    if candidate_salary > 0 and job_salary > 0:
        salary_ratio = job_salary / candidate_salary
        if salary_ratio >= 1.05:  # Job is 5%+ above candidate salary
            salary_score = 25
        elif salary_ratio >= 0.95:  # Within 5% of candidate salary
            salary_score = 15
        elif salary_ratio < 0.95:  # Job is more than 5% below
            salary_score = -10
    
    score_breakdown['salary_trajectory'] = salary_score
    total_score += salary_score
    
    # 3. Geographic Fit (20 pts)
    geo_score = 0
    candidate_location = str(candidate.get('location', '')).lower()
    job_city = str(job.get('city', '')).lower()
    job_state = str(job.get('state', '')).lower()
    
    if candidate_location and job_city:
        if job_city in candidate_location:
            geo_score = 20
        elif job_state in candidate_location:
            geo_score = 10
        else:
            geo_score = 5
    
    score_breakdown['geographic_fit'] = geo_score
    total_score += geo_score
    
    # 4. Confidence (15 pts)
    confidence_score = 10  # Default
    confidence_value = str(candidate.get('confidence', '')).lower()
    
    if 'high' in confidence_value:
        confidence_score = 15
    elif 'moderate' in confidence_value:
        confidence_score = 10
    elif 'low' in confidence_value:
        confidence_score = 5
    
    score_breakdown['confidence'] = confidence_score
    total_score += confidence_score
    
    # 5. Readiness (10 pts)
    readiness_score = 5  # Default
    week = candidate.get('week', 0)
    
    if week >= 6:
        readiness_score = 10
    elif week > 0:
        readiness_score = min(10, week * 1.5)
    
    score_breakdown['readiness'] = readiness_score
    total_score += readiness_score
    
    # Determine match quality
    if total_score >= 80:
        quality = "Excellent"
    elif total_score >= 55:
        quality = "Good"
    elif total_score >= 30:
        quality = "Fair"
    else:
        quality = "Poor"
    
    return {
        'total_score': round(total_score, 1),
        'quality': quality,
        'breakdown': score_breakdown
    }

def get_top_matches(job_id: int, limit: int = 3) -> List[Dict[str, Any]]:
    """
    Get top matching candidates for a specific job position
    
    This function fetches all candidates, calculates match scores against
    the specified job, and returns the top N matches sorted by score.
    
    Args:
        job_id: Job position ID to match against
        limit: Maximum number of matches to return (default: 3)
        
    Returns:
        List of dictionaries containing top matching candidates with scores
    """
    try:
        # Get job data
        open_positions = fetch_open_positions_data()
        target_job = None
        
        for position in open_positions:
            if position.get('id') == job_id:
                target_job = position
                break
        
        if not target_job:
            logger.error(f"Job with ID {job_id} not found")
            return []
        
        # Get all candidates
        df = fetch_google_sheets_data()
        candidates = []
        
        for _, row in df.iterrows():
            candidate_data = process_candidate_data(row)
            candidates.append(candidate_data)
        
        # Calculate match scores
        matches = []
        for candidate in candidates:
            match_result = calculate_match_score(candidate, target_job)
            
            match_data = {
                'candidate': candidate,
                'job': target_job,
                'match_score': match_result['total_score'],
                'match_quality': match_result['quality'],
                'score_breakdown': match_result['breakdown']
            }
            matches.append(match_data)
        
        # Sort by score descending and return top N
        matches.sort(key=lambda x: x['match_score'], reverse=True)
        return matches[:limit]
        
    except Exception as e:
        logger.error(f"Error getting top matches for job {job_id}: {str(e)}")
        return []
