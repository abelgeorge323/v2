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
from difflib import SequenceMatcher
import logging
import os
import json
from config import *

logger = logging.getLogger(__name__)

# =============================================================================
# SAFE COLUMN ACCESS UTILITIES
# =============================================================================

def safe_get(row_or_dict: Any, column_name: str, default: Any = None) -> Any:
    """
    Safely access a column from a pandas Series, DataFrame row, or dictionary.
    
    This function prevents KeyError exceptions when columns are missing,
    renamed, or deleted from Google Sheets.
    
    Args:
        row_or_dict: pandas Series, DataFrame row, or dictionary
        column_name: Name of the column to access
        default: Default value to return if column doesn't exist
        
    Returns:
        Column value if exists, otherwise default value
        
    Example:
        >>> row = pd.Series({'Name': 'John', 'Age': 25})
        >>> safe_get(row, 'Name', 'Unknown')
        'John'
        >>> safe_get(row, 'MissingColumn', 'N/A')
        'N/A'
    """
    try:
        if isinstance(row_or_dict, pd.Series):
            # For pandas Series, use .get() which returns default if key missing
            return row_or_dict.get(column_name, default)
        elif isinstance(row_or_dict, dict):
            # For dictionaries, use .get()
            return row_or_dict.get(column_name, default)
        elif hasattr(row_or_dict, '__getitem__'):
            # For DataFrame rows or other indexable objects
            try:
                return row_or_dict[column_name]
            except (KeyError, IndexError):
                return default
        else:
            return default
    except Exception as e:
        logger.warning(f"Error accessing column '{column_name}': {e}")
        return default

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names in a DataFrame using COLUMN_MAPPING.
    
    This function normalizes column headers to handle:
    - Renamed columns
    - Misspelled columns
    - Extra spaces or non-breaking spaces
    - Case variations
    
    Args:
        df: DataFrame with potentially non-standard column names
        
    Returns:
        DataFrame with standardized column names
        
    Example:
        >>> df = pd.DataFrame({'Trainee Name': ['John'], 'Confidence': [85]})
        >>> df_std = standardize_columns(df)
        >>> 'MIT Name' in df_std.columns
        True
        >>> 'Confidence Score' in df_std.columns
        True
    """
    if df is None or df.empty:
        return df
    
    # Create a copy to avoid modifying original
    df_std = df.copy()
    
    # Normalize existing column names (handle spaces, non-breaking spaces, case)
    df_std.columns = (
        df_std.columns.astype(str)
          .str.replace('\u00A0', ' ', regex=False)  # Replace non-breaking spaces
          .str.replace(r'\s+', ' ', regex=True)      # Collapse multiple spaces
          .str.strip()
    )
    
    # Apply case-insensitive mapping using COLUMN_MAPPING
    norm = lambda s: re.sub(r'\s+', ' ', s.replace('\u00A0', ' ').strip().lower())
    mapping_norm = {norm(k): v for k, v in COLUMN_MAPPING.items()}
    
    # Rename columns based on normalized mapping
    rename_dict = {}
    for col in df_std.columns:
        col_norm = norm(col)
        if col_norm in mapping_norm:
            rename_dict[col] = mapping_norm[col_norm]
    
    if rename_dict:
        df_std.rename(columns=rename_dict, inplace=True)
        logger.debug(f"Standardized columns: {rename_dict}")
    
    return df_std

def validate_schema(df: pd.DataFrame, sheet_name: str = "Sheet") -> None:
    """
    Validate that required columns exist in the DataFrame.
    
    Logs warnings for missing columns but does not stop execution.
    This allows the dashboard to continue functioning even if some columns
    are missing, renamed, or deleted.
    
    Args:
        df: DataFrame to validate
        sheet_name: Name of the sheet being validated (for logging)
        
    Example:
        >>> df = pd.DataFrame({'MIT Name': ['John'], 'Age': [25]})
        >>> validate_schema(df, "Main Sheet")
        # Logs warning if 'Training Program' is missing
    """
    if df is None or df.empty:
        logger.warning(f"{sheet_name}: DataFrame is empty or None")
        return
    
    missing_required = []
    missing_important = []
    
    # Check required columns
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            missing_required.append(col)
    
    # Check important columns
    for col in IMPORTANT_COLUMNS:
        if col not in df.columns:
            missing_important.append(col)
    
    # Log warnings
    if missing_required:
        logger.warning(
            f"{sheet_name}: Missing REQUIRED columns: {missing_required}. "
            f"Dashboard functionality may be limited."
        )
    
    if missing_important:
        logger.warning(
            f"{sheet_name}: Missing IMPORTANT columns: {missing_important}. "
            f"Some features may not work correctly."
        )
    
    if not missing_required and not missing_important:
        logger.debug(f"{sheet_name}: All required and important columns present")

# =============================================================================
# BIOS SUPPORT (loaded from data/bios.json)
# =============================================================================

_BIOS_CACHE: Optional[Dict[str, str]] = None

def _load_bios_file() -> Dict[str, str]:
    """Load bios from data/bios.json once and cache."""
    global _BIOS_CACHE
    if _BIOS_CACHE is not None:
        return _BIOS_CACHE
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_dir, 'data', 'bios.json')
        # If utils.py is at repo root, adjust path accordingly
        if not os.path.exists(data_path):
            data_path = os.path.join(os.path.dirname(base_dir), 'data', 'bios.json')
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Normalize keys to lowercase for lookup
        _BIOS_CACHE = {str(k).lower(): str(v) for k, v in data.items()}
    except Exception:
        _BIOS_CACHE = {}
    return _BIOS_CACHE

def get_bio_for_name(name: Any) -> str:
    """Return a short bio for a candidate using tolerant matching."""
    bios = _load_bios_file()
    key = normalize_name(name)
    # 1) Exact normalized key
    if key in bios:
        return bios[key]
    # 2) Token containment / fuzzy match against bios keys
    for bio_key, bio_text in bios.items():
        if fuzzy_match_name(bio_key, key, threshold=0.82):
            return bio_text
    # 3) No match
    return ''

# =============================================================================
# HEADSHOT RESOLUTION
# =============================================================================

def resolve_headshot_path(name: Any) -> str:
    """Return a headshot path under /headshots supporting multiple extensions.
    File naming convention: <first><last>.<ext> lowercased, spaces removed.
    """
    # Build slug
    text = str(name or '').strip()
    slug = re.sub(r'\s+', '', text).lower()
    candidates = [
        f"headshots/{slug}.png",
        f"headshots/{slug}.jpg",
        f"headshots/{slug}.jpeg",
        f"headshots/{slug}.webp",
    ]
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Try current dir then parent (in case utils.py is inside a subdir)
    search_roots = [base_dir, os.path.dirname(base_dir)]
    for rel in candidates:
        for root in search_roots:
            abs_path = os.path.join(root, rel)
            if os.path.exists(abs_path):
                # Return public URL path
                return '/' + rel.replace('\\', '/')
    # Fallback to default .png (keeps previous behavior)
    return f"/headshots/{slug}.png"

# =============================================================================
# DATA PROCESSING UTILITIES
# =============================================================================

def format_company_start_date(date_value: Any) -> str:
    """
    Format Company Start Date for display
    Converts pandas Timestamp or datetime to readable string format
    """
    if pd.isna(date_value) or date_value == '':
        return 'TBD'
    
    try:
        # If it's a pandas Timestamp or datetime, format it
        if isinstance(date_value, pd.Timestamp):
            return date_value.strftime('%m/%d/%Y')
        elif isinstance(date_value, datetime):
            return date_value.strftime('%m/%d/%Y')
        else:
            return str(date_value)
    except (ValueError, TypeError):
        return 'TBD'

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
        raw = str(salary_value).strip()
        if raw.lower() in {'tbd', 'na', 'n/a', 'none', '-', 'nan'}:
            return 0.0
        # Remove currency symbols and commas
        cleaned = raw.replace('$', '').replace(',', '').replace('USD', '').strip()
        # Extract first numeric token
        match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
        if match:
            return float(match.group(0))
        return 0.0
    except Exception:
        return 0.0

def normalize_name(name: Any) -> str:
    """
    Normalize a person's name for comparison: lowercase, strip punctuation/extra spaces.
    """
    if pd.isna(name):
        return ''
    text = str(name).lower().strip()
    # Replace non-breaking spaces and collapse spaces
    text = text.replace('\u00A0', ' ')
    text = re.sub(r'\s+', ' ', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    return text

def fuzzy_match_name(name1: Any, name2: Any, threshold: float = 0.85) -> bool:
    """
    Enhanced fuzzy name matching with token containment and prefix checks.
    Handles cases like "Stephany Lopez" vs "Stephany Lopez Cardona".
    """
    n1 = normalize_name(name1)
    n2 = normalize_name(name2)
    if not n1 or not n2:
        return False

    # Token containment (handles extra surname): "stephany lopez" vs "stephany lopez cardona"
    t1, t2 = n1.split(), n2.split()
    shorter, longer = (t1, t2) if len(t1) <= len(t2) else (t2, t1)
    if len(shorter) >= 2 and all(tok in longer for tok in shorter):
        return True

    # Prefix containment (nicknames or partials)
    if n1.startswith(n2) or n2.startswith(n1):
        return True

    # Dynamic threshold: lower for multi-token names
    dyn_threshold = 0.80 if len(shorter) >= 2 else threshold
    return SequenceMatcher(None, n1, n2).ratio() >= dyn_threshold

def mentor_match(m1: Any, m2: Any) -> bool:
    """
    Check if two mentor names match after normalization.
    """
    return normalize_name(m1) != '' and normalize_name(m1) == normalize_name(m2)

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
        candidate_name = safe_get(row, 'MIT Name', 'Unknown')
        logger.info(f"=== BUSINESS LESSONS DEBUG FOR {candidate_name} ===")
        logger.info(f"Today's Date: {current_date.strftime('%m/%d/%Y')}")
    
    for lesson in BUSINESS_LESSON_COLUMNS:
        value = safe_get(row, lesson, '')
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
        value = safe_get(row, task, '')
        if pd.notna(value) and str(value).strip().lower() in COMPLETION_KEYWORDS:
            completed += 1
    
    percentage = int((completed / total) * 100) if total > 0 else 0
    return {
        'completed': completed,
        'total': total,
        'percentage': percentage
    }

def calculate_oig_completion(row: pd.Series) -> Dict[str, Any]:
    """
    Calculate OIG (On-Site Integration) completion status
    
    OIG started on 11/17/2024. Candidates who entered the program before this date
    are automatically marked as completed (grandfathered in).
    
    For candidates after 11/17/2024, completion is determined by the 
    "Mentor Certification of Trainee's OIG Completion" column (Column BB).
    
    Args:
        row: Pandas Series containing candidate data
        
    Returns:
        Dict containing OIG completion status:
        {
            'completed': bool,  # True if OIG is completed
            'exempt': bool,     # True if candidate started before OIG program
            'status': str       # 'Completed', 'Exempt', or 'Not Completed'
        }
    """
    from datetime import datetime
    from config import OIG_START_DATE
    
    candidate_name = safe_get(row, 'MIT Name', 'Unknown')
    
    # Get company start date (already converted to datetime in fetch_google_sheets_data)
    company_start_date = safe_get(row, 'Company Start Date')
    oig_completion_value = safe_get(row, 'OIG Completion', '')
    
    # Parse OIG start date
    try:
        oig_program_start = pd.to_datetime(OIG_START_DATE)
    except Exception as e:
        logger.warning(f"Could not parse OIG_START_DATE '{OIG_START_DATE}': {e}")
        oig_program_start = pd.to_datetime("2024-11-17")
    
    # Check if candidate started before OIG program
    # Company Start Date is already a Timestamp from fetch_google_sheets_data()
    if pd.notna(company_start_date):
        try:
            # Convert to Timestamp if it's not already
            if not isinstance(company_start_date, pd.Timestamp):
                candidate_start = pd.to_datetime(company_start_date)
            else:
                candidate_start = company_start_date
            
            logger.debug(f"OIG Check for {candidate_name}: Start date = {candidate_start}, OIG start = {oig_program_start}")
            
            if candidate_start < oig_program_start:
                # Grandfathered in - exempt from OIG
                logger.info(f"{candidate_name} exempt from OIG (started {candidate_start} before {oig_program_start})")
                return {
                    'completed': True,
                    'exempt': True,
                    'status': 'Exempt (Started before OIG program)'
                }
        except Exception as e:
            logger.error(f"Error parsing date for {candidate_name}: {e}, date value: {company_start_date}")
            # Continue to check OIG Completion column
    else:
        logger.warning(f"{candidate_name}: No Company Start Date found for OIG exemption check")
    
    # Check OIG Completion column (Column BB)
    if pd.notna(oig_completion_value):
        oig_value_str = str(oig_completion_value).strip().lower()
        
        # Check for "Yes" or other completion keywords
        if oig_value_str in ['yes', 'y', 'true', '1', 'completed', 'complete', 'x']:
            logger.info(f"{candidate_name}: OIG marked as completed in Column BB")
            return {
                'completed': True,
                'exempt': False,
                'status': 'Completed'
            }
    
    # Not completed
    logger.debug(f"{candidate_name}: OIG not completed")
    return {
        'completed': False,
        'exempt': False,
        'status': 'Not Completed'
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
    
    # Only log verbose details in debug mode (uses logger.debug which is faster)
    candidate_name = safe_get(row, 'MIT Name', 'Unknown')
    logger.debug(f"=== SCORE EXTRACTION DEBUG FOR {candidate_name} ===")
    
    for score_key, column_name in SCORE_COLUMNS.items():
        score_value = safe_get(row, column_name, 0)
        logger.debug(f"  {score_key} -> Column '{column_name}': '{score_value}' (type: {type(score_value)})")
        
        if pd.notna(score_value) and str(score_value).strip():
            try:
                # Clean up invalid spreadsheet entries
                val_str = str(score_value).strip().replace('#REF!', '').replace('#N/A', '')
                val_str = val_str.replace('=', '').replace('"', '')
                
                # Special handling for skill_ranking (text field)
                if score_key == 'skill_ranking':
                    scores[score_key] = val_str if val_str else 'TBD'
                    logger.debug(f"    -> SUCCESS: {scores[score_key]} (text field from '{score_value}')")
                elif score_key == 'mock_qbr_score':
                    # Mock QBR Score is out of 4, not 100
                    scores[score_key] = float(val_str) if val_str else 0.0
                    logger.debug(f"    -> SUCCESS: {scores[score_key]} (Mock QBR out of 4 from '{score_value}')")
                else:
                    # Other numeric fields (out of 100)
                    scores[score_key] = float(val_str) if val_str else 0.0
                    logger.debug(f"    -> SUCCESS: {scores[score_key]} (cleaned from '{score_value}')")
            except (ValueError, TypeError) as e:
                if score_key == 'skill_ranking':
                    scores[score_key] = 'TBD'
                    logger.debug(f"    -> ERROR: {e}, defaulting to 'TBD'")
                else:
                    scores[score_key] = 0.0
                    logger.debug(f"    -> ERROR: {e}, defaulting to 0.0")
        else:
            if score_key == 'skill_ranking':
                scores[score_key] = 'TBD'
                logger.debug(f"    -> EMPTY/NULL, defaulting to 'TBD'")
            else:
                scores[score_key] = 0.0
                logger.debug(f"    -> EMPTY/NULL, defaulting to 0.0")
    
    logger.debug(f"Final scores: {scores}")
    logger.debug("=" * 50)
    
    return scores

# =============================================================================
# AVERAGE CALCULATION UTILITIES (for Executive Report)
# =============================================================================

def calculate_avg_onboarding_completion(candidates: List[Dict[str, Any]]) -> float:
    """
    Calculate average onboarding completion percentage across candidates
    
    Args:
        candidates: List of candidate dictionaries with onboarding_progress
        
    Returns:
        float: Average onboarding completion percentage (0-100)
    """
    if not candidates:
        return 0.0
    
    total_percentage = 0.0
    count = 0
    
    for candidate in candidates:
        onboarding = candidate.get('onboarding_progress', {})
        if isinstance(onboarding, dict):
            percentage = onboarding.get('percentage', 0)
            if isinstance(percentage, (int, float)):
                total_percentage += float(percentage)
                count += 1
    
    return round(total_percentage / count, 1) if count > 0 else 0.0

def calculate_avg_lessons_completion(candidates: List[Dict[str, Any]]) -> float:
    """
    Calculate average business lessons completion percentage across candidates
    
    Args:
        candidates: List of candidate dictionaries with business_lessons_progress
        
    Returns:
        float: Average lessons completion percentage (0-100)
    """
    if not candidates:
        return 0.0
    
    total_percentage = 0.0
    count = 0
    
    for candidate in candidates:
        lessons = candidate.get('business_lessons_progress', {})
        if isinstance(lessons, dict):
            percentage = lessons.get('percentage', 0)
            if isinstance(percentage, (int, float)):
                total_percentage += float(percentage)
                count += 1
    
    return round(total_percentage / count, 1) if count > 0 else 0.0

def calculate_avg_qbr_score(candidates: List[Dict[str, Any]]) -> Optional[float]:
    """
    Calculate average Mock QBR score across candidates
    
    Args:
        candidates: List of candidate dictionaries with scores
        
    Returns:
        Optional[float]: Average Mock QBR score, or None if no valid scores
    """
    if not candidates:
        return None
    
    total_score = 0.0
    count = 0
    
    for candidate in candidates:
        scores = candidate.get('scores', {})
        qbr_score = scores.get('mock_qbr_score', 0)
        if isinstance(qbr_score, (int, float)) and qbr_score > 0:
            total_score += float(qbr_score)
            count += 1
    
    return round(total_score / count, 2) if count > 0 else None

def calculate_avg_perf_score(candidates: List[Dict[str, Any]]) -> Optional[float]:
    """
    Calculate average Performance Evaluation score across candidates
    
    Args:
        candidates: List of candidate dictionaries with scores
        
    Returns:
        Optional[float]: Average performance score, or None if no valid scores
    """
    if not candidates:
        return None
    
    total_score = 0.0
    count = 0
    
    for candidate in candidates:
        scores = candidate.get('scores', {})
        perf_score = scores.get('perf_evaluation_score', 0)
        if isinstance(perf_score, (int, float)) and perf_score > 0:
            total_score += float(perf_score)
            count += 1
    
    return round(total_score / count, 2) if count > 0 else None

# =============================================================================
# DATA FETCHING UTILITIES
# =============================================================================

def derive_status(row: pd.Series) -> str:
    """
    Derive candidate status based on available data
    
    Args:
        row: Pandas Series containing candidate data
        
    Returns:
        str: Candidate status ('training', 'ready', 'pending start', etc.)
    """
    week = safe_get(row, 'Week', 0)
    completion_status = safe_get(row, 'Completion Status', '')
    
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
        if 'pending' in status_str:
            return "pending start"
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
        # (Row 6 becomes the header row with Job Title, JV ID, etc.)
        df = pd.read_csv(OPEN_POSITIONS_URL, skiprows=5)
        
        # Standardize columns FIRST
        df = standardize_columns(df)
        
        # Validate schema
        validate_schema(df, "Open Positions")
        
        # Drop the first empty column (the leading comma in the CSV)
        if len(df.columns) > 0 and 'Unnamed' in str(df.columns[0]):
            logger.info(f"Dropping empty first column: {df.columns[0]}")
            df = df.drop(df.columns[0], axis=1)
        
        # Debug: Log available columns
        logger.info(f"Available columns in open positions: {list(df.columns)}")
        
        # Clean up the data - remove any completely empty rows
        df = df.dropna(how='all')
        
        # Filter out rows where Job Title is empty or NaN
        if 'Job Title' in df.columns:
            df = df[df['Job Title'].notna() & (df['Job Title'].astype(str).str.strip() != '')]
        else:
            logger.error("'Job Title' column not found in open positions data")
            return []
        
        positions = []
        for _, row in df.iterrows():
            # Skip header-like rows (like "Available Placement Options")
            job_title = str(safe_get(row, 'Job Title', '')).strip()
            if job_title.lower() in ['job title', 'available placement options', '']:
                continue
            
            # Parse location from Location column
            location = str(safe_get(row, 'Location', '')).strip()
            city, state = '', ''
            
            if location and location.lower() not in ['nan', 'none', '']:
                # Split "Detroit, MI" into city and state
                parts = location.split(',')
                if len(parts) >= 2:
                    city = parts[0].strip()
                    state = parts[1].strip()
                else:
                    city = location.strip()
            
            # Parse JV ID - handle empty values and formats like "15422-1"
            jv_id_raw = safe_get(row, 'JV ID', '')
            try:
                # Try to convert to int for the 'id' field
                if pd.notna(jv_id_raw) and str(jv_id_raw).strip():
                    jv_id_clean = str(jv_id_raw).strip().split('-')[0]  # Handle "15422-1" format
                    job_id = int(float(jv_id_clean))
                else:
                    job_id = 0
            except (ValueError, TypeError):
                job_id = 0
            
            position = {
                'id': job_id,
                'title': job_title,
                'jv_id': str(jv_id_raw).strip() if pd.notna(jv_id_raw) else '',
                'jv_link': '',  # Not in your sheet, leaving empty
                'vertical': str(safe_get(row, 'VERT', '')).strip(),
                'account': str(safe_get(row, 'Account', '')).strip(),
                'city': city,
                'state': state,
                'salary': parse_salary(safe_get(row, 'Salary', 0))
            }
            positions.append(position)
        
        logger.info(f"Successfully fetched {len(positions)} open positions")
        if positions:
            logger.info(f"Sample positions: {[p['title'] + ' at ' + p['account'] for p in positions[:3]]}")
        return positions
        
    except Exception as e:
        logger.error(f"Error fetching open positions data: {str(e)}", exc_info=True)
        # Return empty list on error
        return []

def _read_csv_normalized(url: str, dtype: Optional[Dict[str, Any]] = None, skiprows: Optional[int] = None) -> pd.DataFrame:
    """
    Helper: read CSV from URL and normalize headers (strip spaces, collapse, replace NBSP).
    """
    df = pd.read_csv(url, dtype=dtype, skiprows=skiprows)
    df.columns = (
        df.columns.astype(str)
          .str.replace('\u00A0', ' ', regex=False)
          .str.replace(r'\s+', ' ', regex=True)
          .str.strip()
    )
    return df

def fetch_mit_tracking_data() -> pd.DataFrame:
    """
    Fetch the MIT Tracking Sheet which has two sections in one CSV:
    - Top section: active MIT candidates (columns present)
    - Second section after the row containing 'MIT Entering the Program': new entrants
    Returns a unified DataFrame with consistent column names. New entrants are tagged as offer pending.
    """
    try:
        # Read the CSV directly
        df = pd.read_csv(MIT_TRACKING_SHEET_URL, dtype=str)
        
        # The actual data structure is different - let's inspect it
        logger.info(f"Raw CSV shape: {df.shape}")
        logger.info(f"Raw columns: {list(df.columns)}")
        
        # The first row contains the headers, but they're spread across columns
        # Let's find the actual header row and data rows
        header_row = None
        data_start_row = None
        
        for idx, row in df.iterrows():
            # Look for row containing 'MIT Name' in any column
            if 'MIT Name' in str(row.values):
                header_row = idx
                data_start_row = idx + 1
                break
        
        if header_row is None:
            logger.error("Could not find header row with 'MIT Name'")
            return pd.DataFrame(columns=['MIT Name'])
        
        # Extract the header row and create proper column mapping
        header_values = df.iloc[header_row].values
        logger.info(f"Header values: {list(header_values)}")
        
        # Create a new DataFrame starting from the data rows
        df_data = df.iloc[data_start_row:].copy()
        
        # Map the columns based on the header row
        column_mapping = {}
        for i, header in enumerate(header_values):
            if pd.notna(header) and str(header).strip():
                column_mapping[df.columns[i]] = str(header).strip()
        
        logger.info(f"Column mapping: {column_mapping}")
        
        # Rename columns
        df_data.rename(columns=column_mapping, inplace=True)
        
        # Normalize column headers
        df_data.columns = (
            df_data.columns.astype(str)
              .str.replace('\u00A0', ' ', regex=False)
              .str.replace(r'\s+', ' ', regex=True)
              .str.strip()
        )
        
        # Standardize columns using COLUMN_MAPPING
        df_data = standardize_columns(df_data)
        
        # Standardize column names (additional mapping for MIT Tracking specific columns)
        rename_map = {
            'Week (in MIT budget)': 'Week',
            'Start date': 'Start date',
            'VERT': 'VERT',
            'Training Site': 'Training Site',
            'Location': 'Location',
            'Salary': 'Salary',
            'Level': 'Level',
            'Status': 'Status',
            'Confidence': 'Confidence',
            'Mentor': 'Mentor',
            'MIT Name': 'MIT Name',
            'New Candidate Name': 'MIT Name'  # For the second section
        }
        df_data.rename(columns={k: v for k, v in rename_map.items() if k in df_data.columns}, inplace=True)
        
        # Validate schema
        validate_schema(df_data, "MIT Tracking Sheet")
        
        # Filter out empty rows and header rows
        if 'MIT Name' in df_data.columns:
            df_data = df_data[df_data['MIT Name'].notna()]
            df_data = df_data[df_data['MIT Name'].astype(str).str.strip().str.lower() != 'mit name']
            df_data = df_data[df_data['MIT Name'].astype(str).str.strip() != '']
            df_data = df_data[df_data['MIT Name'].astype(str).str.strip() != 'New Candidate Name']  # Remove header row from second section
        else:
            logger.warning("'MIT Name' column missing - cannot filter rows")
        
        # Handle the two sections
        # Mark candidates from second section as pending start
        if 'JV' in df_data.columns:
            df_data.loc[df_data['JV'].notna(), 'Status'] = 'Pending Start'
        
        # Also mark rows where Status is empty as Pending Start (for second section)
        if 'Status' in df_data.columns:
            df_data['Status'] = df_data['Status'].fillna('')
            if 'MIT Name' in df_data.columns:
                df_data.loc[(df_data['Status'] == '') & (df_data['MIT Name'].notna()), 'Status'] = 'Pending Start'
        
        # ========================================
        # DETECT INCOMING MITS (Future Start Dates)
        # ========================================
        
        # Primary check: Start date is in the future
        # Check for both "Start date" (training section) and "Start Date" (incoming section)
        start_date_col = None
        if 'Start date' in df_data.columns:
            start_date_col = 'Start date'
        elif 'Start Date' in df_data.columns:
            start_date_col = 'Start Date'
        
        if start_date_col:
            try:
                # Parse start dates
                df_data['Start date parsed'] = pd.to_datetime(df_data[start_date_col], errors='coerce')
                current_date = pd.Timestamp.now()
                
                # Mark candidates with future start dates as "Offer Pending"
                future_start_mask = (
                    df_data['Start date parsed'].notna() & 
                    (df_data['Start date parsed'] > current_date) &
                    df_data['MIT Name'].notna()
                )
                df_data.loc[future_start_mask, 'Status'] = 'Pending Start'
                
                logger.info(f"Marked {future_start_mask.sum()} candidates with future start dates as Pending Start (using column '{start_date_col}')")
                
            except Exception as e:
                logger.warning(f"Could not parse start dates for incoming MIT detection: {e}")
        
        # Backup check: Status contains "Training starting" keywords AND has future start date
        # This catches edge cases where date parsing worked but we want to be extra sure
        if start_date_col and 'Start date parsed' in df_data.columns:
            # Add column existence checks before accessing
            if 'Status' in df_data.columns and 'MIT Name' in df_data.columns:
                backup_mask = (
                    df_data['Status'].astype(str).str.contains('Training starting|starting TBD', case=False, na=False, regex=True) &
                    df_data['MIT Name'].notna() &
                    df_data['Start date parsed'].notna() &
                    (df_data['Start date parsed'] > current_date)  # ONLY if future start date
                )
                df_data.loc[backup_mask, 'Status'] = 'Pending Start'
                logger.info(f"Backup check marked {backup_mask.sum()} additional candidates as Pending Start")
            else:
                logger.warning("Cannot create backup mask: 'Status' or 'MIT Name' column missing")
        
        # Safe logging - check if Status column exists before accessing
        if 'Status' in df_data.columns:
            logger.info(f"Total candidates marked as Pending Start: {(df_data['Status'] == 'Pending Start').sum()}")
        else:
            logger.warning("'Status' column missing - cannot count Pending Start candidates")
        
        # Normalize Week to numeric
        if 'Week' in df_data.columns:
            df_data['Week'] = pd.to_numeric(df_data['Week'], errors='coerce').fillna(0).astype(int)
        
        # Ensure all expected columns exist
        expected_cols = ['MIT Name', 'Week', 'Start date', 'VERT', 'Training Site', 'Location', 'Salary', 'Level', 'Status', 'Confidence', 'Mentor', 'Notes']
        for col in expected_cols:
            if col not in df_data.columns:
                df_data[col] = None
        
        logger.info(f"MIT Tracking data: Found {len(df_data)} candidates")
        if 'MIT Name' in df_data.columns:
            logger.info(f"Candidate names: {list(df_data['MIT Name'].dropna())}")
        
        return df_data
    except Exception as e:
        logger.error(f"Error fetching MIT Tracking data: {e}")
        return pd.DataFrame(columns=['MIT Name'])

def fetch_fallback_candidate_data() -> pd.DataFrame:
    """
    Fetch fallback/historical candidate sheet. It contains an extra first column.
    Drop the extra column and apply normal column mapping to align with main sheet.
    """
    try:
        df = _read_csv_normalized(FALLBACK_CANDIDATE_SHEET_URL, dtype=str)
        
        # Drop the first column (the extra "6 month survey sent?" column)
        # This ensures all columns align with the main sheet structure
        if len(df.columns) > 0:
            df = df.drop(df.columns[0], axis=1)
        
        # Standardize columns using COLUMN_MAPPING
        df = standardize_columns(df)
        
        # Validate schema
        validate_schema(df, "Fallback Candidate Sheet")

        # Preserve original Company Start Date string for display parity with main sheet
        if "Company Start Date" in df.columns and "Company Start Date Original" not in df.columns:
            df["Company Start Date Original"] = df["Company Start Date"].copy()

        # Ensure key columns exist
        for key in ['MIT Name', 'Company Start Date', 'Training Program', 'Status']:
            if key not in df.columns:
                df[key] = None

        return df
    except Exception as e:
        logger.error(f"Error fetching fallback candidate data: {e}")
        return pd.DataFrame(columns=['MIT Name'])

def fetch_placed_mits_data() -> List[Dict[str, Any]]:
    """
    Fetch placed MIT graduates from the Placed MITs Google Sheet
    
    Data structure:
    - Rows 1-3: Empty rows
    - Row 4: Header row
    - Rows 5+: "Placed MITS" data (starts after header)
    - Later rows: "Dropped Out/Terminated" section (we stop before this)
    
    Returns:
        List of dictionaries containing placed MIT graduate data
    """
    try:
        logger.info("Fetching placed MITs data from Google Sheets")
        
        # Use pandas to read CSV
        import io
        import requests
        response = requests.get(PLACED_MITS_URL)
        response.raise_for_status()
        
        # Read CSV skipping first 4 rows (rows 1-4 are empty/metadata)
        # Row 5 is the header row, so it becomes column names
        df = pd.read_csv(io.StringIO(response.text), skiprows=4, dtype=str)
        
        # Standardize columns FIRST
        df = standardize_columns(df)
        
        # Validate schema
        validate_schema(df, "Placed MITs Sheet")
        
        # Drop the first column (empty Column A)
        if len(df.columns) > 0:
            logger.info(f"Columns before drop: {df.columns.tolist()}")
            logger.info(f"First few rows of data:\n{df.head(3)}")
            df = df.drop(df.columns[0], axis=1)
            logger.info(f"Columns after drop: {df.columns.tolist()}")
            logger.info(f"First few rows after drop:\n{df.head(3)}")
        
        placed_mits = []
        
        for idx, row in df.iterrows():
            mit_name = str(safe_get(row, 'MIT Name', '')).strip()
            logger.info(f"Processing row {idx}: MIT Name = '{mit_name}'")
            
            # Stop at "Dropped Out/Terminated" section
            if 'Dropped Out' in mit_name or 'Terminated' in mit_name:
                logger.info(f"Reached 'Dropped Out/Terminated' section at row {idx}, stopping")
                break
            
            # Skip empty rows and header-like rows
            if not mit_name or mit_name.lower() in ['nan', 'none', '', 'mit name'] or 'Placed MITS' in mit_name:
                logger.info(f"Skipping row {idx}: empty or header row")
                continue
            
            # Extract placement data
            placed_mit = {
                'name': mit_name,
                'weeks_in_program': str(safe_get(row, 'Weeks in Program', 'TBD')).strip(),
                'training_start_date': str(safe_get(row, 'Start date', 'TBD')).strip(),
                'training_vertical': str(safe_get(row, 'VERT', 'TBD')).strip(),
                'training_site': str(safe_get(row, 'Training Site', 'TBD')).strip(),
                'training_location': str(safe_get(row, 'Location', 'TBD')).strip(),
                'training_salary': str(safe_get(row, 'Salary', 'TBD')).strip(),
                'level': str(safe_get(row, 'Level', 'TBD')).strip(),
                'status': str(safe_get(row, 'Status', 'TBD')).strip(),
                'confidence': str(safe_get(row, 'Confidence', 'TBD')).strip(),
                'notes': str(safe_get(row, 'Notes', '')).strip(),
                # Placement information (NEW - not in other sheets)
                'placement_site': str(safe_get(row, 'Placement Site', 'TBD')).strip(),
                'placement_title': str(safe_get(row, 'Title', 'TBD')).strip(),
                'placement_start_date': str(safe_get(row, 'New Start Date', 'TBD')).strip(),
            }
            
            logger.info(f"Found placed MIT: {mit_name}")
            placed_mits.append(placed_mit)
        
        logger.info(f"Successfully fetched {len(placed_mits)} placed MIT graduates")
        return placed_mits
        
    except Exception as e:
        logger.error(f"Error fetching placed MITs data: {e}", exc_info=True)
        return []

def find_candidate_in_sheet(name: str, df: pd.DataFrame, hint_mentor: str = None) -> Optional[pd.Series]:
    """
    Find candidate by name using exact normalized match, then fuzzy fallback.
    If hint_mentor is provided, use mentor matching as a tiebreaker for fuzzy matches.
    """
    target = normalize_name(name)
    if 'MIT Name' not in df.columns:
        return None

    # Exact first
    for _, row in df.iterrows():
        if normalize_name(safe_get(row, 'MIT Name', '')) == target:
            return row

    # Fuzzy with mentor boost
    for _, row in df.iterrows():
        nm = safe_get(row, 'MIT Name', '')
        if fuzzy_match_name(name, nm, threshold=0.85):
            return row
        # If mentor matches, accept with lower name threshold
        if hint_mentor and mentor_match(hint_mentor, safe_get(row, 'Mentor Name', '')):
            if fuzzy_match_name(name, nm, threshold=0.78):
                return row

    return None

def create_basic_profile_from_mit(tracking_row: pd.Series) -> Dict[str, Any]:
    """
    Create a minimal candidate profile from MIT Tracking fields.
    """
    name_value = str(safe_get(tracking_row, 'MIT Name', 'TBD'))
    return {
        'name': name_value,
        'training_site': str(safe_get(tracking_row, 'Training Site', 'TBD')),
        'location': str(safe_get(tracking_row, 'Location', 'TBD')),
        'week': int(pd.to_numeric(safe_get(tracking_row, 'Week', 0), errors='coerce') or 0),
        'expected_graduation_week': 'TBD',
        'salary': parse_salary(safe_get(tracking_row, 'Salary', 0)),
        'status': str(safe_get(tracking_row, 'Status', 'pending start')).lower(),
        'training_program': 'MIT',
        'mentor_name': str(safe_get(tracking_row, 'Mentor', 'TBD')),
        'mentor_title': 'TBD',
        'bio': get_bio_for_name(name_value),
        'scores': {},
        'onboarding_progress': None,
        'business_lessons_progress': None,
        'operation_details': {
            'company_start_date': str(safe_get(tracking_row, 'Start date', 'TBD')),
            'training_start_date': 'TBD',
            'title': 'TBD',
            'operation_location': str(safe_get(tracking_row, 'Training Site', 'TBD')),
            'vertical': str(safe_get(tracking_row, 'VERT', 'TBD'))
        },
        'resume_link': '',
        'profile_image': resolve_headshot_path(name_value),
        'data_quality': 'limited'
    }

def merge_candidate_sources() -> List[Dict[str, Any]]:
    """
    Build a unified candidate list from MIT Tracking (master), Main sheet, and Fallback sheet.
    MIT Tracking is the SOURCE OF TRUTH for status (especially for incoming MITs with future start dates).
    """
    mit_df = fetch_mit_tracking_data()
    main_df = fetch_google_sheets_data()
    fallback_df = fetch_fallback_candidate_data()

    unified: List[Dict[str, Any]] = []

    for _, trow in mit_df.iterrows():
        candidate_name = safe_get(trow, 'MIT Name', '')
        mentor_name = safe_get(trow, 'Mentor', '')
        if not str(candidate_name).strip():
            continue

        # MIT Tracking is source of truth for status (detects future start dates)
        mit_tracking_status = str(safe_get(trow, 'Status', '')).lower()

        # Try main with mentor hint
        found = find_candidate_in_sheet(candidate_name, main_df, hint_mentor=mentor_name)
        if found is not None:
            cand = process_candidate_data(found)
            cand['data_quality'] = 'full'
            # Override with MIT Tracking status (source of truth for incoming MITs)
            cand['status'] = mit_tracking_status
            unified.append(cand)
            continue

        # Try fallback with mentor hint
        found = find_candidate_in_sheet(candidate_name, fallback_df, hint_mentor=mentor_name)
        if found is not None:
            cand = process_candidate_data(found)
            cand['data_quality'] = 'archive'
            # Override with MIT Tracking status (source of truth for incoming MITs)
            cand['status'] = mit_tracking_status
            unified.append(cand)
            continue

        # Basic profile from MIT Tracking
        unified.append(create_basic_profile_from_mit(trow))

    return unified

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
        
        # Standardize column names FIRST (before any processing)
        df = standardize_columns(df)
        
        # Validate schema (logs warnings but doesn't stop)
        validate_schema(df, "Main Google Sheets")
        
        # 🧩 DEBUG LOG
        logger.info(f"[COLUMNS AFTER STANDARDIZATION] {list(df.columns)}")
        
        # Filter for target programs (use safe column check)
        if 'Training Program' in df.columns:
            df = df[df['Training Program'].isin(TARGET_PROGRAMS)]
        else:
            logger.warning("'Training Program' column missing - cannot filter by program")
        
        # Calculate Week from Company Start Date
        if "Company Start Date" in df.columns:
            # Save original string values before conversion
            if "Company Start Date Original" not in df.columns:
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
            logger.warning("'Company Start Date' column missing - cannot calculate weeks")
            df["Week"] = 0
        
        # Derive Status column
        df["Status"] = df.apply(derive_status, axis=1)
        
        logger.info(f"Successfully fetched and processed {len(df)} records")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching Google Sheets data: {str(e)}")
        raise

# =============================================================================
# WEEK-BASED CATEGORIZATION
# =============================================================================

def categorize_candidates_by_week(candidates: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Categorize candidates by training week for pipeline stages.
    Uses Company Start Date to calculate current week.
    
    Returns:
        Dict with keys: 'weeks_0_3' (0-2), 'weeks_4_6' (3-5), 'week_7_only' (6-7), 
                       'weeks_8_plus' (8+), 'offer_pending' (pending start/future dates), 'total_candidates', 'total_training'
    """
    weeks_0_3 = []
    weeks_4_6 = []
    week_7_only = []  # Weeks 6-7 priority spotlight
    weeks_8_plus = []  # Weeks 8+ for ready for placement
    offer_pending = []
    
    for candidate in candidates:
        status = candidate.get('status', '').lower()
        week = candidate.get('week', 0)
        
        # Pending start takes priority (incoming MITs with future start dates)
        if 'pending' in status:
            offer_pending.append(candidate)
        elif 0 <= week <= 2:
            weeks_0_3.append(candidate)
        elif 3 <= week <= 5:  # Fixed: Changed from 3-7 to 3-5
            weeks_4_6.append(candidate)
        elif 6 <= week <= 7:  # Now reachable! Critical placement window
            week_7_only.append(candidate)
        elif week >= 8:
            weeks_8_plus.append(candidate)
    
    return {
        'weeks_0_3': weeks_0_3,  # Operational Overview (Weeks 0-2)
        'weeks_4_6': weeks_4_6,  # Active Training (Weeks 3-5)
        'week_7_only': week_7_only,  # Weeks 6-7 Priority
        'weeks_8_plus': weeks_8_plus,  # Ready for Placement (Weeks 8+)
        'offer_pending': offer_pending,  # Pending Start (incoming MITs)
        'total_candidates': len(candidates),
        'total_training': len([c for c in candidates if c.get('week', 0) > 0])
    }

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
    week_value = calculate_week_from_start_date(safe_get(row, 'Company Start Date'))
    
    # Parse salary
    salary_value = parse_salary(safe_get(row, 'Salary', 0))
    
    # Extract real scores
    real_scores = extract_real_scores(row)
    
    # Calculate progress metrics
    onboarding_progress = calculate_onboarding_progress(row)
    business_lessons_progress = calculate_business_lessons_progress(row)
    oig_completion = calculate_oig_completion(row)
    
    # Resolve headshot path from candidate name
    candidate_name = str(safe_get(row, 'MIT Name', 'Unknown'))
    profile_image_path = resolve_headshot_path(candidate_name)
    
    logger.info(f"Generated profile image path for {candidate_name}: {profile_image_path}")
    
    # Debug graduation week extraction
    graduation_week_raw = safe_get(row, 'Expected Graduation Week', 'NOT_FOUND')
    logger.info(f"=== GRADUATION WEEK DEBUG FOR {candidate_name} ===")
    logger.info(f"Raw graduation week value: '{graduation_week_raw}' (type: {type(graduation_week_raw)})")
    
    # Build candidate data dictionary
    candidate_data = {
        'name': candidate_name,
        'training_site': str(safe_get(row, 'Training Site', safe_get(row, 'Ops Account- Location', 'TBD'))),
        'location': str(safe_get(row, 'Location', 'TBD')),
        'week': week_value,
        'expected_graduation_week': str(safe_get(row, 'Expected Graduation Week', 'TBD')),
        'salary': salary_value,
        'status': str(safe_get(row, 'Status', 'TBD')),
        'training_program': str(safe_get(row, 'Training Program', 'TBD')),
        'mentor_name': str(safe_get(row, 'Mentor Name', 'TBD')),
        'mentor_title': str(safe_get(row, 'Title of Mentor', 'TBD')),
        'bio': get_bio_for_name(candidate_name),
        'scores': {k: convert_numpy_types(v) for k, v in real_scores.items()},
        'onboarding_progress': {k: convert_numpy_types(v) for k, v in onboarding_progress.items()},
        'business_lessons_progress': {k: convert_numpy_types(v) for k, v in business_lessons_progress.items()},
        'oig_completion': {k: convert_numpy_types(v) for k, v in oig_completion.items()},
        'operation_details': {
            'company_start_date': str(safe_get(row, 'Company Start Date Original', 'TBD')),  # Display original string
            'training_start_date': str(safe_get(row, 'Training Start Date', 'TBD')),  # Display only
            'title': str(safe_get(row, 'Title', 'TBD')),
            'operation_location': str(safe_get(row, 'Ops Account- Location', 'TBD')),
            'vertical': str(safe_get(row, 'Vertical', 'TBD'))
        },
        # Mock QBR scheduling
        'mock_qbr_date': str(safe_get(row, 'Mock QBR Date', '')),
        # Local file paths
        'resume_link': str(safe_get(row, 'Resume', '')),
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
    
    This function implements the NEW 100-point scoring system based on 5 criteria:
    1. Vertical Alignment (20 pts max)
    2. Salary Trajectory (20 pts max)
    3. Geographic Fit (20 pts max)
    4. Confidence (20 pts max)
    5. Readiness (20 pts max)
    
    Args:
        candidate: Candidate data dictionary
        job: Job position data dictionary
        
    Returns:
        Dict containing total score and breakdown by category
    """
    total_score = 0
    score_breakdown = {}
    
    # 1. Vertical Alignment (20 pts max)
    vertical_score = 0
    candidate_vertical = str(candidate.get('operation_details', {}).get('vertical', '')).lower().strip()
    job_vertical = str(job.get('vertical', '')).lower().strip()
    training_location = str(candidate.get('operation_details', {}).get('operation_location', '')).lower()
    
    # Normalize "technology" and "tech" to be treated as the same
    candidate_vertical_normalized = 'technology' if candidate_vertical in ['tech', 'technology'] else candidate_vertical
    job_vertical_normalized = 'technology' if job_vertical in ['tech', 'technology'] else job_vertical
    
    if candidate_vertical and job_vertical:
        if candidate_vertical_normalized == job_vertical_normalized:
            # Perfect vertical match
            # Check for Amazon/AVI bonus
            if 'amazon' in training_location or 'avi' in training_location.lower():
                vertical_score = 20
                explanation = f"Perfect match + Amazon/AVI - {candidate_vertical}"
            else:
                vertical_score = 15
                explanation = f"Perfect vertical match - {candidate_vertical}"
        else:
            # No match but give partial credit
            vertical_score = 5
            explanation = f"No match - {candidate_vertical} vs {job_vertical}"
    else:
        vertical_score = 0
        explanation = "No vertical data"
    
    score_breakdown['vertical_alignment'] = {
        'score': vertical_score,
        'max': 20,
        'explanation': explanation
    }
    total_score += vertical_score
    
    # 2. Salary Trajectory (20 pts max, -5 penalty for decrease)
    salary_score = 0
    candidate_salary = candidate.get('salary', 0)
    job_salary = parse_salary(job.get('salary', 0))
    
    if candidate_salary > 0 and job_salary > 0:
        salary_increase_pct = ((job_salary - candidate_salary) / candidate_salary) * 100
        
        if salary_increase_pct >= 15:  # Big increase
            salary_score = 20
            explanation = f"Big increase ({salary_increase_pct:.1f}%)"
        elif salary_increase_pct >= 5:  # Medium increase
            salary_score = 12
            explanation = f"Medium increase ({salary_increase_pct:.1f}%)"
        elif salary_increase_pct >= 1:  # Small increase
            salary_score = 5
            explanation = f"Small increase ({salary_increase_pct:.1f}%)"
        else:  # No increase or decrease
            salary_score = -5
            explanation = f"Decrease ({salary_increase_pct:.1f}%)"
    else:
        salary_score = 0
        explanation = "No salary data"
    
    score_breakdown['salary_trajectory'] = {
        'score': salary_score,
        'max': 20,
        'explanation': explanation
    }
    total_score += salary_score
    
    # 3. Geographic Fit (20 pts max)
    geo_score = 0
    candidate_location = str(candidate.get('location', '')).lower()
    job_city = str(job.get('city', '')).lower()
    job_state = str(job.get('state', '')).lower()
    
    if candidate_location and job_city:
        if job_city in candidate_location:
            geo_score = 20
            explanation = f"Same city - {job_city.title()}"
        else:
            geo_score = 5
            explanation = f"Different location - {job_city.title()}, {job_state.upper()}"
    else:
        geo_score = 0
        explanation = "No location data"
    
    score_breakdown['geographic_fit'] = {
        'score': geo_score,
        'max': 20,
        'explanation': explanation
    }
    total_score += geo_score
    
    # 4. Confidence (20 pts max) — using numeric confidence_score
    numeric_confidence = float(candidate.get('scores', {}).get('confidence_score') or 0)
    
    if numeric_confidence >= 80:
        confidence_score = 20
        confidence_label = 'excellent'
    elif numeric_confidence >= 60:
        confidence_score = 13
        confidence_label = 'good'
    elif numeric_confidence >= 40:
        confidence_score = 7
        confidence_label = 'moderate'
    else:
        confidence_score = 2
        confidence_label = 'low'
    
    score_breakdown['confidence'] = {
        'score': confidence_score,
        'max': 20,
        'explanation': f"Confidence score: {numeric_confidence:.0f}/100 ({confidence_label})"
    }
    total_score += confidence_score
    
    # 5. Readiness (20 pts max) — week-based
    week = candidate.get('week', 0)
    
    if week >= 6:
        readiness_score = 20
        readiness_label = 'ready for placement'
    elif week >= 4:
        readiness_score = 12
        readiness_label = 'almost ready'
    else:
        readiness_score = 5
        readiness_label = 'not ready yet'
    
    score_breakdown['readiness'] = {
        'score': readiness_score,
        'max': 20,
        'explanation': f"Week {week} - {readiness_label}"
    }
    total_score += readiness_score
    
    # Determine match quality (based on 100-point scale)
    if total_score >= 75:
        quality = "Excellent"
    elif total_score >= 55:
        quality = "Good"
    elif total_score >= 35:
        quality = "Fair"
    else:
        quality = "Poor"
    
    return {
        'total_score': round(total_score, 1),
        'quality': quality,
        'breakdown': score_breakdown
    }

def get_top_matches(job_id: int, limit: int = 6) -> List[Dict[str, Any]]:
    """
    Get top matching candidates for a specific job position
    
    This function fetches all candidates from three-tier integration, calculates match scores against
    the specified job, and returns the top N matches sorted by score.
    
    Args:
        job_id: Job position ID to match against
        limit: Maximum number of matches to return (default: 4)
        
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
        
        # Get all candidates from three-tier integration
        candidates = merge_candidate_sources()
        
        # Exclude pending start candidates from matching (incoming MITs who haven't started)
        candidates = [c for c in candidates if 'pending' not in c.get('status', '').lower()]
        
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

def get_candidate_top_matches(candidate_name: str, limit: int = 3) -> List[Dict[str, Any]]:
    """
    Get top matching jobs for a specific candidate (reverse match)
    
    This function fetches all open positions, calculates match scores against
    the specified candidate, and returns the top N job matches sorted by score.
    
    Args:
        candidate_name: Candidate name to match against
        limit: Maximum number of matches to return (default: 3)
        
    Returns:
        List of dictionaries containing top matching jobs with scores
    """
    try:
        # Get candidate data
        candidates = merge_candidate_sources()
        target_candidate = None
        target_norm = normalize_name(candidate_name)
        
        for c in candidates:
            if normalize_name(c.get('name', '')) == target_norm:
                target_candidate = c
                break
        
        if not target_candidate:
            logger.error(f"Candidate {candidate_name} not found")
            return []
        
        # Get all open positions
        open_positions = fetch_open_positions_data()
        
        # Calculate match scores for each job
        matches = []
        for job in open_positions:
            match_result = calculate_match_score(target_candidate, job)
            
            match_data = {
                'job': job,
                'candidate': target_candidate,
                'match_score': match_result['total_score'],
                'match_quality': match_result['quality'],
                'score_breakdown': match_result['breakdown']
            }
            matches.append(match_data)
        
        # Sort by score descending and return top N
        matches.sort(key=lambda x: x['match_score'], reverse=True)
        return matches[:limit]
        
    except Exception as e:
        logger.error(f"Error getting top matches for candidate {candidate_name}: {str(e)}")
        return []


# =============================================================================
# MENTOR EFFECTIVENESS FUNCTIONS
# =============================================================================

def fetch_mentor_relationships_data() -> List[Dict[str, Any]]:
    """
    Fetch individual mentor-trainee relationship data from Google Sheets
    
    Returns:
        List of dictionaries with mentor-trainee pairs and effectiveness scores
    """
    try:
        logger.info("Fetching mentor relationships data from Google Sheets")
        df = pd.read_csv(MENTOR_RELATIONSHIPS_URL, dtype=str)
        
        # Clean column names (normalize whitespace, but DON'T apply COLUMN_MAPPING)
        # Mentor sheets have different structure than candidate sheets
        # COLUMN_MAPPING would rename "Trainee Name" -> "MIT Name" which breaks this sheet
        df.columns = (
            df.columns.astype(str)
              .str.replace('\u00A0', ' ', regex=False)  # Replace non-breaking spaces
              .str.replace(r'\s+', ' ', regex=True)      # Collapse multiple spaces
              .str.strip()
        )
        
        # Validate schema (but don't require candidate-specific columns)
        validate_schema(df, "Mentor Relationships")
        
        # Debug: Log what columns we have
        logger.info(f"Mentor Relationships columns: {list(df.columns)}")
        
        # Find the mentor and trainee name columns (handle variations)
        mentor_col = None
        trainee_col = None
        
        # Try to find the columns (handle variations in naming)
        for col in df.columns:
            col_lower = col.lower().strip()
            if 'mentor' in col_lower and 'name' in col_lower:
                mentor_col = col
            if 'trainee' in col_lower and 'name' in col_lower:
                trainee_col = col
        
        # Fallback: try exact matches
        if not mentor_col and 'Mentor Name' in df.columns:
            mentor_col = 'Mentor Name'
        if not trainee_col and 'Trainee Name' in df.columns:
            trainee_col = 'Trainee Name'
        
        if mentor_col and trainee_col:
            logger.info(f"Using columns: Mentor='{mentor_col}', Trainee='{trainee_col}'")
            # Filter out rows where both are empty
            df = df.dropna(subset=[mentor_col, trainee_col], how='all')
            # Also filter out rows where both are empty strings
            df = df[~((df[mentor_col].astype(str).str.strip() == '') & (df[trainee_col].astype(str).str.strip() == ''))]
        else:
            logger.warning(f"Missing required columns for mentor relationships. Found columns: {list(df.columns)}")
            logger.warning(f"Looking for columns containing 'mentor name' and 'trainee name'")
            return []
        
        relationships = []
        for _, row in df.iterrows():
            mentor_name = str(safe_get(row, mentor_col, '')).strip()
            trainee_name = str(safe_get(row, trainee_col, '')).strip()
            
            # Skip rows where both names are empty
            if not mentor_name and not trainee_name:
                continue
            
            relationships.append({
                'mentor_name': mentor_name,
                'trainee_name': trainee_name,
                'training_program': str(safe_get(row, 'Training Program', '')).strip(),
                'completion_status': str(safe_get(row, 'Completion Status', '')).strip(),
                'effectiveness_score': float(safe_get(row, 'Mentor Effectiveness Score', 0)) if pd.notna(safe_get(row, 'Mentor Effectiveness Score', 0)) else 0.0
            })
        
        logger.info(f"Successfully fetched {len(relationships)} mentor relationships")
        return relationships
        
    except Exception as e:
        logger.error(f"Error fetching mentor relationships data: {str(e)}", exc_info=True)
        return []


def fetch_mentor_mei_summary() -> Dict[str, Dict[str, Any]]:
    """
    Fetch aggregated mentor MEI scores from Google Sheets
    
    Returns:
        Dictionary mapping mentor names to their summary stats
    """
    try:
        logger.info("Fetching mentor MEI summary from Google Sheets")
        df = pd.read_csv(MENTOR_MEI_SUMMARY_URL, dtype=str)
        
        # Standardize columns FIRST
        df = standardize_columns(df)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Validate schema
        validate_schema(df, "Mentor MEI Summary")
        
        # Filter out empty rows
        if 'Mentor Name' in df.columns:
            df = df.dropna(subset=['Mentor Name'], how='all')
        else:
            logger.warning("Missing 'Mentor Name' column in MEI summary")
            return {}
        
        summary = {}
        for _, row in df.iterrows():
            mentor_name = str(safe_get(row, 'Mentor Name', '')).strip()
            if mentor_name:
                summary[mentor_name] = {
                    'num_trainees': int(safe_get(row, 'Number of Trainees', 0)) if pd.notna(safe_get(row, 'Number of Trainees', 0)) else 0,
                    'average_mei': float(safe_get(row, 'Average MEI', 0)) if pd.notna(safe_get(row, 'Average MEI', 0)) else 0.0
                }
        
        logger.info(f"Successfully fetched MEI summary for {len(summary)} mentors")
        return summary
        
    except Exception as e:
        logger.error(f"Error fetching mentor MEI summary: {str(e)}")
        return {}


def build_mentor_profiles() -> List[Dict[str, Any]]:
    """
    Build comprehensive mentor profiles by combining relationship and summary data
    
    Returns:
        List of mentor profile dictionaries
    """
    relationships = fetch_mentor_relationships_data()
    mei_summary = fetch_mentor_mei_summary()
    
    logger.info(f"Building mentor profiles: {len(relationships)} relationships, {len(mei_summary)} MEI summaries")
    
    # Create normalized MEI summary lookup for flexible name matching
    mei_summary_normalized = {}
    for mentor_name, data in mei_summary.items():
        normalized = normalize_name(mentor_name)
        if normalized:
            mei_summary_normalized[normalized] = (mentor_name, data)  # Store original name and data
    
    # Group relationships by mentor (using normalized names for consistency)
    mentor_trainees = {}
    mentor_name_map = {}  # Maps normalized name to original name from relationships
    
    for rel in relationships:
        mentor_name = rel['mentor_name']
        if not mentor_name or mentor_name.strip() == '':
            continue  # Skip empty mentor names
        
        normalized = normalize_name(mentor_name)
        if normalized not in mentor_trainees:
            mentor_trainees[normalized] = []
            mentor_name_map[normalized] = mentor_name  # Store original name
        
        mentor_trainees[normalized].append(rel)
    
    logger.info(f"Grouped into {len(mentor_trainees)} unique mentors from relationships")
    
    # If no relationships but we have MEI summary, create profiles from MEI summary only
    if not mentor_trainees and mei_summary:
        logger.info("No relationships found, creating profiles from MEI summary only")
        for mentor_name, summary_data in mei_summary.items():
            if not mentor_name or mentor_name.strip() == '':
                continue
            lifetime_trainees = summary_data.get('num_trainees', 0)
            profiles.append({
                'name': mentor_name,
                'average_mei': round(summary_data.get('average_mei', 0.0), 2),
                'num_trainees': 0,  # No current relationships
                'trainee_count': lifetime_trainees,  # Lifetime total from MEI summary (for frontend)
                'completed': 0,
                'in_progress': 0,
                'unsuccessful': 0,
                'tier': 'Strong Mentor' if summary_data.get('average_mei', 0) >= 2.5 else ('In Progress' if summary_data.get('average_mei', 0) >= 1.5 else 'Needs Improvement'),
                'tier_color': 'green' if summary_data.get('average_mei', 0) >= 2.5 else ('yellow' if summary_data.get('average_mei', 0) >= 1.5 else 'red'),
                'trainees': [],
                'success_rate': 0.0
            })
        profiles.sort(key=lambda x: x['average_mei'], reverse=True)
        logger.info(f"Built {len(profiles)} mentor profiles from MEI summary only")
        return profiles
    
    # Build profiles
    profiles = []
    for normalized_name, trainees in mentor_trainees.items():
        # Get original mentor name (prefer from relationships, fallback to MEI summary)
        original_name = mentor_name_map.get(normalized_name, normalized_name)
        
        # Try to find MEI summary using normalized name matching
        summary = {}
        if normalized_name in mei_summary_normalized:
            original_mei_name, summary = mei_summary_normalized[normalized_name]
            # Use the original name from MEI summary if available (more consistent)
            if original_mei_name:
                original_name = original_mei_name
        
        avg_mei = summary.get('average_mei', 0.0)
        
        # If no summary, calculate from relationships
        if avg_mei == 0 and trainees:
            scores = [t['effectiveness_score'] for t in trainees if t['effectiveness_score'] > 0]
            avg_mei = sum(scores) / len(scores) if scores else 0.0
        
        # Count completion statuses (case-insensitive matching)
        completed = 0
        in_progress = 0
        unsuccessful = 0
        
        logger.debug(f"Processing {len(trainees)} trainees for mentor {original_name}")
        for t in trainees:
            status = str(t.get('completion_status', '')).lower().strip()
            trainee_name = t.get('trainee_name', 'Unknown')
            
            if any(x in status for x in ['complete', 'completed', 'successful', 'graduated']):
                completed += 1
                logger.debug(f"  {trainee_name}: COMPLETED (status: '{t.get('completion_status', '')}')")
            elif any(x in status for x in ['currently', 'pending', 'in progress', 'active', 'training']):
                in_progress += 1
                logger.debug(f"  {trainee_name}: IN PROGRESS (status: '{t.get('completion_status', '')}')")
            elif any(x in status for x in ['removed', 'resigned', 'incomplete', 'failed', 'never', 'unsuccessful', 'terminated']):
                unsuccessful += 1
                logger.debug(f"  {trainee_name}: UNSUCCESSFUL (status: '{t.get('completion_status', '')}')")
            else:
                # Default to in-progress if status is unclear
                in_progress += 1
                logger.debug(f"  {trainee_name}: DEFAULT TO IN PROGRESS (status: '{t.get('completion_status', '')}')")
        
        logger.debug(f"Mentor {original_name}: completed={completed}, in_progress={in_progress}, unsuccessful={unsuccessful}")
        
        # Get lifetime trainee count from MEI summary (more accurate than len(trainees))
        lifetime_trainees = summary.get('num_trainees', len(trainees))
        
        # Calculate success rate: completed / (completed + unsuccessful) * 100
        # Only count finished trainees (exclude in-progress) in denominator
        finished_count = completed + unsuccessful
        success_rate = round((completed / finished_count * 100) if finished_count > 0 else 0.0, 1)
        
        # Determine mentor tier
        if avg_mei >= 2.5:
            tier = 'Strong Mentor'
            tier_color = 'green'
        elif avg_mei >= 1.5:
            tier = 'In Progress'
            tier_color = 'yellow'
        else:
            tier = 'Needs Improvement'
            tier_color = 'red'
        
        profiles.append({
            'name': original_name,  # Use original name (from MEI summary if available)
            'average_mei': round(avg_mei, 2),
            'num_trainees': len(trainees),  # Current active relationships
            'trainee_count': lifetime_trainees,  # Lifetime total from MEI summary (for frontend)
            'completed': completed,
            'in_progress': in_progress,
            'unsuccessful': unsuccessful,
            'tier': tier,
            'tier_color': tier_color,
            'trainees': trainees,
            'success_rate': success_rate
        })
    
    # Sort by average MEI descending
    profiles.sort(key=lambda x: x['average_mei'], reverse=True)
    
    logger.info(f"Built {len(profiles)} mentor profiles")
    return profiles


def get_mentor_dashboard_metrics() -> Dict[str, Any]:
    """
    Calculate dashboard metrics for mentor block
    
    Returns:
        Dictionary with mentor metrics for dashboard display
    """
    profiles = build_mentor_profiles()
    
    if not profiles:
        return {
            'total_mentors': 0,
            'average_mei': 0.0,
            'strong_mentors': 0,
            'needs_improvement': 0
        }
    
    total = len(profiles)
    avg_mei = sum(p['average_mei'] for p in profiles) / total if total > 0 else 0.0
    strong = sum(1 for p in profiles if p['average_mei'] >= 2.5)
    needs_improvement = sum(1 for p in profiles if p['average_mei'] < 1.5)
    
    return {
        'total_mentors': total,
        'average_mei': round(avg_mei, 2),
        'strong_mentors': strong,
        'needs_improvement': needs_improvement
    }


def get_active_training_mentors() -> Dict[str, Any]:
    """
    Get mentors who are actively training current MIT candidates
    Only includes candidates with status 'In Training'
    
    Returns:
        Dictionary with training mentor statistics and lists grouped by performance
    """
    try:
        logger.info("Building active training mentors list")
        candidates = merge_candidate_sources()
        mentor_trainee_map = {}
        
        # Build mentor-trainee relationships for "In Training" candidates only
        for candidate in candidates:
            status = str(candidate.get('status', '')).strip().lower()
            # Include both "training" and candidates in weeks 0-22
            week = candidate.get('week', 0)
            if status == 'training' or (week >= 0 and week <= 22 and status not in ['pending start', 'ready'] and 'pending' not in status):
                mentor_name = str(candidate.get('mentor_name', '')).strip()
                # Filter out invalid mentor names
                if mentor_name and mentor_name.lower() not in ['nan', 'none', '', 'n/a', 'tbd', '—']:
                    if mentor_name not in mentor_trainee_map:
                        mentor_trainee_map[mentor_name] = {
                            'trainees': [],
                            'mei_score': 0.0,
                            'mentor_title': str(candidate.get('mentor_title', 'Mentor')).strip()
                        }
                    mentor_trainee_map[mentor_name]['trainees'].append({
                        'name': candidate.get('name', ''),
                        'week': candidate.get('week', 0)
                    })
        
        logger.info(f"Found {len(mentor_trainee_map)} mentors with active trainees")
        
        # Get MEI scores for these mentors
        mei_data = fetch_mentor_mei_summary()
        for mentor_name in mentor_trainee_map:
            normalized_name = normalize_name(mentor_name)
            # Try to find MEI score by normalized name match
            for mei_mentor_name, mei_info in mei_data.items():
                if normalize_name(mei_mentor_name) == normalized_name:
                    mentor_trainee_map[mentor_name]['mei_score'] = mei_info.get('average_mei', 0.0)
                    break
        
        # Categorize mentors by performance
        strong = []
        needs_help = []
        monitoring = []
        
        # Fetch mentor relationships to calculate accurate lifetime trainee counts
        relationships = fetch_mentor_relationships_data()
        
        for mentor_name, data in mentor_trainee_map.items():
            mei = data['mei_score']
            
            # Calculate lifetime trainee count and success rate from Relationships sheet
            normalized_name = normalize_name(mentor_name)
            lifetime_count = 0
            completed_count = 0
            unsuccessful_count = 0
            in_progress_count = 0
            
            for rel in relationships:
                rel_mentor = str(rel.get('mentor_name', '')).strip()
                if rel_mentor and normalize_name(rel_mentor) == normalized_name:
                    lifetime_count += 1
                    completion_status = str(rel.get('completion_status', '')).strip().lower()
                    
                    # Count completed trainees
                    if completion_status in ['completed', 'complete', 'successful', 'graduated']:
                        completed_count += 1
                    # Count unsuccessful trainees
                    elif completion_status in ['unsuccessful', 'removed', 'incomplete', 'failed', 'resigned', 'terminated']:
                        unsuccessful_count += 1
                    # Everything else is in progress
                    else:
                        in_progress_count += 1
            
            # Calculate success rate - ONLY from finished trainees (exclude in-progress)
            finished_count = completed_count + unsuccessful_count
            success_rate = (completed_count / finished_count * 100) if finished_count > 0 else 0
            
            # Fallback to current active count if no relationships found
            if lifetime_count == 0:
                lifetime_count = len(data['trainees'])
                success_rate = 0
            
            mentor_info = {
                'name': mentor_name,
                'trainees': data['trainees'],
                'mei_score': mei,
                'title': data['mentor_title'],
                'trainee_count': lifetime_count,  # Lifetime total from Relationships sheet
                'success_rate': success_rate  # Success rate percentage
            }
            
            # Categorize based on MEI score
            if mei >= 2.5:
                strong.append(mentor_info)
            elif mei < 2.0 and mei > 0:
                needs_help.append(mentor_info)
            else:
                monitoring.append(mentor_info)
        
        # Sort each category by MEI score (descending)
        strong.sort(key=lambda x: x['mei_score'], reverse=True)
        needs_help.sort(key=lambda x: x['mei_score'])
        monitoring.sort(key=lambda x: x['mei_score'], reverse=True)
        
        logger.info(f"Categorized: {len(strong)} strong, {len(needs_help)} needs help, {len(monitoring)} monitoring")
        
        return {
            'total_training_mentors': len(mentor_trainee_map),
            'strong_mentors': strong,
            'needs_help_mentors': needs_help,
            'monitoring_mentors': monitoring,
            'total_trainees': sum(len(m['trainees']) for m in mentor_trainee_map.values())
        }
        
    except Exception as e:
        logger.error(f"Error getting active training mentors: {str(e)}")
        return {
            'total_training_mentors': 0,
            'strong_mentors': [],
            'needs_help_mentors': [],
            'monitoring_mentors': [],
            'total_trainees': 0
        }

def get_mit_alumni() -> Dict[str, Any]:
    """
    Get MIT alumni with placement information
    
    Logic:
    1. Fetch placed MITs sheet (master list - source of truth for who graduated)
    2. For each MIT graduate, try to find them in Fallback sheet
    3. Merge placement data (from Placed MITs) + training data (from Fallback)
    4. Return list of alumni profiles with full information
    
    Returns:
        Dictionary with total count and list of alumni profiles
    """
    try:
        logger.info("Building MIT alumni list")
        
        # Step 1: Get placed MITs (master list)
        placed_mits = fetch_placed_mits_data()
        
        if not placed_mits:
            logger.warning("No placed MITs found")
            return {'total_alumni': 0, 'alumni': []}
        
        # Step 2: Get fallback data for enrichment
        fallback_df = fetch_fallback_candidate_data()
        
        # Step 3: Get current MIT tracking candidates (to exclude from alumni)
        current_candidates = merge_candidate_sources()
        current_names = {normalize_name(c['name']) for c in current_candidates}
        
        alumni_profiles = []
        
        for placed_mit in placed_mits:
            name = placed_mit['name']
            normalized_name = normalize_name(name)
            
            # EXCLUSION LOGIC: Skip if they're still in current MIT tracking
            if normalized_name in current_names:
                logger.info(f"Skipping {name} - still in active MIT tracking")
                continue
            
            # Start with placement data from Placed MITs sheet
            alumni_profile = {
                'name': name,
                # Placement information (priority - from Placed MITs sheet)
                'placement_site': placed_mit.get('placement_site', 'TBD'),
                'placement_title': placed_mit.get('placement_title', 'TBD'),
                'placement_start_date': placed_mit.get('placement_start_date', 'TBD'),
                'weeks_in_program': placed_mit.get('weeks_in_program', 'TBD'),
                # Training information (from Placed MITs sheet)
                'training_site': placed_mit.get('training_site', 'TBD'),
                'training_location': placed_mit.get('training_location', 'TBD'),
                'training_salary': placed_mit.get('training_salary', 'TBD'),
                'training_vertical': placed_mit.get('training_vertical', 'TBD'),
                'training_start_date': placed_mit.get('training_start_date', 'TBD'),
                'level': placed_mit.get('level', 'TBD'),
                'confidence': placed_mit.get('confidence', 'TBD'),
                'notes': placed_mit.get('notes', ''),
                # Defaults for enrichment data (will be overridden if found in Fallback)
                'mentor_name': 'TBD',
                'mei_score': 0.0,
                'mock_qbr_score': 0.0,
                'assessment_score': 0.0,
                'perf_eval_score': 0.0,
                'confidence_score': 0.0,
                'skill_ranking': 'TBD',
                'graduation_date': 'TBD',
                'image': resolve_headshot_path(name),
                'bio': get_bio_for_name(name)
            }
            
            # Step 4: Try to enrich with data from Fallback sheet
            fallback_row = find_candidate_in_sheet(name, fallback_df)
            
            if fallback_row is not None:
                logger.info(f"Found enrichment data for {name} in Fallback sheet")
                
                # Add mentor information
                alumni_profile['mentor_name'] = str(safe_get(fallback_row, 'Mentor Name', 'TBD')).strip()
                
                # Add scores if available
                alumni_profile['mock_qbr_score'] = float(pd.to_numeric(safe_get(fallback_row, 'Mock QBR Score', 0), errors='coerce') or 0)
                alumni_profile['assessment_score'] = float(pd.to_numeric(safe_get(fallback_row, 'Assessment Score', 0), errors='coerce') or 0)
                alumni_profile['perf_eval_score'] = float(pd.to_numeric(safe_get(fallback_row, 'Perf Evaluation Score', 0), errors='coerce') or 0)
                alumni_profile['confidence_score'] = float(pd.to_numeric(safe_get(fallback_row, 'Confidence Score', 0), errors='coerce') or 0)
                
                skill_rank = str(safe_get(fallback_row, 'Skill Ranking', 'TBD')).strip()
                alumni_profile['skill_ranking'] = skill_rank if skill_rank.lower() not in ['nan', 'none', ''] else 'TBD'
                
                # Add graduation date if available
                grad_date = str(safe_get(fallback_row, 'Graduation Date', '')).strip()
                if grad_date and grad_date.lower() not in ['nan', 'none', '']:
                    alumni_profile['graduation_date'] = grad_date
            else:
                logger.info(f"No enrichment data found for {name} in Fallback sheet - using Placed MITs data only")
            
            alumni_profiles.append(alumni_profile)
        
        logger.info(f"Built {len(alumni_profiles)} MIT alumni profiles")
        
        return {
            'total_alumni': len(alumni_profiles),
            'alumni': alumni_profiles
        }
        
    except Exception as e:
        logger.error(f"Error getting MIT alumni: {str(e)}")
        return {
            'total_alumni': 0,
            'alumni': []
        }

def fetch_meeting_insights() -> List[str]:
    """
    Fetch weekly meeting bullet points from Google Sheets
    
    Returns:
        List of insight strings (bullet points from the sheet)
    """
    try:
        import requests
        response = requests.get(MEETING_INSIGHTS_URL, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        
        # Read CSV into DataFrame
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
        
        insights = []
        # Get the first column (should be "Meeting Notes")
        first_column = df.columns[0]
        
        for _, row in df.iterrows():
            insight = str(safe_get(row, first_column, '')).strip()
            # Only add non-empty insights (skip header row and empty rows)
            if insight and insight.lower() not in ['nan', 'none', '', 'n/a', 'meeting notes', 'meeting note']:
                insights.append(insight)
        
        logger.info(f"Fetched {len(insights)} meeting insights")
        return insights
    except Exception as e:
        logger.error(f"Error fetching meeting insights: {str(e)}")
        return []
