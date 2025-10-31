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
                elif score_key == 'mock_qbr_score':
                    # Mock QBR Score is out of 4, not 100
                    scores[score_key] = float(val_str) if val_str else 0.0
                    logger.info(f"    -> SUCCESS: {scores[score_key]} (Mock QBR out of 4 from '{score_value}')")
                else:
                    # Other numeric fields (out of 100)
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
        
        # Debug: Log available columns
        logger.info(f"Available columns in open positions: {list(df.columns)}")
        
        # Clean up the data - remove any completely empty rows
        df = df.dropna(how='all')
        
        # Filter out rows where Job Title is empty or NaN
        df = df[df['Job Title'].notna() & (df['Job Title'] != '')]
        
        positions = []
        for _, row in df.iterrows():
            # Parse location from Column F (Location) - try different possible column names
            location = ''
            for col_name in ['Location', 'location', 'LOCATION', 'City', 'city', 'CITY']:
                if col_name in df.columns:
                    location = str(row.get(col_name, ''))
                    if location and location != 'nan':
                        break
            
            city, state = '', ''
            if location and location != 'nan':
                # Split "Detroit, MI" into city and state
                parts = location.split(',')
                if len(parts) >= 2:
                    city = parts[0].strip()
                    state = parts[1].strip()
                else:
                    city = location.strip()
            
            position = {
                'id': int(row.get('JV ID', 0)) if pd.notna(row.get('JV ID')) else 0,
                'title': str(row.get('Job Title', '')),
                'jv_id': str(row.get('JV ID', '')),
                'jv_link': str(row.get('JV Link', '')),
                'vertical': str(row.get('VERT', '')),
                'account': str(row.get('Account', '')),
                'city': city,
                'state': state,
                'salary': parse_salary(row.get('Salary', 0))
            }
            positions.append(position)
        
        logger.info(f"Successfully fetched {len(positions)} open positions")
        return positions
        
    except Exception as e:
        logger.error(f"Error fetching open positions data: {str(e)}")
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
        
        # Standardize column names
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
        
        # Filter out empty rows and header rows
        df_data = df_data[df_data['MIT Name'].notna()]
        df_data = df_data[df_data['MIT Name'].astype(str).str.strip().str.lower() != 'mit name']
        df_data = df_data[df_data['MIT Name'].astype(str).str.strip() != '']
        df_data = df_data[df_data['MIT Name'].astype(str).str.strip() != 'New Candidate Name']  # Remove header row from second section
        
        # Handle the two sections
        # Mark candidates from second section as offer pending
        if 'JV' in df_data.columns:
            df_data.loc[df_data['JV'].notna(), 'Status'] = 'Offer Pending'
        
        # Also mark rows where Status is empty as Offer Pending (for second section)
        df_data['Status'] = df_data['Status'].fillna('')
        df_data.loc[(df_data['Status'] == '') & (df_data['MIT Name'].notna()), 'Status'] = 'Offer Pending'
        
        # Normalize Week to numeric
        if 'Week' in df_data.columns:
            df_data['Week'] = pd.to_numeric(df_data['Week'], errors='coerce').fillna(0).astype(int)
        
        # Ensure all expected columns exist
        expected_cols = ['MIT Name', 'Week', 'Start date', 'VERT', 'Training Site', 'Location', 'Salary', 'Level', 'Status', 'Confidence', 'Mentor', 'Notes']
        for col in expected_cols:
            if col not in df_data.columns:
                df_data[col] = None
        
        logger.info(f"MIT Tracking data: Found {len(df_data)} candidates")
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
        
        # Now apply the normal column mapping
        norm = lambda s: re.sub(r'\s+', ' ', s.replace('\u00A0', ' ').strip().lower())
        mapping_norm = {norm(k): v for k, v in COLUMN_MAPPING.items()}
        df.rename(columns=lambda c: mapping_norm.get(norm(c), c), inplace=True)

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
        if normalize_name(row.get('MIT Name', '')) == target:
            return row

    # Fuzzy with mentor boost
    for _, row in df.iterrows():
        nm = row.get('MIT Name', '')
        if fuzzy_match_name(name, nm, threshold=0.85):
            return row
        # If mentor matches, accept with lower name threshold
        if hint_mentor and mentor_match(hint_mentor, row.get('Mentor Name', '')):
            if fuzzy_match_name(name, nm, threshold=0.78):
                return row

    return None

def create_basic_profile_from_mit(tracking_row: pd.Series) -> Dict[str, Any]:
    """
    Create a minimal candidate profile from MIT Tracking fields.
    """
    name_value = str(tracking_row.get('MIT Name', '—'))
    return {
        'name': name_value,
        'training_site': str(tracking_row.get('Training Site', '—')),
        'location': str(tracking_row.get('Location', '—')),
        'week': int(pd.to_numeric(tracking_row.get('Week', 0), errors='coerce') or 0),
        'expected_graduation_week': '—',
        'salary': parse_salary(tracking_row.get('Salary', 0)),
        'status': str(tracking_row.get('Status', 'offer_pending')).lower(),
        'training_program': 'MIT',
        'mentor_name': str(tracking_row.get('Mentor', '—')),
        'mentor_title': '—',
        'bio': get_bio_for_name(name_value),
        'scores': {},
        'onboarding_progress': None,
        'business_lessons_progress': None,
        'operation_details': {
            'company_start_date': str(tracking_row.get('Start date', '—')),
            'training_start_date': '—',
            'title': '—',
            'operation_location': str(tracking_row.get('Training Site', '—')),
            'vertical': str(tracking_row.get('VERT', '—'))
        },
        'resume_link': '',
        'profile_image': resolve_headshot_path(name_value),
        'data_quality': 'limited'
    }

def merge_candidate_sources() -> List[Dict[str, Any]]:
    """
    Build a unified candidate list from MIT Tracking (master), Main sheet, and Fallback sheet.
    Matching uses exact normalized name, then fuzzy.
    """
    mit_df = fetch_mit_tracking_data()
    main_df = fetch_google_sheets_data()
    fallback_df = fetch_fallback_candidate_data()

    unified: List[Dict[str, Any]] = []

    for _, trow in mit_df.iterrows():
        candidate_name = trow.get('MIT Name', '')
        mentor_name = trow.get('Mentor', '')
        if not str(candidate_name).strip():
            continue

        # Try main with mentor hint
        found = find_candidate_in_sheet(candidate_name, main_df, hint_mentor=mentor_name)
        if found is not None:
            cand = process_candidate_data(found)
            cand['data_quality'] = 'full'
            unified.append(cand)
            continue

        # Try fallback with mentor hint
        found = find_candidate_in_sheet(candidate_name, fallback_df, hint_mentor=mentor_name)
        if found is not None:
            cand = process_candidate_data(found)
            cand['data_quality'] = 'archive'
            unified.append(cand)
            continue

        # Basic profile
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
        
        # --- Normalize headers: fix non-breaking spaces, spacing, and case issues ---
        df.columns = (
            df.columns.astype(str)
              .str.replace('\u00A0', ' ', regex=False)   # Replace non-breaking spaces
              .str.replace(r'\s+', ' ', regex=True)      # Collapse multiple spaces
              .str.strip()
        )
        
        # Apply case-insensitive renaming using COLUMN_MAPPING
        import re
        norm = lambda s: re.sub(r'\s+', ' ', s.replace('\u00A0', ' ').strip().lower())
        mapping_norm = {norm(k): v for k, v in COLUMN_MAPPING.items()}
        df.rename(columns=lambda c: mapping_norm.get(norm(c), c), inplace=True)
        
        # 🧩 DEBUG LOG
        logger.info(f"[COLUMNS AFTER NORMALIZATION] {list(df.columns)}")
        
        # Filter for target programs
        df = df[df['Training Program'].isin(TARGET_PROGRAMS)]
        
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
# WEEK-BASED CATEGORIZATION
# =============================================================================

def categorize_candidates_by_week(candidates: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Categorize candidates by training week for pipeline stages.
    Uses Company Start Date to calculate current week.
    
    Returns:
        Dict with keys: 'weeks_0_3', 'week_7_only', 'weeks_8_plus', 'offer_pending', 
                       'total_candidates', 'total_training'
    """
    weeks_0_3 = []
    weeks_4_6 = []
    week_7_only = []  # Week 7 for placement priority
    weeks_8_plus = []  # Weeks 8+ for ready for placement
    offer_pending = []
    
    for candidate in candidates:
        status = candidate.get('status', '').lower()
        week = candidate.get('week', 0)
        
        # Offer pending takes priority
        if 'offer' in status or 'pending' in status:
            offer_pending.append(candidate)
        elif 0 <= week <= 3:
            weeks_0_3.append(candidate)
        elif 4 <= week <= 6:
            weeks_4_6.append(candidate)
        elif week == 7:
            week_7_only.append(candidate)
        elif week >= 8:
            weeks_8_plus.append(candidate)
    
    return {
        'weeks_0_3': weeks_0_3,  # Operational Overview (Weeks 0-3)
        'weeks_4_6': weeks_4_6,  # Active Training (Weeks 4-6)
        'week_7_only': week_7_only,  # Week 7 Priority
        'weeks_8_plus': weeks_8_plus,  # Ready for Placement (Weeks 8+)
        'offer_pending': offer_pending,  # Offer Pending
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
    week_value = calculate_week_from_start_date(row.get('Company Start Date'))
    
    # Parse salary
    salary_value = parse_salary(row.get('Salary', 0))
    
    # Extract real scores
    real_scores = extract_real_scores(row)
    
    # Calculate progress metrics
    onboarding_progress = calculate_onboarding_progress(row)
    business_lessons_progress = calculate_business_lessons_progress(row)
    
    # Resolve headshot path from candidate name
    candidate_name = str(row.get('MIT Name', 'Unknown'))
    profile_image_path = resolve_headshot_path(candidate_name)
    
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
        'bio': get_bio_for_name(candidate_name),
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
    
    # Create detailed breakdown with explanations
    score_breakdown['vertical_alignment'] = {
        'score': vertical_score,
        'max': 40,
        'explanation': f"{'Perfect match' if vertical_score >= 30 else 'No match'} - {candidate_vertical} vs {job_vertical}" + 
                      (f" (+{vertical_score-30} bonus for Amazon/Aviation)" if vertical_score > 30 else "")
    }
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
    
    score_breakdown['salary_trajectory'] = {
        'score': salary_score,
        'max': 25,
        'explanation': f"Job offers {'5%+ increase' if salary_score == 25 else 'similar pay' if salary_score == 15 else 'lower pay' if salary_score == -10 else 'no data'}"
    }
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
    
    score_breakdown['geographic_fit'] = {
        'score': geo_score,
        'max': 20,
        'explanation': f"{'Same city' if geo_score == 20 else 'Same state' if geo_score == 10 else 'Different location' if geo_score == 5 else 'No location data'}"
    }
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
    
    score_breakdown['confidence'] = {
        'score': confidence_score,
        'max': 15,
        'explanation': f"Candidate confidence: {confidence_value or 'moderate'}"
    }
    total_score += confidence_score
    
    # 5. Readiness (10 pts)
    readiness_score = 5  # Default
    week = candidate.get('week', 0)
    
    if week >= 6:
        readiness_score = 10
    elif week > 0:
        readiness_score = min(10, week * 1.5)
    
    score_breakdown['readiness'] = {
        'score': readiness_score,
        'max': 10,
        'explanation': f"Week {week} of training"
    }
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
        
        # Exclude offer pending candidates from matching
        candidates = [c for c in candidates if c.get('status', '').lower() != 'offer_pending']
        
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
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Filter out empty rows
        df = df.dropna(subset=['Mentor Name', 'Trainee Name'], how='all')
        
        relationships = []
        for _, row in df.iterrows():
            relationships.append({
                'mentor_name': str(row.get('Mentor Name', '')).strip(),
                'trainee_name': str(row.get('Trainee Name', '')).strip(),
                'training_program': str(row.get('Training Program', '')).strip(),
                'completion_status': str(row.get('Completion Status', '')).strip(),
                'effectiveness_score': float(row.get('Mentor Effectiveness Score', 0)) if pd.notna(row.get('Mentor Effectiveness Score')) else 0.0
            })
        
        logger.info(f"Successfully fetched {len(relationships)} mentor relationships")
        return relationships
        
    except Exception as e:
        logger.error(f"Error fetching mentor relationships data: {str(e)}")
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
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Filter out empty rows
        df = df.dropna(subset=['Mentor Name'], how='all')
        
        summary = {}
        for _, row in df.iterrows():
            mentor_name = str(row.get('Mentor Name', '')).strip()
            if mentor_name:
                summary[mentor_name] = {
                    'num_trainees': int(row.get('Number of Trainees', 0)) if pd.notna(row.get('Number of Trainees')) else 0,
                    'average_mei': float(row.get('Average MEI', 0)) if pd.notna(row.get('Average MEI')) else 0.0
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
    
    # Group relationships by mentor
    mentor_trainees = {}
    for rel in relationships:
        mentor_name = rel['mentor_name']
        if mentor_name not in mentor_trainees:
            mentor_trainees[mentor_name] = []
        mentor_trainees[mentor_name].append(rel)
    
    # Build profiles
    profiles = []
    for mentor_name, trainees in mentor_trainees.items():
        # Get MEI summary or calculate from relationships
        summary = mei_summary.get(mentor_name, {})
        avg_mei = summary.get('average_mei', 0.0)
        
        # If no summary, calculate from relationships
        if avg_mei == 0 and trainees:
            scores = [t['effectiveness_score'] for t in trainees if t['effectiveness_score'] > 0]
            avg_mei = sum(scores) / len(scores) if scores else 0.0
        
        # Count completion statuses
        completed = sum(1 for t in trainees if 'Complete' in t['completion_status'])
        in_progress = sum(1 for t in trainees if 'Currently' in t['completion_status'] or 'Pending' in t['completion_status'])
        unsuccessful = sum(1 for t in trainees if any(x in t['completion_status'] for x in ['Removed', 'Resigned', 'Incomplete', 'Never']))
        
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
            'name': mentor_name,
            'average_mei': round(avg_mei, 2),
            'num_trainees': len(trainees),
            'completed': completed,
            'in_progress': in_progress,
            'unsuccessful': unsuccessful,
            'tier': tier,
            'tier_color': tier_color,
            'trainees': trainees,
            'success_rate': round((completed / len(trainees) * 100) if trainees else 0, 1)
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
