"""
Configuration file for MIT Dashboard API Server
==============================================

This file contains all configuration settings for the dashboard system.
Modify these settings to customize the behavior of the API server.

Author: AI Assistant
Date: October 2025
"""

# =============================================================================
# GOOGLE SHEETS CONFIGURATION
# =============================================================================

# Google Sheets URL for the main data source
GOOGLE_SHEETS_URL = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vRH719lKRRdQV8Y4CEEl7Gk-lfrAGulMcOgu3sltQ7zupMRDlP3Rpgaa-sEJlRTNqrRsTuPNcOswlv9/pub?gid=0&single=true&output=csv'

# Google Sheets URL for open positions data
OPEN_POSITIONS_URL = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vTAdbdhuieyA-axzb4aLe8c7zdAYXBLPNrIxKRder6j1ZAlj2g4U1k0YzkZbm_dEcSwBik4CJ57FROJ/pub?gid=1073524035&single=true&output=csv'

# Master list: MIT Tracking Sheet (active candidates; two-section layout)
MIT_TRACKING_SHEET_URL = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vTAdbdhuieyA-axzb4aLe8c7zdAYXBLPNrIxKRder6j1ZAlj2g4U1k0YzkZbm_dEcSwBik4CJ57FROJ/pub?gid=813046237&single=true&output=csv'

# Fallback/historical sheet (similar to main but with an extra leading column)
FALLBACK_CANDIDATE_SHEET_URL = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vRH719lKRRdQV8Y4CEEl7Gk-lfrAGulMcOgu3sltQ7zupMRDlP3Rpgaa-sEJlRTNqrRsTuPNcOswlv9/pub?gid=1149285782&single=true&output=csv'

# Tier 1 Managers Sheet (completed MIT/SMIT - same as fallback sheet)
TIER1_MANAGERS_SHEET_URL = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vRH719lKRRdQV8Y4CEEl7Gk-lfrAGulMcOgu3sltQ7zupMRDlP3Rpgaa-sEJlRTNqrRsTuPNcOswlv9/pub?gid=1149285782&single=true&output=csv'

# Mentor Effectiveness Data URLs
MENTOR_RELATIONSHIPS_URL = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vRH719lKRRdQV8Y4CEEl7Gk-lfrAGulMcOgu3sltQ7zupMRDlP3Rpgaa-sEJlRTNqrRsTuPNcOswlv9/pub?gid=1269990929&single=true&output=csv'
MENTOR_MEI_SUMMARY_URL = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vRH719lKRRdQV8Y4CEEl7Gk-lfrAGulMcOgu3sltQ7zupMRDlP3Rpgaa-sEJlRTNqrRsTuPNcOswlv9/pub?gid=1759772785&single=true&output=csv'

# MIT Alumni/Placed MITs Sheet (MIT graduates with placement information)
PLACED_MITS_URL = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vTAdbdhuieyA-axzb4aLe8c7zdAYXBLPNrIxKRder6j1ZAlj2g4U1k0YzkZbm_dEcSwBik4CJ57FROJ/pub?gid=1835369405&single=true&output=csv'

# Weekly Meeting Insights Sheet
MEETING_INSIGHTS_URL = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSGsWrJ7HL1EreSYuF2595dGglPSXtKAMrhL064HfvFJPi9E_4Usz5Pr7HTJhAzBhRptNLxX-epCjV9/pub?gid=0&single=true&output=csv'

# Client Requests for MIT Placements (hiring pipeline)
CLIENT_MIT_REQUESTS_URL = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vTAdbdhuieyA-axzb4aLe8c7zdAYXBLPNrIxKRder6j1ZAlj2g4U1k0YzkZbm_dEcSwBik4CJ57FROJ/pub?gid=1506398133&single=true&output=csv'

# Transition Master List CSV (local file)
TRANSITION_MASTER_CSV = 'Transition Master List 2025 Report (1).csv'

# Transitions Tracker CSV (local file)
TRANSITION_TRACKER_CSV = 'Transitions Tracker.csv'

# =============================================================================
# TRAINING PROGRAM CONFIGURATION
# =============================================================================

# Target training programs to include in the dashboard
TARGET_PROGRAMS = ['MIT', 'SMIT']

# =============================================================================
# BUSINESS LESSON CONFIGURATION
# =============================================================================

# Business lesson columns (T-Y in Google Sheets)
# These columns contain dates that determine lesson completion
BUSINESS_LESSON_COLUMNS = [
    "SBM Business +KPI's",
    'Contracts 101 + Custodial Process', 
    'Quality Control +Safety',
    '4Insite Strategy',
    'Customer + Employee Engagement',
    'Inventory + At Risk Management'
]

# =============================================================================
# COLUMN MAPPING CONFIGURATION
# =============================================================================

# Mapping from Google Sheets column names to dashboard field names
COLUMN_MAPPING = {
    "Company Start Date": "Company Start Date",  # Keep original name for display AND week calculation
    "Training Start Date": "Training Start Date",  # Keep separate for display
    # Handle potential variants with extra spaces
    "Training Start Date ": "Training Start Date",  # Handle trailing space variant
    " Training Start Date": "Training Start Date",  # Handle leading space variant
    "Trainee Name": "MIT Name",
    "Ops Account- Location": "Training Site",
    # Performance metrics columns
    "Mock QBR Score": "Mock QBR Score",  # Mock QBR Score column (Row AA)
    "Perf Evaluation Score": "Perf Evaluation Score",  # Performance Evaluation column (Row AF)
    "Skill Ranking": "Skill Ranking",  # Skill Ranking column (Row AK)
    # Job matching algorithm columns
    # Normalize any 'Confidence' header variant to the canonical 'Confidence Score'
    "Confidence Score": "Confidence Score",
    "Confidence": "Confidence Score",
    "Vertical": "Vertical",    # Column AV for vertical alignment
    "AT": "Training Location",  # Column AT for training location/background
    # Resume column
    "Resume": "Resume",  # Column AZ - Resume URLs
    # Graduation week column - exact name from Google Sheets
    "Initial Expected Graduation Week": "Expected Graduation Week",  # Column AO - Exact name from your Google Sheets
    # OIG completion column - exact name from Google Sheets
    "Mentor Certification of Trainee's OIG Completion": "OIG Completion",  # Column BB - "Yes" if completed
    # Mock QBR scheduling
    "Mock QBR Date (MIT Only)": "Mock QBR Date"  # Column AD - Date of scheduled Mock QBR
}

# =============================================================================
# OIG (ON-SITE INTEGRATION) CONFIGURATION
# =============================================================================

# OIG program start date - candidates who started before this are exempt
OIG_START_DATE = "2025-11-17"  # November 17, 2025

# =============================================================================
# ONBOARDING TASKS CONFIGURATION
# =============================================================================

# Onboarding tasks used for progress calculation (columns N-Q only)
ONBOARDING_TASKS = [
    'Intro Call Invite Sent?',
    'Intro Call Completed?',
    'Welcome Packet Sent?',
    'Acknowledgements'
]

# =============================================================================
# SCORE COLUMNS CONFIGURATION
# =============================================================================

# Score columns used for performance metrics
SCORE_COLUMNS = {
    'mock_qbr_score': 'Mock QBR Score',
    'assessment_score': 'Assessment Score',
    'perf_eval_score': 'Perf Evaluation Score',
    'confidence_score': 'Confidence Score',
    'skill_ranking': 'Skill Ranking'
}

# =============================================================================
# REQUIRED COLUMNS CONFIGURATION
# =============================================================================

# Core required columns that must exist for the dashboard to function
REQUIRED_COLUMNS = [
    'MIT Name',  # Essential for candidate identification
    'Training Program',  # Essential for filtering
]

# Important columns (warn if missing but don't stop execution)
IMPORTANT_COLUMNS = [
    'Company Start Date',  # Used for week calculation
    'Status',  # Used for status derivation
    'Salary',  # Used for salary parsing
    'Week',  # Used for categorization
]

# =============================================================================
# CACHE CONFIGURATION
# =============================================================================

# Cache duration in minutes (how long to keep data in memory)
CACHE_DURATION_MINUTES = 2  # Cache for 2 minutes - balance between performance and freshness
FORCE_FRESH_DATA = False  # Use cached data when available for better performance

# =============================================================================
# DEBUG CONFIGURATION
# =============================================================================

# Enable debug mode for detailed logging
DEBUG_MODE = True

# =============================================================================
# SERVER CONFIGURATION
# =============================================================================

# Flask server configuration
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 8000
SERVER_DEBUG = False

# =============================================================================
# API CONFIGURATION
# =============================================================================

# API version
API_VERSION = '2.0.0'

# Request timeout in seconds
REQUEST_TIMEOUT = 30

# =============================================================================
# COMPLETION KEYWORDS
# =============================================================================

# Keywords that indicate task completion
COMPLETION_KEYWORDS = ['yes', 'y', 'true', '1', 'completed', 'x']

# =============================================================================
# ERROR MESSAGES
# =============================================================================

ERROR_MESSAGES = {
    'candidate_not_found': 'Candidate not found',
    'data_fetch_error': 'Error fetching data from Google Sheets',
    'invalid_request': 'Invalid request parameters',
    'server_error': 'Internal server error'
}
