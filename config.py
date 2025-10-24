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
GOOGLE_SHEETS_URL = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vR0w31eBwBrasgaLS2h9e_Bj8GWC0SqikQ0R_cuV0_B12HxOzDPLJrZm8MWaNf-7zudxrrZfLXNPR3L/pub?gid=0&single=true&output=csv'

# Google Sheets URL for open positions data
OPEN_POSITIONS_URL = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vTAdbdhuieyA-axzb4aLe8c7zdAYXBLPNrIxKRder6j1ZAlj2g4U1k0YzkZbm_dEcSwBik4CJ57FROJ/pub?gid=1073524035&single=true&output=csv'

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
    "Training Start Date": "Start Date",
    "Trainee Name": "MIT Name",
    "Ops Account- Location": "Training Site",
    # Performance metrics columns
    "Mock QBR Score": "Mock QBR Score",  # Mock QBR Score column (Row AA)
    "Perf Evaluation Score": "Perf Evaluation Score",  # Performance Evaluation column (Row AF)
    "Skill Ranking": "Skill Ranking",  # Skill Ranking column (Row AK)
    # Job matching algorithm columns
    "AJ": "Confidence",  # Column AJ for confidence scoring
    "AV": "Vertical",    # Column AV for vertical alignment
    "AT": "Training Location",  # Column AT for training location/background
    # Resume column
    "Resume": "Resume",  # Column AZ - Resume URLs
    # Graduation week column - exact name from Google Sheets
    "Initial Expected Graduation Week": "Expected Graduation Week"  # Column AO - Exact name from your Google Sheets
}

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
# CACHE CONFIGURATION
# =============================================================================

# Cache duration in minutes (how long to keep data in memory)
CACHE_DURATION_MINUTES = -1  # Disable cache completely for immediate updates
FORCE_FRESH_DATA = True  # Always fetch fresh data

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
SERVER_PORT = 5000
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
