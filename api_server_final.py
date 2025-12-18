"""
MIT Dashboard API Server - Final Optimized Version
=================================================

A Flask-based API server that fetches data from Google Sheets and serves it to the MIT Dashboard.
This is the final, production-ready version with modular design and comprehensive error handling.

Key Features:
- Modular design with separate configuration and utility modules
- Efficient data caching and processing
- Comprehensive error handling and logging
- Date-based business lesson completion tracking
- Real-time Google Sheets integration
- Production-ready with proper documentation

Author: AI Assistant
Date: October 2025
"""

from flask import Flask, jsonify, request, send_file, send_from_directory, render_template
from flask_cors import CORS
from datetime import datetime, timedelta
import logging
import pandas as pd
import os
from io import BytesIO
from typing import Dict, List, Any, Optional

# Import our custom modules
from config import *
from utils import (
    fetch_google_sheets_data, 
    fetch_open_positions_data,
    process_candidate_data, 
    get_top_matches,
    get_candidate_top_matches,
    categorize_candidates_by_week,
    log_debug, 
    log_error,
    merge_candidate_sources,
    normalize_name,
    parse_salary,
    build_mentor_profiles,
    get_mentor_dashboard_metrics,
    get_active_training_mentors,
    get_mit_alumni
)
from executive_report import collect_report_data

# =============================================================================
# FLASK APP SETUP
# =============================================================================

app = Flask(__name__)
CORS(app)

# Configure JSON encoding for proper UTF-8 support
app.config['JSON_AS_ASCII'] = False

# Set up logging
# Use INFO level for performance (skips debug logs). Set to DEBUG only when needed for troubleshooting.
# INFO level: Shows important operations but skips verbose debug logs
logging.basicConfig(
    level=logging.INFO if DEBUG_MODE else logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# DATA CACHING
# =============================================================================

class DataCache:
    """
    Simple in-memory cache for Google Sheets data
    
    This cache stores the processed Google Sheets data in memory to avoid
    repeated API calls and improve performance.
    """
    
    def __init__(self, cache_duration_minutes: int = CACHE_DURATION_MINUTES):
        """
        Initialize the cache
        
        Args:
            cache_duration_minutes: How long to keep data cached (in minutes)
        """
        self.cache_duration = timedelta(minutes=cache_duration_minutes)
        self.cached_data = None
        self.cache_timestamp = None
    
    def get_data(self) -> Optional[pd.DataFrame]:
        """
        Get cached data if still valid
        
        Returns:
            pd.DataFrame if cache is valid, None otherwise
        """
        if (self.cached_data is not None and 
            self.cache_timestamp is not None and 
            datetime.now() - self.cache_timestamp < self.cache_duration):
            log_debug("Using cached data")
            return self.cached_data
        return None
    
    def set_data(self, data: pd.DataFrame) -> None:
        """
        Cache the data with current timestamp
        
        Args:
            data: DataFrame to cache
        """
        self.cached_data = data
        self.cache_timestamp = datetime.now()
        log_debug("Data cached successfully")

# Global cache instance
data_cache = DataCache()

# Cache for merged candidate sources (most expensive operation - 3 HTTP requests + processing)
merged_candidates_cache = {
    'data': None,
    'timestamp': None
}

# Cache for executive report data
executive_report_cache = {
    'data': None,
    'timestamp': None
}

# =============================================================================
# DATA FETCHING WITH CACHING
# =============================================================================

def get_cached_data() -> pd.DataFrame:
    """
    Get data from cache or fetch fresh data if cache is expired.
    Respects FORCE_FRESH_DATA flag to bypass cache.
    
    Returns:
        pd.DataFrame: Processed data from Google Sheets
    """
    # Force fresh data if flag is set
    if FORCE_FRESH_DATA:
        log_debug("FORCE_FRESH_DATA enabled - bypassing cache")
        fresh_data = fetch_google_sheets_data()
        data_cache.set_data(fresh_data)  # Still cache it for future use
        return fresh_data
    
    # Check cache first
    cached_data = data_cache.get_data()
    if cached_data is not None:
        return cached_data
    
    # Fetch fresh data
    log_debug("Cache expired, fetching fresh data")
    fresh_data = fetch_google_sheets_data()
    data_cache.set_data(fresh_data)
    return fresh_data

def get_cached_merged_candidates() -> List[Dict[str, Any]]:
    """
    Get merged candidates from cache or compute fresh if expired.
    This is the most expensive operation (3 HTTP requests + processing all candidates).
    Respects FORCE_FRESH_DATA flag to bypass cache.
    
    Returns:
        List of merged candidate dictionaries
    """
    # Force fresh data if flag is set
    if FORCE_FRESH_DATA:
        log_debug("FORCE_FRESH_DATA enabled - bypassing merged candidates cache")
        fresh_data = merge_candidate_sources()
        merged_candidates_cache['data'] = fresh_data
        merged_candidates_cache['timestamp'] = datetime.now()
        return fresh_data
    
    cache_duration = timedelta(minutes=CACHE_DURATION_MINUTES)
    
    # Check if cache is valid
    if (merged_candidates_cache['data'] is not None and 
        merged_candidates_cache['timestamp'] is not None and
        datetime.now() - merged_candidates_cache['timestamp'] < cache_duration):
        log_debug("Using cached merged candidates")
        return merged_candidates_cache['data']
    
    # Compute fresh data
    log_debug("Cache expired, computing fresh merged candidates")
    fresh_data = merge_candidate_sources()
    merged_candidates_cache['data'] = fresh_data
    merged_candidates_cache['timestamp'] = datetime.now()
    return fresh_data

# =============================================================================
# API ROUTES
# =============================================================================

@app.route('/')
def serve_dashboard():
    """
    Serve the main dashboard HTML file
    
    This endpoint serves the index.html file for the MIT Dashboard.
    This is essential for Heroku deployment where the frontend needs to be served by Flask.
    
    Returns:
        HTML file: The main dashboard interface
    """
    try:
        return send_file('index.html')
    except Exception as e:
        log_error("Error serving dashboard", e)
        return jsonify({'error': 'Dashboard not available'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    
    Returns:
        JSON response with server status and version information
    """
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': API_VERSION,
        'debug_mode': DEBUG_MODE
    })

@app.route('/api/dashboard-data', methods=['GET'])
def get_dashboard_data():
    """
    Get dashboard metrics and statistics
    
    This endpoint provides high-level statistics for the dashboard overview,
    including total candidates, status breakdowns, average salary, and open positions.
    
    Returns:
        JSON response with dashboard metrics
    """
    try:
        # Use unified candidate list from three-tier integration (cached)
        candidates = get_cached_merged_candidates()
        
        # Calculate week-based categories first
        categorized = categorize_candidates_by_week(candidates)
        
        # Calculate dashboard metrics - use same logic as endpoint to ensure consistency
        # Count offer_pending using the same filtering logic as the endpoint
        offer_list = []
        for c in candidates:
            status = str(c.get('status', '')).lower()
            week = c.get('week', 0)
            # Convert week to int
            try:
                if week is None or (isinstance(week, str) and week.lower() in ['n/a', 'na', '']):
                    week = 0
                else:
                    week = int(week)
            except (ValueError, TypeError):
                week = 0
            is_pending_status = 'pending' in status or 'offer' in status
            has_not_started = week == 0
            
            if is_pending_status and has_not_started:
                offer_list.append(c)
        offer_pending = len(offer_list)
        # Active MITs = sum of all active week bands (same logic as executive report)
        total_candidates = (len(categorized['weeks_0_3']) + 
                           len(categorized['weeks_4_6']) + 
                           len(categorized['week_7_only']) + 
                           len(categorized['weeks_8_plus']))
        in_training = len([c for c in candidates if str(c.get('status','')).lower() == 'training'])
        ready_for_placement = len([c for c in candidates if str(c.get('status','')).lower() == 'ready'])
        
        # Calculate average salary
        salaries = [float(c.get('salary', 0) or 0) for c in candidates]
        valid_salaries = [s for s in salaries if s and s > 0]
        avg_salary = sum(valid_salaries) / len(valid_salaries) if valid_salaries else 0
        
        # Get open positions count
        open_positions = fetch_open_positions_data()
        open_positions_count = len(open_positions)
        
        # Get mentor metrics
        mentor_metrics = get_mentor_dashboard_metrics()
        
        dashboard_data = {
            'total_candidates': total_candidates,
            'in_training': in_training,
            'ready_for_placement': ready_for_placement,
            'offer_pending': offer_pending,
            'weeks_0_3': len(categorized['weeks_0_3']),  # Operational Overview
            'weeks_4_6': len(categorized['weeks_4_6']),  # Active Training
            'week_7_only': len(categorized['week_7_only']),  # Week 7 Priority
            'weeks_8_plus': len(categorized['weeks_8_plus']),  # Ready for Placement
            'open_positions': open_positions_count,
            'average_salary': round(avg_salary, 2),
            'mentor_metrics': mentor_metrics
        }
        
        response = jsonify(dashboard_data)
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response
        
    except Exception as e:
        log_error("Error in dashboard data endpoint", e)
        return jsonify({'error': ERROR_MESSAGES['server_error']}), 500

@app.route('/api/candidates', methods=['GET'])
def get_candidates():
    """
    Get candidates ready for placement
    
    Returns:
        JSON response with list of candidates ready for placement
    """
    try:
        unified = get_cached_merged_candidates()
        ready_list = [c for c in unified if str(c.get('status','')).lower() == 'ready']
        response = jsonify(ready_list)
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response
        
    except Exception as e:
        log_error("Error in candidates endpoint", e)
        return jsonify({'error': ERROR_MESSAGES['server_error']}), 500

@app.route('/api/all-candidates', methods=['GET'])
def get_all_candidates():
    """
    Get all ACTIVE candidates (excludes offer pending/incoming MITs who haven't started)
    
    Returns:
        JSON response with active candidates only
    """
    try:
        all_candidates = get_cached_merged_candidates()
        # Use same logic as categorize_candidates_by_week: exclude only week 0 candidates with pending/offer status
        active_only = []
        for c in all_candidates:
            status = str(c.get('status', '')).lower()
            week = c.get('week', 0)
            # Only exclude if they haven't started (week 0 or N/A) AND have pending/offer status
            is_pending_status = 'pending' in status or 'offer' in status
            has_not_started = week == 0 or week is None or (isinstance(week, str) and week.lower() in ['n/a', 'na', ''])
            
            if not (is_pending_status and has_not_started):
                active_only.append(c)
        
        response = jsonify(active_only)
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response
        
    except Exception as e:
        log_error("Error in all candidates endpoint", e)
        return jsonify({'error': ERROR_MESSAGES['server_error']}), 500

@app.route('/api/in-training-candidates', methods=['GET'])
def get_in_training_candidates():
    """
    Get candidates currently in training
    
    Returns:
        JSON response with in-training candidates
    """
    try:
        unified = get_cached_merged_candidates()
        training_list = [c for c in unified if str(c.get('status','')).lower() == 'training']
        response = jsonify(training_list)
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response
        
    except Exception as e:
        log_error("Error in in-training candidates endpoint", e)
        return jsonify({'error': ERROR_MESSAGES['server_error']}), 500

@app.route('/api/offer-pending-candidates', methods=['GET'])
def get_offer_pending_candidates():
    """
    Get candidates with pending start dates (incoming MITs who haven't started yet)
    
    Returns:
        JSON response with pending start candidates
    """
    try:
        unified = get_cached_merged_candidates()
        # Use same logic as categorize_candidates_by_week: only include week 0 candidates with pending/offer status
        offer_list = []
        for c in unified:
            status = str(c.get('status', '')).lower()
            week = c.get('week', 0)
            # Convert week to int
            try:
                if week is None or (isinstance(week, str) and week.lower() in ['n/a', 'na', '']):
                    week = 0
                else:
                    week = int(week)
            except (ValueError, TypeError):
                week = 0
            # Only include if they haven't started (week 0) AND have pending/offer status
            is_pending_status = 'pending' in status or 'offer' in status
            has_not_started = week == 0
            
            if is_pending_status and has_not_started:
                offer_list.append(c)
        
        response = jsonify(offer_list)
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response
        
    except Exception as e:
        log_error("Error in offer-pending candidates endpoint", e)
        return jsonify({'error': ERROR_MESSAGES['server_error']}), 500

@app.route('/api/candidate/<name>', methods=['GET'])
def get_candidate_profile(name: str):
    """
    Get detailed profile for a specific candidate or alumni
    
    Args:
        name: Candidate name (URL encoded)
        
    Returns:
        JSON response with detailed candidate profile
    """
    try:
        # Step 1: Check current MIT candidates
        unified = get_cached_merged_candidates()
        target_norm = normalize_name(name)
        candidate_data = None
        
        for c in unified:
            if normalize_name(c.get('name','')) == target_norm:
                candidate_data = c
                break
        
        # Step 2: If not found in current candidates, check MIT alumni
        if not candidate_data:
            log_debug(f"Candidate '{name}' not found in current roster, checking alumni...")
            alumni_data = get_mit_alumni()
            
            for alum in alumni_data.get('alumni', []):
                if normalize_name(alum.get('name','')) == target_norm:
                    candidate_data = alum
                    # Mark as alumni and add required fields for profile rendering
                    candidate_data['is_alumni'] = True
                    candidate_data['status'] = 'Alumni - Placed'
                    candidate_data['week'] = int(alum.get('weeks_in_program', 0) or 0)
                    # Use placement location instead of training location for alumni
                    candidate_data['location'] = alum.get('placement_site', 'TBD')
                    # Include salary from Placed MITs sheet
                    candidate_data['salary'] = alum.get('training_salary', 'TBD')
                    # Organize scores into expected structure
                    candidate_data['scores'] = {
                        'mock_qbr_score': alum.get('mock_qbr_score', 0),
                        'assessment_score': alum.get('assessment_score', 0),
                        'perf_eval_score': alum.get('perf_eval_score', 0),
                        'confidence_score': alum.get('confidence_score', 0),
                        'skill_ranking': alum.get('skill_ranking', 'TBD')
                    }
                    # Add operation details
                    candidate_data['operation_details'] = {
                        'vertical': alum.get('training_vertical', 'TBD')
                    }
                    log_debug(f"Found '{name}' in alumni list")
                    break
        
        # Step 3: Return 404 if still not found
        if not candidate_data:
            return jsonify({'error': ERROR_MESSAGES['candidate_not_found']}), 404
        
        response = jsonify(candidate_data)
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response
        
    except Exception as e:
        log_error("Error in candidate profile endpoint", e)
        return jsonify({'error': ERROR_MESSAGES['server_error']}), 500

@app.route('/api/open-positions', methods=['GET'])
def get_open_positions():
    """
    Get open job positions from Google Sheets with optional filtering and grouping
    
    Query Parameters:
        vertical: Filter by vertical (case-insensitive)
        salary_min: Minimum salary (integer)
        salary_max: Maximum salary (integer)
        group_by_region: If 'true', return positions grouped by region
    
    Returns:
        JSON response with open positions data (grouped or flat list)
    """
    try:
        from utils import filter_positions, group_positions_by_region
        
        open_positions = fetch_open_positions_data()
        
        # Get filter parameters
        vertical = request.args.get('vertical', None)
        salary_min = request.args.get('salary_min', None, type=int)
        salary_max = request.args.get('salary_max', None, type=int)
        group_by_region = request.args.get('group_by_region', 'false').lower() == 'true'
        
        # Apply filters
        if vertical or salary_min is not None or salary_max is not None:
            open_positions = filter_positions(open_positions, vertical, salary_min, salary_max)
        
        # Group by region if requested
        if group_by_region:
            grouped = group_positions_by_region(open_positions)
            response_data = {
                'grouped': True,
                'regions': grouped,
                'total': len(open_positions)
            }
        else:
            response_data = open_positions
        
        response = jsonify(response_data)
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response
        
    except Exception as e:
        log_error("Error in open positions endpoint", e)
        return jsonify({'error': ERROR_MESSAGES['server_error']}), 500

@app.route('/api/job-matches/<int:job_id>', methods=['GET'])
def get_job_matches(job_id):
    """
    Get top matching candidates for a specific job position
    
    This endpoint calculates match scores for all candidates against the specified job
    and returns the top 3 matches based on the scoring algorithm.
    
    Args:
        job_id: Job position ID to match against
        
    Returns:
        JSON response with top matching candidates and their scores
    """
    try:
        # Get top matches for the job
        matches = get_top_matches(job_id, limit=6)
        
        if not matches:
            return jsonify({'error': f'No matches found for job ID {job_id}'}), 404
        
        # Format response data with job details
        job = matches[0]['job'] if matches else {}
        response_data = {
            'job_id': job_id,
            'job_title': job.get('title', 'Unknown'),
            'job_account': job.get('account', 'Unknown'),
            'job_city': job.get('city', 'Unknown'),
            'job_state': job.get('state', 'Unknown'),
            'job_salary': parse_salary(job.get('salary', 0)),
            'job_vertical': job.get('vertical', 'Unknown'),
            'matches': []
        }
        
        for match in matches:
            match_data = {
                'candidate_name': match['candidate']['name'],
                'match_score': match['match_score'],
                'match_quality': match['match_quality'],
                'score_breakdown': match['score_breakdown'],
                'candidate_data': match['candidate']
            }
            response_data['matches'].append(match_data)
        
        response = jsonify(response_data)
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response
        
    except Exception as e:
        log_error(f"Error in job matches endpoint for job {job_id}", e)
        return jsonify({'error': ERROR_MESSAGES['server_error']}), 500

@app.route('/api/candidate-matches/<candidate_name>', methods=['GET'])
def get_candidate_matches(candidate_name):
    """
    Get top matching jobs for a specific candidate (reverse match)
    
    This endpoint calculates match scores for all open positions against the specified candidate
    and returns the top 3 matches based on the scoring algorithm.
    
    Args:
        candidate_name: Candidate name to match against (URL encoded)
        
    Returns:
        JSON response with top matching jobs and their scores
    """
    try:
        # Get top matches for the candidate
        matches = get_candidate_top_matches(candidate_name, limit=3)
        
        if not matches:
            return jsonify({'error': f'No matches found for candidate {candidate_name}'}), 404
        
        # Format response data
        response_data = {
            'candidate_name': candidate_name,
            'matches': []
        }
        
        for match in matches:
            job = match['job']
            match_data = {
                'job_id': job.get('id', 0),
                'job_title': job.get('title', 'Unknown'),
                'job_account': job.get('account', 'Unknown'),
                'job_city': job.get('city', 'Unknown'),
                'job_state': job.get('state', 'Unknown'),
                'job_salary': job.get('salary', 0),
                'job_vertical': job.get('vertical', 'Unknown'),
                'match_score': match['match_score'],
                'match_quality': match['match_quality'],
                'score_breakdown': match['score_breakdown']
            }
            response_data['matches'].append(match_data)
        
        response = jsonify(response_data)
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response
        
    except Exception as e:
        log_error(f"Error in candidate matches endpoint for {candidate_name}", e)
        return jsonify({'error': ERROR_MESSAGES['server_error']}), 500

# =============================================================================
# STATIC FILE SERVING
# =============================================================================

@app.route('/headshots/<path:filename>')
@app.route('/headshots/<filename>')
def serve_headshot(filename):
    """Serve headshot images from the headshots directory (supports nested paths)."""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        headshots_dir = os.path.join(base_dir, 'headshots')
        return send_from_directory(headshots_dir, filename)
    except FileNotFoundError:
        return jsonify({'error': 'Image not found'}), 404

@app.route('/data/<path:filename>')
def serve_data_file(filename):
    """Serve data files from the data directory (JSON, etc)."""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, 'data')
        return send_from_directory(data_dir, filename)
    except FileNotFoundError:
        return jsonify({'error': 'Data file not found'}), 404

# =============================================================================
# DEBUG ENDPOINT
# =============================================================================

@app.route('/api/debug-columns', methods=['GET'])
def debug_columns():
    """Return list of columns detected from Google Sheets."""
    try:
        df = fetch_google_sheets_data()
        return jsonify({'columns': list(df.columns)}), 200
    except Exception as e:
        log_error("Error debugging columns", e)
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug-oig/<candidate_name>', methods=['GET'])
def debug_oig(candidate_name):
    """Debug OIG calculation for a specific candidate."""
    try:
        from utils import normalize_name
        import pandas as pd
        
        # Get all candidates
        candidates = get_cached_merged_candidates()
        
        # Find the candidate
        target_norm = normalize_name(candidate_name)
        target_candidate = None
        
        for c in candidates:
            if normalize_name(c.get('name', '')) == target_norm:
                target_candidate = c
                break
        
        if not target_candidate:
            return jsonify({'error': f'Candidate {candidate_name} not found'}), 404
        
        # Get raw data from Google Sheets
        df = fetch_google_sheets_data()
        target_row = None
        
        for idx, row in df.iterrows():
            if normalize_name(str(row.get('MIT Name', ''))) == target_norm:
                target_row = row
                break
        
        debug_info = {
            'candidate_name': target_candidate.get('name'),
            'oig_completion_object': target_candidate.get('oig_completion'),
            'company_start_date_raw': str(target_candidate.get('operation_details', {}).get('company_start_date')),
            'week': target_candidate.get('week'),
            'status': target_candidate.get('status'),
        }
        
        if target_row is not None:
            debug_info['raw_sheet_data'] = {
                'company_start_date': str(target_row.get('Company Start Date')),
                'company_start_date_type': str(type(target_row.get('Company Start Date'))),
                'company_start_date_original': str(target_row.get('Company Start Date Original')),
                'oig_completion_column': str(target_row.get('OIG Completion', 'NOT FOUND')),
                'mit_name': str(target_row.get('MIT Name')),
            }
        
        return jsonify(debug_info), 200
        
    except Exception as e:
        log_error(f"Error debugging OIG for {candidate_name}", e)
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/debug-mentor-relationships', methods=['GET'])
def debug_mentor_relationships():
    """Debug endpoint to see mentor relationships data and columns."""
    try:
        import pandas as pd
        from config import MENTOR_RELATIONSHIPS_URL
        
        # Fetch raw data
        df = pd.read_csv(MENTOR_RELATIONSHIPS_URL, dtype=str)
        
        # Clean columns
        df.columns = (
            df.columns.astype(str)
              .str.replace('\u00A0', ' ', regex=False)
              .str.replace(r'\s+', ' ', regex=True)
              .str.strip()
        )
        
        # Get relationships
        relationships = fetch_mentor_relationships_data()
        
        return jsonify({
            'raw_columns': list(df.columns),
            'relationships_count': len(relationships),
            'sample_relationships': relationships[:5] if relationships else [],
            'sample_raw_data': df.head(3).to_dict('records') if not df.empty else []
        }), 200
    except Exception as e:
        log_error("Error debugging mentor relationships", e)
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug-mentor-profiles', methods=['GET'])
def debug_mentor_profiles():
    """Debug endpoint to see mentor profiles build process."""
    try:
        from utils import fetch_mentor_relationships_data, fetch_mentor_mei_summary, build_mentor_profiles
        
        relationships = fetch_mentor_relationships_data()
        mei_summary = fetch_mentor_mei_summary()
        profiles = build_mentor_profiles()
        
        return jsonify({
            'relationships_count': len(relationships),
            'mei_summary_count': len(mei_summary),
            'profiles_count': len(profiles),
            'sample_relationships': relationships[:3] if relationships else [],
            'sample_mei_summary': dict(list(mei_summary.items())[:3]) if mei_summary else {},
            'sample_profiles': profiles[:3] if profiles else [],
            'all_mentor_names': [p['name'] for p in profiles]
        }), 200
    except Exception as e:
        log_error("Error debugging mentor profiles", e)
        return jsonify({'error': str(e)}), 500

# =============================================================================
# WEEK-BASED FILTERING ENDPOINTS
# =============================================================================

@app.route('/api/candidates/weeks-0-3', methods=['GET'])
def get_weeks_0_3():
    """Candidates in weeks 0-3 (Operational Overview)"""
    try:
        candidates = get_cached_merged_candidates()
        categorized = categorize_candidates_by_week(candidates)
        return jsonify(categorized['weeks_0_3']), 200
    except Exception as e:
        log_error("Error fetching weeks 0-3 candidates", e)
        return jsonify({'error': str(e)}), 500

@app.route('/api/candidates/weeks-4-6', methods=['GET'])
def get_weeks_4_6():
    """Candidates in weeks 4-6 (Active Training)"""
    try:
        candidates = get_cached_merged_candidates()
        categorized = categorize_candidates_by_week(candidates)
        return jsonify(categorized['weeks_4_6']), 200
    except Exception as e:
        log_error("Error fetching weeks 4-6 candidates", e)
        return jsonify({'error': str(e)}), 500

@app.route('/api/candidates/week-7-priority', methods=['GET'])
def get_week_7_priority():
    """ONLY Week 7 candidates (Placement Priority)"""
    try:
        candidates = get_cached_merged_candidates()
        categorized = categorize_candidates_by_week(candidates)
        return jsonify(categorized['week_7_only']), 200
    except Exception as e:
        log_error("Error fetching week 7 priority candidates", e)
        return jsonify({'error': str(e)}), 500

@app.route('/api/candidates/weeks-8-plus', methods=['GET'])
def get_weeks_8_plus():
    """Candidates week 8+ (Ready for Placement)"""
    try:
        candidates = get_cached_merged_candidates()
        categorized = categorize_candidates_by_week(candidates)
        return jsonify(categorized['weeks_8_plus']), 200
    except Exception as e:
        log_error("Error fetching weeks 8+ candidates", e)
        return jsonify({'error': str(e)}), 500

# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': ERROR_MESSAGES['server_error']}), 500

# =============================================================================
# MENTOR ENDPOINTS
# =============================================================================

@app.route('/api/mentors', methods=['GET'])
def get_all_mentors():
    """Get list of all mentors with their profiles"""
    profiles = build_mentor_profiles()
    return jsonify(profiles), 200

@app.route('/api/mentor/<mentor_name>', methods=['GET'])
def get_mentor_profile(mentor_name):
    """Get detailed profile for a specific mentor"""
    profiles = build_mentor_profiles()
    
    log_debug(f"Looking for mentor: '{mentor_name}' in {len(profiles)} profiles")
    if profiles:
        log_debug(f"Sample mentor names: {[p['name'] for p in profiles[:5]]}")
    
    # Find mentor by name using normalized matching (handles trailing spaces, case, etc.)
    target_norm = normalize_name(mentor_name)
    log_debug(f"Normalized search name: '{target_norm}'")
    
    mentor = None
    for m in profiles:
        mentor_norm = normalize_name(m['name'])
        log_debug(f"Comparing: '{target_norm}' vs '{mentor_norm}' (from '{m['name']}')")
        if mentor_norm == target_norm:
            mentor = m
            break
    
    if not mentor:
        log_debug(f"Mentor '{mentor_name}' not found. Available mentors: {[p['name'] for p in profiles]}")
        return jsonify({'error': 'Mentor not found'}), 404
    
    return jsonify(mentor), 200

@app.route('/api/mentor-metrics', methods=['GET'])
def get_mentor_metrics():
    """Get mentor metrics for dashboard block"""
    metrics = get_mentor_dashboard_metrics()
    return jsonify(metrics), 200

@app.route('/api/training-mentors', methods=['GET'])
def get_training_mentors_endpoint():
    """Get mentors actively training current MIT candidates"""
    try:
        mentors = get_active_training_mentors()
        response = jsonify(mentors)
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response, 200
    except Exception as e:
        log_error("Error in training mentors endpoint", e)
        return jsonify({'error': ERROR_MESSAGES['server_error']}), 500

@app.route('/api/mit-alumni', methods=['GET'])
def get_mit_alumni_endpoint():
    """Get MIT alumni with placement information"""
    try:
        alumni_data = get_mit_alumni()
        log_debug(f"MIT Alumni endpoint: returning {alumni_data.get('total_alumni', 0)} alumni")
        response = jsonify(alumni_data)
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response, 200
    except Exception as e:
        log_error("Error in MIT alumni endpoint", e)
        return jsonify({'error': ERROR_MESSAGES['server_error']}), 500

# =============================================================================
# EXECUTIVE REPORT ROUTES
# =============================================================================

@app.route('/api/executive-print', methods=['GET'])
def api_executive_print():
    """
    API endpoint to get executive print report data as JSON (cached for performance)
    """
    try:
        cache_duration = timedelta(minutes=10)
        
        if (executive_report_cache['data'] is not None and 
            executive_report_cache['timestamp'] is not None and
            datetime.now() - executive_report_cache['timestamp'] < cache_duration):
            response = jsonify(executive_report_cache['data'])
            response.headers['Content-Type'] = 'application/json; charset=utf-8'
            return response
        
        data = collect_report_data()
        
        executive_report_cache['data'] = data
        executive_report_cache['timestamp'] = datetime.now()
        
        response = jsonify(data)
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response
        
    except Exception as e:
        log_error("Error generating executive print report data", e)
        return jsonify({'error': 'Failed to generate executive print report data'}), 500

@app.route('/executive-print')
def executive_print():
    """
    Render the clean print-optimized executive report HTML
    """
    try:
        data = collect_report_data()
        return render_template('executive_print.html', **data)
    except Exception as e:
        log_error("Error rendering executive print report", e)
        return f"Error generating report: {str(e)}", 500

@app.route('/pipeline')
def pipeline_view():
    """
    Render the Placement & Pipeline View page
    """
    try:
        from utils import merge_candidate_sources, get_mit_alumni
        from datetime import datetime
        
        # Get real active MITs
        candidates = merge_candidate_sources()
        active_mits = [c for c in candidates if c.get('week', 0) >= 1]
        
        # Fetch graduated MITs and add them to Tier 1 Managers
        alumni_data = get_mit_alumni()
        graduated_mits = alumni_data.get('alumni', [])
        
        # Convert graduated MITs to Tier 1 Managers format
        tier1_managers_from_alumni = []
        for alumni in graduated_mits:
            # Calculate months in role from placement_start_date
            months = 0
            placement_date = alumni.get('placement_start_date', '')
            if placement_date and placement_date not in ['TBD', 'nan', '']:
                try:
                    import pandas as pd
                    start_date = pd.to_datetime(str(placement_date), errors='coerce')
                    if pd.notna(start_date):
                        months = (datetime.now() - start_date.to_pydatetime()).days // 30
                except:
                    months = 0
            
            # Mock CSAT (in real implementation, fetch from 4insite)
            csat = 4.0  # Default, should be fetched from actual data
            
            tier1_managers_from_alumni.append({
                'name': alumni.get('name', 'Unknown'),
                'site': alumni.get('placement_site', 'TBD'),
                'months': max(1, months),  # At least 1 month
                'csat': csat,
                'headcount': 20,  # Default Tier 1 headcount (should be fetched from site data)
                'revenue': '$85K/mo',  # Default Tier 1 revenue (should be fetched from site data)
                'trend': 'Stable'  # Default trend
            })
        
        # Mock data for additional Tier 1 Managers (for testing)
        tier1_managers_mock = [
            {'name': 'TestUser1', 'site': 'Ford - Dearborn, MI', 'months': 8, 'csat': 4.2, 'headcount': 25, 'revenue': '$85K/mo', 'trend': 'Up'},
            {'name': 'TestUser2', 'site': 'Intel - Chandler, AZ', 'months': 6, 'csat': 3.8, 'headcount': 18, 'revenue': '$72K/mo', 'trend': 'Stable'},
        ]
        
        # Combine graduated MITs with mock data
        tier1_managers = tier1_managers_from_alumni + tier1_managers_mock
        
        # Mock data for Tier 2 Managers (can move to Tier 3)
        tier2_managers = [
            {'name': 'TestUser5', 'site': 'Apple - Austin, TX', 'months': 14, 'csat': 4.4, 'headcount': 35, 'revenue': '$320K/mo', 'trend': 'Up'},
            {'name': 'TestUser6', 'site': 'Google - Reston, VA', 'months': 12, 'csat': 4.1, 'headcount': 32, 'revenue': '$280K/mo', 'trend': 'Stable'},
            {'name': 'TestUser7', 'site': 'Meta - Austin, TX', 'months': 18, 'csat': 3.9, 'headcount': 38, 'revenue': '$450K/mo', 'trend': 'Down'},
        ]
        
        # Mock data for Tier 3 Managers (can move to Tier 4)
        tier3_managers = [
            {'name': 'TestUser8', 'site': 'Amazon - Nashville, TN', 'months': 24, 'csat': 4.6, 'headcount': 45, 'revenue': '$680K/mo', 'trend': 'Up'},
            {'name': 'TestUser9', 'site': 'Microsoft - Redmond, WA', 'months': 20, 'csat': 4.3, 'headcount': 48, 'revenue': '$720K/mo', 'trend': 'Stable'},
        ]
        
        # Mock data for Tier 4 Managers (can move to Tier 5)
        tier4_managers = [
            {'name': 'TestUser10', 'site': 'Tesla - Fremont, CA', 'months': 30, 'csat': 4.7, 'headcount': 85, 'revenue': '$950K/mo', 'trend': 'Up'},
        ]
        
        # Mock data for Tier 5 Managers (can move to Tier 6)
        tier5_managers = [
            {'name': 'TestUser11', 'site': 'Amazon - JFK8, NY', 'months': 36, 'csat': 4.8, 'headcount': 220, 'revenue': '$1.3M/mo', 'trend': 'Up'},
        ]
        
        # Mock openings for Tier 1 (1-28 HC, <$100K/mo)
        tier1_openings = [
            {'site': 'Boeing - Everett, WA', 'role': 'Site Manager', 'headcount': 22, 'revenue': '$78K/mo', 'programs': 1, 'type': 'Growth', 'matched': None, 'urgent': False, 'target': '02/01/2025'},
            {'site': 'Lockheed - Fort Worth, TX', 'role': 'Area Manager', 'headcount': 18, 'revenue': '$65K/mo', 'programs': 1, 'type': 'Backfill', 'matched': 'Evan Tichenor', 'urgent': False, 'target': None},
            {'site': 'SpaceX - Hawthorne, CA', 'role': 'Site Manager', 'headcount': 25, 'revenue': '$92K/mo', 'programs': 1, 'type': 'Growth', 'matched': None, 'urgent': True, 'target': '01/15/2025'},
        ]
        
        # Mock openings for Tier 2 (29-39 HC, $100K-$500K/mo)
        tier2_openings = [
            {'site': 'Tesla - Austin, TX', 'role': 'Site Manager', 'headcount': 35, 'revenue': '$380K/mo', 'programs': 2, 'type': 'Growth', 'matched': None, 'urgent': False, 'target': '02/15/2025'},
            {'site': 'Apple - Cupertino, CA', 'role': 'Operations Manager', 'headcount': 32, 'revenue': '$420K/mo', 'programs': 2, 'type': 'Backfill', 'matched': None, 'urgent': True, 'target': '01/20/2025'},
        ]
        
        # Mock openings for Tier 3 (40-50 HC, $500K-$750K/mo)
        tier3_openings = [
            {'site': 'Amazon - Seattle, WA', 'role': 'Senior Site Manager', 'headcount': 45, 'revenue': '$680K/mo', 'programs': 3, 'type': 'Growth', 'matched': None, 'urgent': False, 'target': '03/01/2025'},
            {'site': 'Google - Mountain View, CA', 'role': 'Site Director', 'headcount': 48, 'revenue': '$720K/mo', 'programs': 3, 'type': 'Backfill', 'matched': 'TestUser8', 'urgent': False, 'target': None},
            {'site': 'Microsoft - Bellevue, WA', 'role': 'Operations Director', 'headcount': 42, 'revenue': '$650K/mo', 'programs': 3, 'type': 'Growth', 'matched': None, 'urgent': True, 'target': '01/25/2025'},
        ]
        
        # Mock openings for Tier 4 (51-100 HC, $750K-$1M/mo)
        tier4_openings = [
            {'site': 'Amazon - Nashville, TN', 'role': 'Regional Manager', 'headcount': 85, 'revenue': '$950K/mo', 'programs': 5, 'type': 'Growth', 'matched': None, 'urgent': False, 'target': '04/01/2025'},
            {'site': 'Tesla - Fremont, CA', 'role': 'Site Manager', 'headcount': 72, 'revenue': '$880K/mo', 'programs': 6, 'type': 'Backfill', 'matched': None, 'urgent': True, 'target': '01/30/2025'},
        ]
        
        # Mock openings for Tier 5 (101-250 HC, $1M-$1.5M/mo)
        tier5_openings = [
            {'site': 'Amazon - JFK8, NY', 'role': 'Regional Manager', 'headcount': 220, 'revenue': '$1.3M/mo', 'programs': 9, 'type': 'Growth', 'matched': None, 'urgent': False, 'target': '05/01/2025'},
        ]
        
        # Mock openings for Tier 6 (250+ HC, >$1.5M/mo)
        tier6_openings = [
            {'site': 'Amazon - DFW7, TX', 'role': 'Senior Regional Manager', 'headcount': 380, 'revenue': '$2.1M/mo', 'programs': 12, 'type': 'Growth', 'matched': None, 'urgent': False, 'target': '06/01/2025'},
        ]
        
        total_openings = len(tier1_openings) + len(tier2_openings) + len(tier3_openings) + len(tier4_openings) + len(tier5_openings) + len(tier6_openings)
        
        return render_template('pipeline_view.html',
            active_mits=active_mits,
            tier1_managers=tier1_managers,
            tier2_managers=tier2_managers,
            tier3_managers=tier3_managers,
            tier4_managers=tier4_managers,
            tier5_managers=tier5_managers,
            tier1_openings=tier1_openings,
            tier2_openings=tier2_openings,
            tier3_openings=tier3_openings,
            tier4_openings=tier4_openings,
            tier5_openings=tier5_openings,
            tier6_openings=tier6_openings,
            total_openings=total_openings
        )
    except Exception as e:
        log_error("Error rendering pipeline view", e)
        return f"Error generating pipeline view: {str(e)}", 500

@app.route('/executive-print/pdf')
def executive_print_pdf():
    """
    Generate PDF version of executive report using WeasyPrint
    """
    try:
        from weasyprint import HTML
        import os
        
        data = collect_report_data()
        html_string = render_template('executive_print.html', **data)
        
        css_path = os.path.join(os.path.dirname(__file__), 'static', 'print.css')
        
        pdf = HTML(string=html_string).write_pdf(stylesheets=[css_path])
        
        return send_file(
            BytesIO(pdf),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'MIT_Executive_Report_{datetime.now().strftime("%Y%m%d")}.pdf'
        )
        
    except ImportError:
        return "WeasyPrint not installed. Install with: pip install weasyprint", 500
    except Exception as e:
        log_error("Error generating PDF", e)
        return f"Error generating PDF: {str(e)}", 500

# =============================================================================
# INDIVIDUAL CANDIDATE REPORT ROUTES
# =============================================================================

def get_micah_survey_data():
    """
    Hardcoded survey data for Micah Scherrei
    In the future, this could be fetched from a Google Sheet or database
    """
    return [
        {
            'date': '11/18/2025',
            'section1': [
                {'name': 'Completes tasks accurately and on schedule', 'score': 5},
                {'name': 'Understands daily site operations', 'score': 5},
                {'name': 'Navigates 4insite and completes audits', 'score': 4},
                {'name': 'Follows safety and site standards', 'score': 5},
                {'name': 'Attention to detail and quality', 'score': 5},
            ],
            'section1_avg': 4.8,
            'section2': [
                {'name': 'Sense of urgency and initiative', 'score': 5},
                {'name': 'Adapts to schedule/task changes', 'score': 5},
                {'name': 'Communicates with peers and leadership', 'score': 5},
                {'name': 'Coachable and applies feedback', 'score': 5},
                {'name': 'Professional appearance and PPE', 'score': 5},
            ],
            'section2_avg': 5.0,
            'section3': [
                {'name': 'Actively engaged in learning', 'score': 5},
                {'name': 'Learns quickly and retains info', 'score': 5},
                {'name': 'Applies new skills confidently', 'score': 5},
                {'name': 'Builds positive relationships', 'score': 5},
                {'name': 'Willingness to work multiple shifts', 'score': 4},
            ],
            'section3_avg': 4.8,
            'section4': {
                'confidence': 5,
                'responsibility': 5,
                'status': 'Exceeding Expectations'
            },
            'overall_avg': 4.9,
            'observations': 'Great engagement and willing to learn! He is ambitious and very professional to all! Fully engaged with all site activities.'
        },
        {
            'date': '11/25/2025',
            'section1': [
                {'name': 'Completes tasks accurately and on schedule', 'score': 5},
                {'name': 'Understands daily site operations', 'score': 5},
                {'name': 'Navigates 4insite and completes audits', 'score': 3},
                {'name': 'Follows safety and site standards', 'score': 5},
                {'name': 'Attention to detail and quality', 'score': 4},
            ],
            'section1_avg': 4.4,
            'section2': [
                {'name': 'Sense of urgency and initiative', 'score': 4},
                {'name': 'Adapts to schedule/task changes', 'score': 4},
                {'name': 'Communicates with peers and leadership', 'score': 5},
                {'name': 'Coachable and applies feedback', 'score': 4},
                {'name': 'Professional appearance and PPE', 'score': 5},
            ],
            'section2_avg': 4.4,
            'section3': [
                {'name': 'Actively engaged in learning', 'score': 5},
                {'name': 'Learns quickly and retains info', 'score': 5},
                {'name': 'Applies new skills confidently', 'score': 3},
                {'name': 'Builds positive relationships', 'score': 5},
                {'name': 'Willingness to work multiple shifts', 'score': 3},
            ],
            'section3_avg': 4.2,
            'section4': {
                'confidence': 4,
                'responsibility': 4,
                'status': 'Progressing as Expected'
            },
            'overall_avg': 4.25,
            'observations': 'Created fliers to help attract people to apply for open custodian positions.'
        }
    ]

def calculate_cohort_averages(candidates):
    """Calculate cohort averages for comparison"""
    if not candidates:
        return {
            'total_candidates': 0,
            'avg_week': 0,
            'avg_onboarding': 0,
            'avg_lessons': 0,
            'avg_assessment': 3.5  # Default average
        }
    
    # Filter out pending candidates
    active = [c for c in candidates if 'pending' not in str(c.get('status', '')).lower()]
    
    if not active:
        return {
            'total_candidates': 0,
            'avg_week': 0,
            'avg_onboarding': 0,
            'avg_lessons': 0,
            'avg_assessment': 3.5
        }
    
    total = len(active)
    avg_week = sum(c.get('week', 0) for c in active) / total
    avg_onboarding = sum(c.get('onboarding_progress', {}).get('percentage', 0) or 0 for c in active) / total
    avg_lessons = sum(c.get('business_lessons_progress', {}).get('percentage', 0) or 0 for c in active) / total
    
    return {
        'total_candidates': total,
        'avg_week': avg_week,
        'avg_onboarding': avg_onboarding,
        'avg_lessons': avg_lessons,
        'avg_assessment': 3.8  # Estimated cohort average for mentor assessments
    }

def generate_candidate_insights(candidate, surveys, cohort):
    """Generate key insights for the candidate report"""
    insights = []
    
    # Overall performance
    latest_avg = surveys[-1]['overall_avg']
    if latest_avg >= 4.5:
        insights.append(f"<strong>{candidate.get('name')}</strong> is performing at an <strong>exceptional level</strong> with an overall assessment score of {latest_avg}/5.")
    elif latest_avg >= 4.0:
        insights.append(f"<strong>{candidate.get('name')}</strong> is performing <strong>above expectations</strong> with an overall assessment score of {latest_avg}/5.")
    elif latest_avg >= 3.5:
        insights.append(f"<strong>{candidate.get('name')}</strong> is <strong>progressing well</strong> with an overall assessment score of {latest_avg}/5.")
    else:
        insights.append(f"<strong>{candidate.get('name')}</strong> has opportunities for improvement with an overall assessment score of {latest_avg}/5.")
    
    # Week-over-week trend
    if len(surveys) >= 2:
        first_avg = surveys[0]['overall_avg']
        change = latest_avg - first_avg
        if change > 0.2:
            insights.append(f"Showing <strong>positive improvement</strong> from Survey 1 ({first_avg}/5) to Survey 2 ({latest_avg}/5).")
        elif change < -0.2:
            insights.append(f"Scores have <strong>decreased slightly</strong> from Survey 1 ({first_avg}/5) to Survey 2 ({latest_avg}/5) — recommend mentor follow-up.")
        else:
            insights.append(f"Maintaining <strong>consistent performance</strong> across both survey periods.")
    
    # Strengths
    strengths = []
    latest = surveys[-1]
    if latest['section2_avg'] >= 4.5:
        strengths.append("Leadership & Soft Skills")
    if latest['section3_avg'] >= 4.5:
        strengths.append("Engagement & Learning Aptitude")
    if latest['section1_avg'] >= 4.5:
        strengths.append("Core Competencies")
    
    if strengths:
        insights.append(f"Key strengths identified: <strong>{', '.join(strengths)}</strong>.")
    
    # Areas for development
    areas = []
    if latest['section1_avg'] < 4.0:
        areas.append("Core Competencies")
    if latest['section2_avg'] < 4.0:
        areas.append("Leadership Skills")
    if latest['section3_avg'] < 4.0:
        areas.append("Engagement & Aptitude")
    
    if areas:
        insights.append(f"Areas for continued development: <strong>{', '.join(areas)}</strong>.")
    
    # Status
    status = latest['section4']['status']
    if status == 'Exceeding Expectations':
        insights.append(f"Mentor assessment: <strong>Exceeding Expectations</strong> — candidate is ready for increased responsibility.")
    elif status == 'Progressing as Expected':
        insights.append(f"Mentor assessment: <strong>Progressing as Expected</strong> — on track for graduation timeline.")
    else:
        insights.append(f"Mentor assessment: <strong>{status}</strong> — may require additional support or extended training.")
    
    return insights

@app.route('/candidate-report/<candidate_name>')
def candidate_report(candidate_name):
    """
    Render individual candidate report HTML page
    """
    try:
        # Get candidate data
        candidates = get_cached_merged_candidates()
        target_norm = normalize_name(candidate_name)
        candidate = None
        
        for c in candidates:
            if normalize_name(c.get('name', '')) == target_norm:
                candidate = c
                break
        
        if not candidate:
            return f"Candidate '{candidate_name}' not found", 404
        
        # Get survey data (currently hardcoded for Micah)
        if 'micah' in target_norm:
            surveys = get_micah_survey_data()
        else:
            # Default empty surveys for other candidates
            surveys = [
                {
                    'date': 'N/A',
                    'section1': [{'name': 'No data', 'score': 0}] * 5,
                    'section1_avg': 0,
                    'section2': [{'name': 'No data', 'score': 0}] * 5,
                    'section2_avg': 0,
                    'section3': [{'name': 'No data', 'score': 0}] * 5,
                    'section3_avg': 0,
                    'section4': {'confidence': 0, 'responsibility': 0, 'status': 'No Data'},
                    'overall_avg': 0,
                    'observations': 'No survey data available for this candidate.'
                }
            ] * 2
        
        # Calculate cohort averages
        cohort = calculate_cohort_averages(candidates)
        
        # Generate insights
        insights = generate_candidate_insights(candidate, surveys, cohort)
        
        # Render template
        return render_template('candidate_report.html',
            candidate=candidate,
            surveys=surveys,
            cohort=cohort,
            insights=insights,
            timestamp=datetime.now().strftime('%B %d, %Y at %I:%M %p'),
            dashboard_url='https://mit-training-dashboard-dd693bfc9f5a.herokuapp.com'
        )
        
    except Exception as e:
        log_error(f"Error generating candidate report for {candidate_name}", e)
        return f"Error generating report: {str(e)}", 500

# =============================================================================
# MAIN APPLICATION
# =============================================================================

if __name__ == '__main__':
    # Use Heroku's PORT environment variable if available, otherwise use configured port
    port = int(os.environ.get('PORT', SERVER_PORT))
    
    print("=" * 60)
    print("MIT Dashboard API Server - Final Optimized Version")
    print("=" * 60)
    print(f"Version: {API_VERSION}")
    print(f"Debug Mode: {'Enabled' if DEBUG_MODE else 'Disabled'}")
    print(f"Cache Duration: {CACHE_DURATION_MINUTES} minutes")
    print()
    print("API Endpoints:")
    print("   - GET / - Main dashboard interface")
    print("   - GET /api/dashboard-data - Dashboard metrics")
    print("   - GET /api/candidates - Ready for placement candidates")
    print("   - GET /api/candidate/<name> - Individual candidate profile")
    print("   - GET /api/all-candidates - All candidates")
    print("   - GET /api/in-training-candidates - In training candidates")
    print("   - GET /api/offer-pending-candidates - Offer pending candidates")
    print("   - GET /api/open-positions - Open job positions")
    print("   - GET /api/job-matches/<job_id> - Top candidate matches for a job")
    print("   - GET /api/candidates/weeks-0-3 - Weeks 0-3 candidates")
    print("   - GET /api/candidates/weeks-4-6 - Weeks 4-6 candidates")
    print("   - GET /api/candidates/week-7-priority - Week 7 priority candidates")
    print("   - GET /api/candidates/weeks-8-plus - Weeks 8+ candidates")
    print("   - GET /api/mentors - All mentors with profiles")
    print("   - GET /api/mentor/<name> - Individual mentor profile")
    print("   - GET /api/mentor-metrics - Mentor dashboard metrics")
    print("   - GET /api/training-mentors - Active training mentors")
    print("   - GET /api/mit-alumni - MIT alumni data")
    print("   - GET /api/executive-print - Executive report data (JSON)")
    print("   - GET /executive-print - Executive report (HTML)")
    print("   - GET /executive-print/pdf - Executive report (PDF)")
    print("   - GET /headshots/<filename> - Serve headshot images")
    print("   - GET /data/<filename> - Serve data files (JSON)")
    print("   - GET /api/health - Health check")
    print()
    print(f"API Server running on: http://localhost:{port}")
    print("Open your HTML file to use the dashboard!")
    print("=" * 60)
    
    # Run the Flask app
    app.run(
        host=SERVER_HOST, 
        port=port, 
        debug=SERVER_DEBUG
    )
