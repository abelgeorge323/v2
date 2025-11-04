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

from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from datetime import datetime, timedelta
import logging
import pandas as pd
import os
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
    get_active_training_mentors
)

# =============================================================================
# FLASK APP SETUP
# =============================================================================

app = Flask(__name__)
CORS(app)

# Configure JSON encoding for proper UTF-8 support
app.config['JSON_AS_ASCII'] = False

# Set up logging
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

# =============================================================================
# DATA FETCHING WITH CACHING
# =============================================================================

def get_cached_data() -> pd.DataFrame:
    """
    Get data from cache or fetch fresh data if cache is expired
    
    Returns:
        pd.DataFrame: Processed data from Google Sheets
    """
    # Check cache first
    cached_data = data_cache.get_data()
    if cached_data is not None:
        return cached_data
    
    # Fetch fresh data
    log_debug("Cache expired, fetching fresh data")
    fresh_data = fetch_google_sheets_data()
    data_cache.set_data(fresh_data)
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
        # Use unified candidate list from three-tier integration
        candidates = merge_candidate_sources()
        
        # Calculate week-based categories first
        categorized = categorize_candidates_by_week(candidates)
        
        # Calculate dashboard metrics (exclude offer pending from total)
        offer_pending_candidates = categorized['offer_pending']
        offer_pending = len(offer_pending_candidates)
        total_candidates = len(candidates) - offer_pending  # Exclude offer pending from total
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
        unified = merge_candidate_sources()
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
    Get all candidates regardless of status
    
    Returns:
        JSON response with all candidates
    """
    try:
        response = jsonify(merge_candidate_sources())
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
        unified = merge_candidate_sources()
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
    Get candidates with pending offers
    
    Returns:
        JSON response with offer-pending candidates
    """
    try:
        unified = merge_candidate_sources()
        offer_list = [c for c in unified if 'offer' in str(c.get('status','')).lower()]
        response = jsonify(offer_list)
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response
        
    except Exception as e:
        log_error("Error in offer-pending candidates endpoint", e)
        return jsonify({'error': ERROR_MESSAGES['server_error']}), 500

@app.route('/api/candidate/<name>', methods=['GET'])
def get_candidate_profile(name: str):
    """
    Get detailed profile for a specific candidate
    
    Args:
        name: Candidate name (URL encoded)
        
    Returns:
        JSON response with detailed candidate profile
    """
    try:
        unified = merge_candidate_sources()
        target_norm = normalize_name(name)
        candidate_data = None
        for c in unified:
            if normalize_name(c.get('name','')) == target_norm:
                candidate_data = c
                break
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
    Get open job positions from Google Sheets
    
    Returns:
        JSON response with open positions data
    """
    try:
        open_positions = fetch_open_positions_data()
        
        response = jsonify(open_positions)
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

# =============================================================================
# WEEK-BASED FILTERING ENDPOINTS
# =============================================================================

@app.route('/api/candidates/weeks-0-3', methods=['GET'])
def get_weeks_0_3():
    """Candidates in weeks 0-3 (Operational Overview)"""
    try:
        candidates = merge_candidate_sources()
        categorized = categorize_candidates_by_week(candidates)
        return jsonify(categorized['weeks_0_3']), 200
    except Exception as e:
        log_error("Error fetching weeks 0-3 candidates", e)
        return jsonify({'error': str(e)}), 500

@app.route('/api/candidates/weeks-4-6', methods=['GET'])
def get_weeks_4_6():
    """Candidates in weeks 4-6 (Active Training)"""
    try:
        candidates = merge_candidate_sources()
        categorized = categorize_candidates_by_week(candidates)
        return jsonify(categorized['weeks_4_6']), 200
    except Exception as e:
        log_error("Error fetching weeks 4-6 candidates", e)
        return jsonify({'error': str(e)}), 500

@app.route('/api/candidates/week-7-priority', methods=['GET'])
def get_week_7_priority():
    """ONLY Week 7 candidates (Placement Priority)"""
    try:
        candidates = merge_candidate_sources()
        categorized = categorize_candidates_by_week(candidates)
        return jsonify(categorized['week_7_only']), 200
    except Exception as e:
        log_error("Error fetching week 7 priority candidates", e)
        return jsonify({'error': str(e)}), 500

@app.route('/api/candidates/weeks-8-plus', methods=['GET'])
def get_weeks_8_plus():
    """Candidates week 8+ (Ready for Placement)"""
    try:
        candidates = merge_candidate_sources()
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
    
    # Find mentor by name using normalized matching (handles trailing spaces, case, etc.)
    target_norm = normalize_name(mentor_name)
    mentor = next((m for m in profiles if normalize_name(m['name']) == target_norm), None)
    
    if not mentor:
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
    print("   - GET /headshots/<filename> - Serve headshot images")
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
