import os
import importlib
import argparse
from flask import Flask, request, jsonify, render_template, make_response, send_from_directory
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from huggingface_hub import upload_folder
from werkzeug.exceptions import RequestEntityTooLarge
from config import Config
from src.ranking import rank_job_descriptions
from src.utils import validate_and_clean_input, dump_ranked_jobs, process_job_dataframe
import logging
import asyncio
import gc

# Flask app setup
app = Flask(__name__, template_folder=Config.TEMPLATE_FOLDER, static_folder=Config.STATIC_FOLDER)
app.config.from_object(Config)

# Set up rate limiter
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["50 per minute"]
)

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_fetcher_modules(fetcher_dir):
    """Dynamically load fetchers modules from the specified directory."""
    fetcher_modules = {}

    for filename in os.listdir(fetcher_dir):
        if filename.endswith(".py") and filename != "__init__.py":
            module_name = filename[:-3]  # Remove the .py extension
            module_path = f"src.fetchers.{module_name}"

            # Dynamically import the module
            module = importlib.import_module(module_path)

            # Ensure the module provides a `fetch_jobs` function
            if hasattr(module, 'fetch_jobs'):
                fetcher_modules[module_name] = module.fetch_jobs
            else:
                logger.warning(f"Module {module_name} does not have a 'fetch_jobs' function")

    return fetcher_modules

# Initialize job fetching function based on configuration
def load_job_fetching_function(fetcher_name, fetcher_modules):
    """Initialize the job fetching function based on the specified fetcher name."""
    if fetcher_name in fetcher_modules:
        logger.info(f"Using {fetcher_name}-based job fetching.")
        return fetcher_modules[fetcher_name]
    else:
        available_fetchers = ', '.join(fetcher_modules.keys())
        raise ValueError(f"Unknown fetcher: {fetcher_name}. Available fetchers: {available_fetchers}. Check the .env setting and your fetcher implementation in src/fetchers/")

# Load fetcher modules and initialize job fetching function
fetcher_modules = load_fetcher_modules("src/fetchers")
job_fetching_function = load_job_fetching_function(Config.FETCHER_NAME, fetcher_modules)


@app.after_request
def add_security_headers(response):
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "  # Only allow resources from the same origin
        "script-src 'self' https://cdn.jsdelivr.net https://code.jquery.com 'unsafe-inline'; "  # Allow scripts from jsdelivr.net and code.jquery.com
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "  # Allow inline styles and from jsdelivr.net
        "img-src 'self' data:; "  # Allow images from self and data URIs
        "font-src 'self' https://cdn.jsdelivr.net; "  # Allow fonts from jsdelivr.net
        "connect-src 'self' https://ipapi.co; "  # Allow API calls to ipapi.co
        "object-src 'none'; "  # Disallow object and embed elements
        "frame-ancestors 'none'; "  # Prevent clickjacking by disallowing framing
    )
    return response


@app.route('/', methods=['GET'])
def index():
    rendered_page = render_template('home.html')
    response = make_response(rendered_page)
    response.headers['ngrok-skip-browser-warning'] = 'any_value'
    return response


@app.route('/search-jobs', methods=['POST'])
@limiter.limit("10 per minute")
def search_jobs():
    """Handle job search requests by dynamically selecting the fetcher (scraper or JSearch)."""
    result = validate_and_clean_input(request.form, request.files)
    if isinstance(result, tuple) and result[1] == 400:
        return result

    logger.info(f"Incoming request. Terms: {result['search_terms']} - Location: {result['location']} - Posted since: {result['interval']}")
    honeypot = request.form.get('jamesbond')
    if honeypot:  # Bots will fill this, humans won't
        logger.warning("Bot detected!")
        return jsonify({'error': 'Something fishy is going on!'}), 400

    try:
        # Use the dynamically selected job fetching function
        all_jobs_df = asyncio.run(job_fetching_function(result['search_terms'], result['location'], Config.DEFAULT_RADIUS, result['interval']))

        # Check if the DataFrame is empty (i.e., no jobs found)
        if all_jobs_df.empty:
            return jsonify({
                'jobs': [],
                'message': 'No jobs found 😔. Please try again with different terms or locations.'
            }), 200  # Return a 200 status code with an empty job list and a message.

        all_jobs_df = process_job_dataframe(all_jobs_df)
        ranked_jobs_df = rank_job_descriptions(all_jobs_df, result['cv_text'], result['keywords'])
        dump_ranked_jobs(ranked_jobs_df, Config.DUMP_FILE_NAME)
        ranked_jobs = ranked_jobs_df[['display_title', 'job_url', 'display_company', 'date_posted', 'combined_score', 'tier']].head(Config.RESULTS_WANTED).to_dict(orient='records')

        del all_jobs_df, ranked_jobs_df # Free DataFrames explicitly after use
        gc.collect() # Force garbage collection
        return jsonify({'jobs': ranked_jobs})
    except Exception as e:
        logger.error(f"Error fetching jobs: {str(e)}", exc_info=True)  # Log the full traceback for debugging
        return jsonify({'error': 'An unexpected error occurred while fetching jobs 🫤. Please try again later.'}), 500


# Error handler for file size limit
@app.errorhandler(RequestEntityTooLarge)
def handle_file_size_error(e):
    return jsonify({'error': 'File size exceeds the limit.'}), 413

@app.route('/healthz', methods=['GET'])
def health_check():
    """Health check endpoint to ensure the app is running."""
    return jsonify({"status": "healthy"}), 200

@app.route('/robots.txt')
def robots_txt():
    return send_from_directory(app.static_folder, 'robots.txt')

if __name__ == '__main__':
    app.run(port=5000)