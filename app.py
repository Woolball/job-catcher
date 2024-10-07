import os
import importlib
import argparse
from flask import Flask, request, jsonify, render_template, make_response
from werkzeug.exceptions import RequestEntityTooLarge
from config import Config
from src.ranking import rank_job_descriptions
from src.utils import validate_and_clean_input, dump_ranked_jobs, process_job_dataframe
import logging
import asyncio
import gc

# Flask app setup
app = Flask(__name__)
app.config.from_object(Config)

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
        raise ValueError(f"Unknown fetcher: {fetcher_name}. Check the .env setting and your fetcher implementation in src/fetchers/")


# Load fetcher modules and initialize job fetching function
fetcher_modules = load_fetcher_modules("src/fetchers")
job_fetching_function = load_job_fetching_function(Config.FETCHER_NAME, fetcher_modules)

@app.route('/', methods=['GET'])
def index():
    rendered_page = render_template('home.html')
    response = make_response(rendered_page)
    response.headers['ngrok-skip-browser-warning'] = 'any_value'
    return response


@app.route('/search-jobs', methods=['POST'])
def search_jobs():
    """Handle job search requests by dynamically selecting the fetcher (scraper or JSearch)."""
    result = validate_and_clean_input(request.form, request.files)
    if isinstance(result, tuple) and result[1] == 400:
        return result

    # Use the dynamically selected job fetching function
    all_jobs_df = asyncio.run(job_fetching_function(result['search_terms'], result['location'], Config.DEFAULT_RADIUS, result['interval']))

    # Check if the DataFrame is empty (i.e., no jobs found)
    if all_jobs_df.empty:
        return jsonify({
            'jobs': [],
            'message': 'No jobs found 😔. Please try again with different terms or locations.'
        }), 200  # Return a 200 status code with an empty job list and a message.

    all_jobs_df = process_job_dataframe(all_jobs_df)
    ranked_jobs_df = rank_job_descriptions(result['cv_text'], all_jobs_df, result['keywords'])
    dump_ranked_jobs(ranked_jobs_df, Config.DUMP_FILE_NAME)
    ranked_jobs = ranked_jobs_df[['display_title', 'job_url', 'combined_score', 'display_company', 'date_posted']].head(Config.RESULTS_WANTED).to_dict(orient='records')

    del all_jobs_df, ranked_jobs_df # Free DataFrames explicitly after use
    gc.collect() # Force garbage collection
    return jsonify({'jobs': ranked_jobs})


# Error handler for file size limit
@app.errorhandler(RequestEntityTooLarge)
def handle_file_size_error(e):
    return jsonify({'error': 'File size exceeds the limit.'}), 413

@app.route('/healthz', methods=['GET'])
def health_check():
    """Health check endpoint to ensure the app is running."""
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(port=5000)