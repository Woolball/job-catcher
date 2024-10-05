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

# Flask app setup
app = Flask(__name__)
app.config.from_object(Config)

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the job fetching function globally
job_fetching_function = None

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
    all_jobs_df = process_job_dataframe(all_jobs_df)
    ranked_jobs_df = rank_job_descriptions(result['cv_text'], all_jobs_df, result['keywords'])
    dump_ranked_jobs(ranked_jobs_df, Config.DUMP_FILE_NAME)
    ranked_jobs = ranked_jobs_df[['display_title', 'job_url', 'combined_score', 'display_company', 'date_posted']].head(Config.RESULTS_WANTED).to_dict(orient='records')
    return jsonify({'jobs': ranked_jobs})


# Error handler for file size limit
@app.errorhandler(RequestEntityTooLarge)
def handle_file_size_error(e):
    return jsonify({'error': 'File size exceeds the limit.'}), 413


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


def main(fetcher, fetcher_modules):
    """Main function to run the Flask app with dynamic job fetcher selection."""
    global job_fetching_function
    if fetcher in fetcher_modules:
        logger.info(f"Using {fetcher}-based job fetching.")
        job_fetching_function = fetcher_modules[fetcher]
    else:
        raise ValueError(f"Unknown fetcher: {fetcher}")

    app.run(debug=True, port=5001)


if __name__ == '__main__':
    # Load fetchers modules from the src/fetchers/ directory
    fetcher_modules = load_fetcher_modules("src/fetchers")

    # Argument parser to dynamically populate fetcher from fetchers modules
    parser = argparse.ArgumentParser(description="Run the Job Matcher app with dynamically loaded job fetchers.")
    parser.add_argument('--fetcher', choices=fetcher_modules.keys(), default='scraper',
                        help="Specify the job fetcher.")

    args = parser.parse_args()
    main(args.fetcher, fetcher_modules)