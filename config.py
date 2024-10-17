import os
from dotenv import load_dotenv
import logging

# Initialize logger
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

class Config:
    UPLOAD_FOLDER = 'uploads/'
    DATA_FOLDER = 'data/'
    STATIC_FOLDER = 'static/'
    MAX_CONTENT_LENGTH = 2 * 1024 * 1024  # 2MB limit
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'rtf'}
    ALLOWED_MIME_TYPES = {
        'application/pdf',
        'application/msword',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'text/plain',
        'application/rtf'
    }
    CV_TEXT_LIMIT = 2000 #limit cv text to first 2000 words
    RESULTS_WANTED = 50
    DUMP_FILE_NAME = 'data/dump_search.csv'
    DEFAULT_RADIUS = 50
    INTERVAL_MAPPING = {'month': 30, 'week': 7, '3days': 3, 'today': 1}
    EXCLUDED_JOB_PUBLISHERS = "BeBee, Learn4Good, Joinrs"

    FETCHER_NAME = os.getenv('FETCHER', 'scraper').lower()

    # API-related settings (optional for JSearch)
    JSEARCH_API_URL = os.getenv("JSEARCH_API_URL", "https://jsearch.p.rapidapi.com/search")  # Default URL
    JSEARCH_API_KEY = os.getenv("JSEARCH_API_KEY", "")  # Default to empty string if not provided
    JSEARCH_API_HOST = os.getenv("JSEARCH_API_HOST", "jsearch.p.rapidapi.com")  # Default host
    JSEARCH_API_RATE_LIMIT_CALLS = int(os.getenv("JSEARCH_API_RATE_LIMIT_CALLS", 5)) # Default to 5 calls
    JSEARCH_API_RATE_LIMIT_PERIOD = float(os.getenv("JSEARCH_API_RATE_LIMIT_PERIOD", 1)) # Default to 1 second

    REDIS_HOST = os.getenv("REDIS_HOST", "")
    REDIS_PORT = os.getenv("REDIS_PORT", 6379)

    # Check essential environment variables
    if FETCHER_NAME == "jsearch" and not JSEARCH_API_KEY:
        raise RuntimeError("Required environment variable 'JSEARCH_API_KEY' is not set. THis is required when working with the {FETCHER_NAME} fetcher.")
    if FETCHER_NAME == "jsearch" and not REDIS_HOST:
        logger.warning("Redis host is not provided in .env --> defaulting to localhost. Make sure to have a local Redis server functional.")
        REDIS_HOST = "localhost"

    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
