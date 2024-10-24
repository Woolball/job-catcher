import os
from dotenv import load_dotenv
import logging

# Initialize logger
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

class Config:
    TEMPLATE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    STATIC_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
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

    LANG_COVERAGE_SBERT = ('ar, bg, ca, cs, da, de, el, en, es, et, fa, fi, fr, fr, gl, gu, he, hi, hr, hu, hy, id,'
                           ' it, ja, ka, ko, ku, lt, lv, mk, mn, mr, ms, my, nb, nl, pl, pt, ro, ru, sk, sl, sq,'
                           ' sr, sv, th, tr, uk, ur, vi, zh').split(', ')
    LANG_COVERAGE_LANGDETECT = ('af, am, an, ar, as, az, be, bg, bn, br, bs, ca, cs, cy, da, de, dz, el, en, eo, es, et,'
                                ' eu, fa, fi, fo, fr, ga, gl, gu, he, hi, hr, ht, hu, hy, id, is, it, ja, jv, ka, kk, km,'
                                ' kn, ko, ku, ky, la, lb, lo, lt, lv, mg, mk, ml, mn, mr, ms, mt, nb, ne, nl, nn, no, oc,'
                                ' or, pa, pl, ps, pt, qu, ro, ru, rw, se, si, sk, sl, sq, sr, sv, sw, ta, te, th, tl, tr,'
                                ' ug, uk, ur, vi, vo, wa, xh, zh, zu').split(', ')


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
