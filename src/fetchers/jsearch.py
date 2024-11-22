import pandas as pd
import logging
import redis
import time
import asyncio
import aiohttp
from config import Config
import random
from ..utils import devise_date_from_human_readable, filter_jobs_by_date


# Constants for JSearch
API_URL = Config.JSEARCH_API_URL
API_HEADERS = {
    "x-rapidapi-key": Config.JSEARCH_API_KEY,
    "x-rapidapi-host": Config.JSEARCH_API_HOST
}
RATE_LIMIT_CALLS = Config.JSEARCH_API_RATE_LIMIT_CALLS  # Number of allowed API calls
RATE_LIMIT_PERIOD = Config.JSEARCH_API_RATE_LIMIT_PERIOD  # Time period in seconds

logger = logging.getLogger(__name__)

# Initialize Redis connection with error handling
try:
    redis_client = redis.StrictRedis(host=Config.REDIS_HOST, port=Config.REDIS_PORT, db=0)
    redis_client.ping()  # Test connection
except redis.exceptions.ConnectionError as e:
    logger.error(f"Redis connection failed: {e}")
    redis_client = None  # Fallback: Redis unavailable, could implement in-memory rate-limiting if necessary

# Redis key to track API call timestamps
REDIS_KEY = "jsearch_api_call_timestamps"

# Utility function to rename JSearch columns to match internal structure
def rename_jsearch_columns(jobs_df):
    return jobs_df.rename(columns={
        'job_title': 'title',
        'employer_name': 'company',
        'job_city': 'location',
        'job_description': 'description',
        'job_apply_link': 'job_url',
        'job_posted_at_datetime_utc': 'date_posted'
    })


# Rate limiting with Redis error handling
async def enforce_rate_limit():
    """
    Ensure that no more than RATE_LIMIT_CALLS API calls are made within the RATE_LIMIT_PERIOD.
    """
    if not redis_client:
        logger.warning("Rate limiting is disabled: Redis unavailable")
        return

    while True:
        current_time = time.time()
        try:
            timestamps = redis_client.lrange(REDIS_KEY, 0, -1)
            valid_timestamps = [float(ts.decode()) for ts in timestamps if float(ts.decode()) > current_time - RATE_LIMIT_PERIOD]

            # **Trim the Redis list to only valid timestamps** (those within the RATE_LIMIT_PERIOD)
            redis_client.ltrim(REDIS_KEY, -len(valid_timestamps), -1)

            # If we have room to make an API call, proceed
            if len(valid_timestamps) < RATE_LIMIT_CALLS:
                break

            logger.info("Rate limit hit. Waiting for 1 second...")
            await asyncio.sleep(RATE_LIMIT_PERIOD + 0.5) # Wait for 1 second plus 0.5 for safety

        except redis.exceptions.RedisError as e:
            logger.error(f"Redis error during rate limit enforcement: {e}")
            break  # If Redis is failing, skip rate limiting

    # Record current timestamp for rate limiting
    redis_client.rpush(REDIS_KEY, current_time)


# Fetch jobs for a single search term, with retry, backoff, and rate limit enforcement combined
async def fetch_jobs_for_search_term(session, search_term, country, location, interval, retries=3, backoff_factor=0.5):
    """
    Fetch jobs for a single search term from JSearch API, with retry and rate limiting.
    """
    querystring = {
        "query": f"{search_term} in {location}" if location else search_term,
        "country": country['code'],
        "date_posted": interval,
        "num_pages" : Config.NUM_SEARCH_PAGES,
        #"exclude_job_publishers": " ".join(Config.EXCLUDED_JOB_PUBLISHERS),
    }

    for attempt in range(retries):
        await enforce_rate_limit()  # Ensure rate limit is respected before every attempt
        try:
            async with session.get(API_URL, headers=API_HEADERS, params=querystring) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"Error fetching jobs (status {response.status}) for query: {querystring} - Attempt {attempt + 1}/{retries}")
        except aiohttp.ClientError as e:
            logger.error(f"Client error during JSearch API call: {e} - Attempt {attempt + 1}/{retries}")
        except asyncio.TimeoutError as e:
            logger.error(f"Timeout during JSearch API call: {e} - Attempt {attempt + 1}/{retries}")
        except Exception as e:
            logger.error(f"Unexpected error during JSearch API call: {e} - Attempt {attempt + 1}/{retries}")

        # Exponential backoff with jitter
        delay = backoff_factor * (2 ** attempt) + random.uniform(0, 0.1)
        await asyncio.sleep(delay)

    logger.error(f"Failed to fetch jobs from JSearch after {retries} attempts.")
    return None


# Fetch jobs concurrently for multiple search terms
async def fetch_jobs(search_terms, country, location, interval):
    """
    Fetch jobs concurrently for multiple search terms, handling retries, rate limits, and errors.
    """
    if not Config.JSEARCH_API_KEY:
        raise ValueError("JSearch API key is missing. Please set JSEARCH_API_KEY in the environment.")

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_jobs_for_search_term(session, term, country, location, interval) for term in search_terms]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        all_jobs_df = pd.DataFrame()
        for data in results:
            # Skip any failed tasks or invalid results (like dicts or errors)
            if isinstance(data, Exception):
                logger.error(f"Error in search task: {data}")
                continue
            if isinstance(data, dict) and 'data' in data:
                jobs_df = pd.DataFrame(data.get('data', []))
                jobs_df = rename_jsearch_columns(jobs_df)
                all_jobs_df = pd.concat([all_jobs_df, jobs_df], ignore_index=True)
            else:
                logger.warning(f"Unexpected result type: {type(data)}, skipping this result.")

        # additional logic to filter jsearch results due to api malfunction
        if not all_jobs_df.empty:
            all_jobs_df['job_posted_human_readable'] = all_jobs_df['job_posted_human_readable'].replace(['', None], 'today').fillna('today')
            day_interval = Config.INTERVAL_MAPPING.get(interval, 30)
            all_jobs_df = devise_date_from_human_readable(all_jobs_df, 'job_posted_human_readable', 'date_posted')
            all_jobs_df = filter_jobs_by_date(all_jobs_df, day_interval, 'date_posted')

        return all_jobs_df
