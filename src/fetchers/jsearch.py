import pandas as pd
import logging
import redis
import time
import asyncio
import aiohttp
from config import Config

# Constants for JSearch
API_URL = Config.JSEARCH_API_URL
API_HEADERS = {
    "x-rapidapi-key": Config.JSEARCH_API_KEY,
    "x-rapidapi-host": Config.JSEARCH_API_HOST
}
RATE_LIMIT_CALLS = Config.JSEARCH_API_RATE_LIMIT_CALLS  # Number of allowed API calls
RATE_LIMIT_PERIOD = Config.JSEARCH_API_RATE_LIMIT_PERIOD + 0.1  # Time period in seconds (add 0.1  sec as safety buffer)

logger = logging.getLogger(__name__)

# Initialize Redis connection
redis_client = redis.StrictRedis(host=Config.REDIS_HOST, port=Config.REDIS_PORT, db=0)

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


async def make_jsearch_api_request(session, querystring):
    """
    Make a request to the JSearch API asynchronously.
    """
    async with session.get(API_URL, headers=API_HEADERS, params=querystring) as response:
        if response.status == 200:
            return await response.json()
        else:
            logger.error(f"Error fetching jobs from JSearch: {response.status}")
            return None


async def enforce_rate_limit():
    """
    Ensure that no more than 5 API calls are made within the last second.
    """
    while True:
        current_time = time.time()
        # Retrieve all timestamps from the Redis list
        timestamps = redis_client.lrange(REDIS_KEY, 0, -1)
        print(f"All timestamps: {timestamps}")
        # Convert Redis bytes to float timestamps and filter out timestamps older than 1 second
        valid_timestamps = [float(ts.decode()) for ts in timestamps if
                            float(ts.decode()) > current_time - RATE_LIMIT_PERIOD]
        print(f"Time back: {current_time - RATE_LIMIT_PERIOD}")
        print(f"Valid timestamps: {valid_timestamps}")
        # Remove old timestamps from Redis (timestamps older than 1 second)
        if len(valid_timestamps) == 0:
            redis_client.delete(REDIS_KEY)
        else:
            redis_client.ltrim(REDIS_KEY, -len(valid_timestamps), -1)
        # Check how many calls have been made in the last second
        call_count = len(valid_timestamps)

        # If we have room to make an API call, proceed
        if call_count < RATE_LIMIT_CALLS:
            break

        # If rate limit is hit, wait before trying again
        logger.info("Rate limit hit. Waiting for 1 second...")
        await asyncio.sleep(1)  # Wait for 1 second


# Fetch jobs asynchronously using the JSearch API with rate limiting
async def fetch_jobs_for_search_term(session, search_term, location, radius, interval):
    """
    Asynchronous task to fetch jobs for a single search term.
    """
    querystring = {
        "query": f"{search_term} in {location}",
        "date_posted": interval,
        "exclude_job_publishers": Config.EXCLUDED_JOB_PUBLISHERS
    }
    if radius:
        querystring["radius"] = radius

    # Enforce the rate limit before making an API request
    await enforce_rate_limit()

    # Log the request time in Redis **before** sending the request
    redis_client.rpush(REDIS_KEY, time.time())
    print(f"Pushing to Redis at time: {time.time()}")

    # Make the request asynchronously
    data = await make_jsearch_api_request(session, querystring)

    if data:
        # Process and return the jobs
        jobs_df = pd.DataFrame(data.get('data', []))
        jobs_df = rename_jsearch_columns(jobs_df)
        return jobs_df
    else:
        return pd.DataFrame()


async def fetch_jobs(search_terms, location, radius, interval):
    """Fetch jobs concurrently for multiple search terms, while respecting the API rate limit."""
    if not Config.JSEARCH_API_KEY:
        logger.error("JSearch API key is missing. Please set JSEARCH_API_KEY in the environment.")
        raise ValueError("API key is missing. Please configure the API key in the environment.")

    async with aiohttp.ClientSession() as session:
        # Create a list of tasks for each search term
        tasks = [
            fetch_jobs_for_search_term(session, search_term, location, radius, interval)
            for search_term in search_terms
        ]

        # Run all tasks concurrently and gather results
        all_jobs_df = pd.DataFrame()
        results = await asyncio.gather(*tasks)

        for jobs_df in results:
            all_jobs_df = pd.concat([all_jobs_df, jobs_df], ignore_index=True)

        return all_jobs_df.drop_duplicates(subset=['title', 'company'])

