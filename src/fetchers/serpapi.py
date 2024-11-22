import pandas as pd
import logging
import asyncio
import aiohttp
import random
from config import Config
from ..utils import devise_date_from_human_readable, filter_jobs_by_date

# Constants for SerpAPI
API_URL = Config.SERP_API_URL
API_KEY = Config.SERP_API_KEY

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Utility to rename columns for internal use
def rename_serpapi_columns(jobs_df):
    jobs_df = jobs_df.rename(columns={
        'title': 'title',
        'company_name': 'company',
        'location': 'location',
        'description': 'description',
        'thumbnail': 'thumbnail',
        'date_posted': 'date_posted',
    })

    # Rename keys inside the `apply_options` column
    if 'apply_options' in jobs_df.columns:
        jobs_df['apply_options'] = jobs_df['apply_options'].apply(lambda options: [
            {k.replace('title', 'publisher').replace('link', 'apply_link'): v for k, v in option.items()}
            for option in options
        ] if isinstance(options, list) else options)
    return jobs_df


async def fetch_jobs_for_search_term(session, search_term, country, location, retries=3, backoff_factor=0.5):
    """
    Fetch jobs for a single search term from SerpAPI with retries, error handling, and pagination.
    """
    params = {
        "engine": "google_jobs",
        "q": f"{search_term}",
        "location": f'{location}' if location else country['name'],
        "gl": country['code'],
        "api_key": API_KEY,
    }

    all_jobs = []
    next_page_token = None

    for page in range(Config.NUM_SEARCH_PAGES):
        if next_page_token:
            params["next_page_token"] = next_page_token

        for attempt in range(retries):
            try:
                async with session.get(API_URL, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        jobs = data.get("jobs_results", [])
                        logger.info(
                            f"Fetched {len(jobs)} jobs for search term '{search_term}' on page {page + 1}, attempt {attempt + 1}."
                        )
                        all_jobs.extend(jobs)

                        # Get next page token
                        next_page_token = data.get("serpapi_pagination", {}).get("next_page_token")
                        break  # Exit retry loop on success
                    else:
                        logger.warning(
                            f"Error fetching jobs (status {response.status}) for term '{search_term}' on page {page + 1} - Attempt {attempt + 1}/{retries}"
                        )
            except aiohttp.ClientError as e:
                logger.error(f"Client error: {e} - Page {page + 1}, Attempt {attempt + 1}/{retries}")
            except asyncio.TimeoutError as e:
                logger.error(f"Timeout error: {e} - Page {page + 1}, Attempt {attempt + 1}/{retries}")

            # Exponential backoff with jitter
            delay = backoff_factor * (2 ** attempt) + random.uniform(0, 0.1)
            await asyncio.sleep(delay)

        if not next_page_token:
            logger.info(f"No more pages available after page {page + 1}.")
            break

    logger.info(f"Total jobs fetched for '{search_term}': {len(all_jobs)}")
    return all_jobs


async def fetch_jobs(search_terms, country, location, interval):
    """
    Fetch jobs concurrently for multiple search terms and process them as a DataFrame.
    """
    day_interval = Config.INTERVAL_MAPPING.get(interval, 30)

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_jobs_for_search_term(session, term, country, location) for term in search_terms]
        results = await asyncio.gather(*tasks)

        # Flatten results into a list of jobs
        all_jobs = [job for result in results for job in result if result]
        if not all_jobs:
            return pd.DataFrame()
        all_jobs_df = pd.DataFrame(all_jobs)
        print(all_jobs_df['detected_extensions'])

        # Extract and handle 'posted_at'
        all_jobs_df['posted_at'] = all_jobs_df.apply(
            lambda job: job['detected_extensions']['posted_at']
            if 'detected_extensions' in job and isinstance(job['detected_extensions'], dict) and 'posted_at' in job['detected_extensions']
            else 'today',
            axis=1
        )

        all_jobs_df['posted_at'] = all_jobs_df['posted_at'].replace(['', None], 'today').fillna('today')
        all_jobs_df = devise_date_from_human_readable(all_jobs_df, 'posted_at', 'date_posted')
        all_jobs_df = filter_jobs_by_date(all_jobs_df, day_interval, 'date_posted')
        if not all_jobs_df.empty:
            all_jobs_df = rename_serpapi_columns(all_jobs_df)
        return all_jobs_df


# Example usage
if __name__ == "__main__":
    search_terms = ["UX Researcher", "UX Designer"]
    location = "Vaud"
    country = {'name': "Switzerland", 'code': 'ch'}
    interval = 'month'

    all_jobs_df = asyncio.run(fetch_jobs(search_terms, country, location, interval))
    print(all_jobs_df['apply_options'])
