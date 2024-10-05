import pandas as pd
from jobspy import scrape_jobs
from config import Config
import logging

logger = logging.getLogger(__name__)


def fetch_jobs(search_terms, location, radius, interval):
    all_jobs_df = pd.DataFrame()
    for search_term in search_terms:
        jobs_df = scrape_jobs(
            site_name=["indeed", "linkedin", "glassdoor"],
            search_term=search_term,
            location=location,
            distance=radius,
            results_wanted=Config.RESULTS_WANTED,
            hours_old=Config.INTERVAL_MAPPING.get(interval, 30) * 24,
            country_indeed= location.split(', '),
            linkedin_fetch_description=True,
        )
        all_jobs_df = pd.concat([all_jobs_df, jobs_df], ignore_index=True)
    return all_jobs_df.drop_duplicates(subset=['title', 'company'])