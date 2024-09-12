"""
Job Matcher
Copyright (C) 2024 Ammar Halabi

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at your option)
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
import csv
from flask import Flask, request, render_template, jsonify
import pandas as pd
import logging
from datetime import datetime
from dateutil.parser import parse
from jobspy import scrape_jobs

import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'txt', 'json'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DB_FILE_NAME = "data/job_search_cache.db"
DUMP_FILE_NAME = "data/dump_search_scraper.csv"
INTERVAL_MAPPING = {'month': 30, 'week': 7, '3days': 3, 'today': 1}
EXCLUDED_JOB_PUBLISHERS = "BeBee, Learn4Good, Joinrs"
RESULTS_WANTED_FROM_SCRAPERS = 50

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer('paraphrase-MiniLM-L6-v2').to(device)
tfidf_vectorizer = TfidfVectorizer()

# Utility functions

def preprocess_text(text):
    """Cleans text by removing special characters and converting to lowercase."""
    if not text:
        return ''
    return ''.join(c for c in text.lower().replace('\n', ' ') if c.isalnum() or c.isspace())


def process_job_dataframe(jobs_df):
    """Standardize and preprocess the job DataFrame efficiently."""
    if not jobs_df.empty:
        # Handle NaN values directly with fillna for efficiency
        jobs_df['date_posted'].fillna(datetime.today().strftime('%Y-%m-%d'), inplace=True)
        # Use vectorized operations for datetime parsing and formatting
        def process_date(date_value):
            try:
                # Parse string dates and convert to the desired format
                parsed_date = pd.to_datetime(date_value, errors='coerce')
                return parsed_date.strftime("%b %d")
            except Exception:
                return 'NA'
        # Apply the function to the column
        jobs_df['date_posted'] = jobs_df['date_posted'].apply(process_date)
        # Process other fields with standard pandas operations
        jobs_df['description'] = jobs_df['description'].fillna('').apply(preprocess_text)
        jobs_df['title'] = jobs_df['title'].str.strip()
        jobs_df['company'] = jobs_df['company'].str.strip()
    return jobs_df


def calculate_keyword_score(cv_text, job_text, keywords):
    """Calculate keyword match score based on overlap of CV and job description."""
    normalized_keywords = [keyword.lower().strip() for keyword in keywords]
    if len(normalized_keywords) == 0:
        return 0.0
    matching_keywords = set(
        keyword for keyword in normalized_keywords if keyword in cv_text.lower() and keyword in job_text.lower()
    )
    return len(matching_keywords) / len(normalized_keywords)


def calculate_similarity_scores(cv_text, job_texts, keywords):
    """Calculate similarity scores using TF-IDF, SBERT, and keyword matching."""
    # TF-IDF
    documents = [cv_text] + job_texts
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    tfidf_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # SBERT Embeddings
    embeddings = model.encode([cv_text] + job_texts, convert_to_tensor=True).to(device).cpu().numpy()
    sbert_scores = cosine_similarity(embeddings[0:1], embeddings[1:]).flatten()

    # Keyword Matching
    keyword_scores = [calculate_keyword_score(cv_text, jd, keywords) for jd in job_texts]

    return tfidf_scores, sbert_scores, keyword_scores


def rank_job_descriptions(cv_text, jobs_df, keywords):
    """Rank job descriptions based on combined similarity, keyword, and TF-IDF scores."""
    tfidf_scores, sbert_scores, keyword_scores = calculate_similarity_scores(
        cv_text, jobs_df['description'].tolist(), keywords
    )

    jobs_df['tfidf_score'] = tfidf_scores
    jobs_df['sbert_similarity'] = sbert_scores
    jobs_df['keyword_score'] = keyword_scores

    # Normalize scores and calculate combined score
    for score in ['sbert_similarity', 'keyword_score', 'tfidf_score']:
        range = (jobs_df[score].max() - jobs_df[score].min())
        if range == 0:
            jobs_df[score] = 0
        else:
            jobs_df[score] = (jobs_df[score] - jobs_df[score].min()) / (jobs_df[score].max() - jobs_df[score].min())

    jobs_df['combined_score'] = jobs_df.apply(
        lambda row: np.mean(sorted([row['sbert_similarity'], row['keyword_score'], row['tfidf_score']], reverse=True)[:2]), axis=1
    )

    return jobs_df.sort_values(by='combined_score', ascending=False)


def fetch_jobs(search_term, location, distance, interval):
    """Fetch and preprocess job descriptions from various sources."""
    logger.info(f"Searching for jobs with term: {search_term}")
    jobs = scrape_jobs(
        site_name=["indeed", "linkedin", "glassdoor"],
        search_term=search_term,
        location=location,
        distance=distance,
        results_wanted=RESULTS_WANTED_FROM_SCRAPERS,
        hours_old=INTERVAL_MAPPING.get(interval, 30) * 24,
        country_indeed='Switzerland',
        linkedin_fetch_description=True,
    )
    if not jobs.empty:
        jobs = jobs[['title', 'company', 'location', 'date_posted', 'job_url', 'description']].copy()
        return process_job_dataframe(jobs)
    return pd.DataFrame()


def fetch_jobs_multiple(search_func, search_terms, location, interval='month', radius=None):
    """Fetch jobs from scrapers for multiple search terms."""
    all_jobs_df = pd.DataFrame()
    for search_term in search_terms:
        new_jobs_df = search_func(search_term, location, radius, interval)
        all_jobs_df = pd.concat([all_jobs_df, new_jobs_df], ignore_index=True)
    all_jobs_df = all_jobs_df.drop_duplicates(subset=['title', 'company'])
    return all_jobs_df


def dump_ranked_jobs(ranked_jobs_df, file_path):
    """Save the ranked jobs to a CSV file."""
    logger.info(f"Saving ranked job descriptions to {DUMP_FILE_NAME}...")
    ranked_jobs_df.to_csv(file_path, quoting=csv.QUOTE_NONNUMERIC, escapechar="\\", index=False)
    logger.info("Ranked job descriptions saved successfully.")


@app.route('/', methods=['GET'])
def index():
    """Renders the main page with the job search form."""
    return render_template('index.html')


def handle_search_jobs_request(request, search_func, results_wanted, dump_file_name):
    """Handles the common logic for search-jobs across multiple implementations."""
    search_terms = request.form['search_terms'].split(',')
    location = request.form['location'].strip()
    interval = request.form.get('posted_since', 'month')
    radius = request.form.get('distance', None)
    keywords = request.form['keywords'].split(',')
    cv_text = preprocess_text(request.form['cv_text'])

    # Fetch jobs using the provided search function (JSearch or Scraper)
    all_jobs_df = fetch_jobs_multiple(search_func, search_terms, location, interval, radius)

    if not all_jobs_df.empty:
        # Rank the fetched jobs and return as JSON
        ranked_jobs_df = rank_job_descriptions(cv_text, all_jobs_df, keywords)
        dump_ranked_jobs(ranked_jobs_df, dump_file_name)  # Dump to CSV for logging
        ranked_jobs = ranked_jobs_df[['title', 'job_url', 'combined_score', 'company', 'date_posted']].head(results_wanted).to_dict(orient='records')
        return jsonify({'jobs': ranked_jobs})
    else:
        return jsonify({'jobs': [], 'message': 'No jobs found.'})


@app.route('/search-jobs', methods=['POST'])
def search_jobs():
    """Scraper-based search."""
    return handle_search_jobs_request(request, fetch_jobs, 50, DUMP_FILE_NAME)


if __name__ == '__main__':
    app.run(debug=True, port=5000)