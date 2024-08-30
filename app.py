"""
Job Matcher
Copyright (C) 2023 Your Name

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

from flask import Flask, request, render_template
import sqlite3
import pandas as pd
import logging
from datetime import datetime
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
SEARCH_TERMS = ["UX Designer", "User Experience", "UX Manager", "UX Researcher", "UX Research", "User Research",
                "UX Strategy", "Customer Insights", "Customer Experience", "Product Strategy",
                "Service Design", "Product Design"]
KEYWORDS = [
    "UX Research", "User Experience Research", "User Research", "Research", "Method", "Qualitative",
    "Usability", "Usability Engineer", "Human Factors", "Strategy", "Insights", "Customer Insights",
    "UX", "User Experience", "Customer Experience", "User Engagement", "User Satisfaction",
    "Product Design", "Interaction Design", "Human-Computer Interaction", "HCI",
    "Lead", "Senior", "Principal", "Leader", "Director"
]

# Initialize SBERT model and TF-IDF Vectorizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer('paraphrase-MiniLM-L6-v2').to(device)
tfidf_vectorizer = TfidfVectorizer()

def preprocess_text(text):
    """Preprocess the text by removing special characters and converting to lower case."""
    return ''.join(c for c in text.lower().replace('\n', ' ') if c.isalnum() or c.isspace())

def get_embedding(text):
    """Generate embeddings for the input text using SBERT."""
    return model.encode(text, convert_to_tensor=True).to(device).cpu().numpy()

def calculate_keyword_score(cv_text, job_text, keywords):
    """Calculate keyword matching score based on the presence of important keywords."""
    normalized_keywords = [keyword.lower() for keyword in keywords]
    matching_keywords = set(
        keyword for keyword in normalized_keywords if keyword in cv_text.lower() and keyword in job_text.lower())
    return len(matching_keywords) / len(normalized_keywords)

def calculate_tfidf_score(cv_text, job_texts):
    """Calculate TF-IDF similarity score between the CV and each job description."""
    documents = [cv_text] + job_texts
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    return cosine_similarities

def rank_job_descriptions(cv_text, jobs_df, keywords):
    """Rank job descriptions based on combined similarity, keyword, and TF-IDF scores."""
    jobs_df['tfidf_score'] = calculate_tfidf_score(cv_text, jobs_df['description'].tolist())
    cv_embedding = get_embedding(cv_text)
    jobs_df['sbert_similarity'] = jobs_df['description'].apply(
        lambda jd: cosine_similarity(cv_embedding.reshape(1, -1), get_embedding(jd).reshape(1, -1))[0][0]
    )
    jobs_df['keyword_score'] = jobs_df['description'].apply(
        lambda jd: calculate_keyword_score(cv_text, jd, keywords)
    )
    for score in ['sbert_similarity', 'keyword_score', 'tfidf_score']:
        min_score, max_score = jobs_df[score].min(), jobs_df[score].max()
        jobs_df[score] = (jobs_df[score] - min_score) / (max_score - min_score)
    jobs_df['combined_score'] = jobs_df.apply(
        lambda row: np.mean(sorted([row['sbert_similarity'], row['keyword_score'], row['tfidf_score']], reverse=True)[:2]), axis=1
    )
    return jobs_df.sort_values(by='combined_score', ascending=False)

def cache_job_results(search_term, location, distance, days_old, conn):
    """Cache job results in the database, avoiding duplicates."""
    cur = conn.cursor()
    cur.execute('''SELECT MAX(last_searched) FROM jobs WHERE search_term = ? AND location = ? AND distance = ?''',
                (search_term, location, distance))
    last_search_date = cur.fetchone()[0]
    if last_search_date:
        last_search_date = datetime.strptime(last_search_date, '%Y-%m-%d')
        days_since_last_search = (datetime.today() - last_search_date).days
        days_back = min(days_since_last_search, days_old)
        logger.info(f"Last search for {search_term} was on {last_search_date}.")
    else:
        days_back = days_old
    if days_back:
        new_jobs_df = fetch_jobs(search_term, location, distance, days_back)
        if not new_jobs_df.empty:
            new_jobs_df.drop_duplicates(subset=['title', 'company'], keep='first', inplace=True)
            new_jobs_df.to_sql('jobs', conn, if_exists='append', index=False)
            logger.info(f"Cached {len(new_jobs_df)} new job postings.")
        else:
            logger.warning("No new job postings to cache.")
    cur.close()

def fetch_jobs(search_term, location, distance, days_back):
    """Fetch and preprocess job descriptions from various sources."""
    logger.info(f"Searching for jobs with term: {search_term}")
    jobs = scrape_jobs(
        site_name=["indeed", "linkedin", "glassdoor"],
        search_term=search_term,
        location=location,
        distance=distance,
        results_wanted=100,
        hours_old=days_back * 24,
        country_indeed='Switzerland',
        linkedin_fetch_description=True,
    )
    if not jobs.empty:
        jobs['description'] = jobs['description'].fillna('')
        jobs['description'] = jobs['description'].apply(preprocess_text)
        jobs = jobs[['title', 'company', 'location', 'date_posted', 'job_url', 'description']].copy()
        jobs['search_term'] = search_term
        jobs['location'] = location
        jobs['distance'] = distance
        jobs['last_searched'] = datetime.today().strftime('%Y-%m-%d')
        return jobs
    return pd.DataFrame()

def retrieve_cached_jobs(search_term, location, distance, days_old, conn):
    """Retrieve cached jobs for a specific query."""
    cur = conn.cursor()
    query = '''SELECT title, company, location, date_posted, job_url, description 
               FROM jobs 
               WHERE search_term = ? AND location = ? AND distance = ? 
               AND date_posted >= DATE('now', ?) 
               ORDER BY date_posted DESC'''
    cur.execute(query, (search_term, location, distance, f'-{days_old} days'))
    rows = cur.fetchall()
    cur.close()
    if rows:
        return pd.DataFrame(rows, columns=['title', 'company', 'location', 'date_posted', 'job_url', 'description'])
    return pd.DataFrame()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def handle_file_uploads(request):
    cv_text = None
    params = None

    # Handle CV file upload
    if 'cv_file' in request.files:
        cv_file = request.files['cv_file']
        if cv_file and allowed_file(cv_file.filename):
            cv_text = preprocess_text(cv_file.read().decode('utf-8'))

    # Handle JSON params file upload
    if 'params_file' in request.files:
        params_file = request.files['params_file']
        if params_file and allowed_file(params_file.filename):
            params = json.load(params_file)

    return cv_text, params

def parse_json_params(params):
    if params:
        search_terms = [term.strip() for term in params.get('search_terms', '').split(',')]
        keywords = [keyword.strip() for keyword in params.get('keywords', '').split(',')]
        time_window = int(params.get('time_window', 7))
        location = params.get('location', '').strip()
        distance = int(params.get('distance', 0))
        return search_terms, keywords, time_window, location, distance
    return [], [], 14, '', 0

@app.route('/', methods=['GET', 'POST'])
def index():
    cv_text, params = None, None

    if request.method == 'POST':
        # Handle file uploads
        cv_text, params = handle_file_uploads(request)
        if params:
            search_terms, keywords, time_window, location, distance, results_wanted = parse_json_params(params)
        else:
            # Handle form inputs if no params file
            cv_text = cv_text or preprocess_text(request.form['cv_text'])
            search_terms = [term.strip() for term in request.form['search_terms'].split(',')]
            keywords = [keyword.strip() for keyword in request.form['keywords'].split(',')]
            time_window = int(request.form['time_window'].strip())
            location = request.form['location'].strip()
            distance = int(request.form['distance'].strip())

        # Process job search and ranking as before
        conn = sqlite3.connect(DB_FILE_NAME)
        for search_term in search_terms:
            cache_job_results(search_term.strip(), location, distance, time_window, conn)

        all_cached_jobs = pd.DataFrame()
        for search_term in search_terms:
            cached_jobs_df = retrieve_cached_jobs(search_term.strip(), location, distance, time_window, conn)
            all_cached_jobs = pd.concat([all_cached_jobs, cached_jobs_df], ignore_index=True)
        all_cached_jobs = all_cached_jobs.drop_duplicates(subset=['title', 'company'])

        if not all_cached_jobs.empty:
            ranked_jobs_df = rank_job_descriptions(cv_text, all_cached_jobs, keywords)
            ranked_jobs = ranked_jobs_df[['title', 'job_url', 'combined_score', 'company', 'date_posted']].head(20).to_dict(orient='records')
            conn.close()
            return render_template('results.html', jobs=ranked_jobs)
        else:
            conn.close()
            return render_template('index.html', warning="No job data available for ranking.")

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)