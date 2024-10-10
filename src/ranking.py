from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .models import sbert_model, tfidf_vectorizer
import logging

# Initialize logger
logger = logging.getLogger(__name__)

def calculate_keyword_scores(keywords_list, job_texts):
    """Calculate keyword match score based on overlap of CV and job description."""
    if not keywords_list or not job_texts:
        return [0.0] * len(job_texts)
    matching_keywords_list = [list(keyword for keyword in keywords_list if keyword in job_text) for job_text in job_texts]
    keyword_scores = [len(matching_keywords) / len(keywords_list) for matching_keywords in matching_keywords_list]
    return keyword_scores


def calculate_tfidf_scores(cv_text, job_texts):
    """Calculate TF-IDF match score based on overlap of CV and job description."""
    if not cv_text or not job_texts:
        return [0.0] * len(job_texts)
    documents = [cv_text] + job_texts
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()


def calculate_sbert_scores(cv_text, job_texts):
    if not cv_text or not job_texts:
        return [0.0] * len(job_texts)
    embeddings = sbert_model.encode([cv_text] + job_texts, convert_to_tensor=True).cpu().numpy()
    sbert_scores = cosine_similarity(embeddings[0:1], embeddings[1:]).flatten()
    del embeddings  # Manually delete the embeddings tensor to free memory
    return sbert_scores


def rank_job_descriptions(jobs_df, cv_text, keywords):
    """Rank job descriptions based on combined similarity, keyword, and TF-IDF scores."""
    if jobs_df.empty or 'description' not in jobs_df.columns:
        logger.warning("Job DataFrame is empty or missing descriptions. Skipping ranking.")
        return jobs_df  # Return the empty DataFrame

    job_descriptions_list = jobs_df['description'].tolist()
    jobs_df['tfidf_score'] = calculate_tfidf_scores(cv_text, job_descriptions_list)
    jobs_df['sbert_similarity'] = calculate_sbert_scores(cv_text, job_descriptions_list)
    jobs_df['keyword_score'] = calculate_keyword_scores(keywords, job_descriptions_list)

    # Normalize scores and calculate combined score
    for score in ['sbert_similarity', 'keyword_score', 'tfidf_score']:
        score_range = (jobs_df[score].max() - jobs_df[score].min())
        if score_range == 0:
            jobs_df[score] = 0.0
        else:
            jobs_df[score] = (jobs_df[score] - jobs_df[score].min()) / score_range

    jobs_df['combined_score'] = jobs_df.apply(
        lambda row: np.mean(sorted([row['sbert_similarity'], row['keyword_score'], row['tfidf_score']], reverse=True)[:2]), axis=1
    )

    return jobs_df.sort_values(by='combined_score', ascending=False)