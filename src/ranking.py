from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .models import sbert_model, tfidf_vectorizer

def calculate_keyword_scores(keywords_list, job_texts):
    """Calculate keyword match score based on overlap of CV and job description."""
    if len(keywords_list) == 0:
        return 0.0
    matching_keywords_list = [list(keyword for keyword in keywords_list if keyword in job_text) for job_text in job_texts]
    keyword_scores = [len(matching_keywords) / len(keywords_list) for matching_keywords in matching_keywords_list]
    return keyword_scores


def calculate_tfidf_scores(cv_text, job_texts):
    """Calculate TF-IDF match score based on overlap of CV and job description."""
    documents = [cv_text] + job_texts
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()


def calculate_similarity_scores(cv_text, job_texts, keywords_list):
    """Calculate similarity scores using TF-IDF, SBERT, and keyword matching."""
    tfidf_scores = [0] * len(job_texts)
    sbert_scores = [0] * len(job_texts)
    if cv_text:
        tfidf_scores = calculate_tfidf_scores(cv_text, job_texts)
        embeddings = sbert_model.encode([cv_text] + job_texts, convert_to_tensor=True).cpu().numpy()
        sbert_scores = cosine_similarity(embeddings[0:1], embeddings[1:]).flatten()
    keyword_scores = calculate_keyword_scores(keywords_list, job_texts)
    return tfidf_scores, sbert_scores, keyword_scores


def rank_job_descriptions(cv_text, jobs_df, keywords):
    """Rank job descriptions based on combined similarity, keyword, and TF-IDF scores."""
    tfidf_scores, sbert_scores, keyword_scores = calculate_similarity_scores(cv_text, jobs_df['description'].tolist(), keywords)

    jobs_df['tfidf_score'] = tfidf_scores
    jobs_df['sbert_similarity'] = sbert_scores
    jobs_df['keyword_score'] = keyword_scores

    # Normalize scores and calculate combined score
    for score in ['sbert_similarity', 'keyword_score', 'tfidf_score']:
        score_range = (jobs_df[score].max() - jobs_df[score].min())
        if score_range == 0:
            jobs_df[score] = 0
        else:
            jobs_df[score] = (jobs_df[score] - jobs_df[score].min()) / score_range

    jobs_df['combined_score'] = jobs_df.apply(
        lambda row: np.mean(sorted([row['sbert_similarity'], row['keyword_score'], row['tfidf_score']], reverse=True)[:2]), axis=1
    )

    return jobs_df.sort_values(by='combined_score', ascending=False)