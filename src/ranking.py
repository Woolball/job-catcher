from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .models import semantic_model, tfidf_vectorizer
import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from .utils import detect_language, remove_stopwords

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


def calculate_semantic_scores(cv_text, job_texts):
    """Calculate SBERT similarity scores between a CV and multiple job descriptions."""
    if not cv_text or not job_texts:
        return [0.0] * len(job_texts)
    texts = [cv_text] + job_texts  # Prepare all texts together for batch processing
    return semantic_model.calculate_semantic_scores(cv_text, job_texts)


# Extract unique elite keywords based on term frequency and entropy
def calculate_elite_scores(cv_text, job_texts, top_n=30):
    # Preprocess documents
    if not cv_text or not job_texts or len(job_texts) < 2:
        return [0.0] * len(job_texts)

    documents = [cv_text] + job_texts

    # Vectorize documents using TF-IDF with unigrams and bigrams
    vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    # Calculate term frequencies
    term_frequencies = np.asarray(tfidf_matrix.sum(axis=0)).flatten()

    # Calculate entropy for each term
    # Vectorized entropy calculation using NumPy
    prob_dist_matrix = tfidf_matrix.toarray() / np.sum(tfidf_matrix.toarray(), axis=0)
    entropy_per_term = -np.sum(prob_dist_matrix * np.log2(prob_dist_matrix + 1e-9), axis=0)

    # Combine term frequency and entropy to filter elite keywords
    combined_ranking = (term_frequencies * (1 / (entropy_per_term + 1e-9)))
    elite_keywords_idx = np.argsort(combined_ranking)[-top_n:]
    elite_keywords = [feature_names[idx] for idx in elite_keywords_idx]

    # Reduce each document to its elite keywords
    elite_keywords_set = set(elite_keywords)
    reduced_documents = [
        ' '.join([word for word in doc.split() if word in elite_keywords_set])
        for doc in documents
    ]

    # Calculate SBERT-based semantic similarity
    elite_scores = calculate_semantic_scores(reduced_documents[0], reduced_documents[1:])
    return elite_scores


def rank_job_descriptions(jobs_df, cv_text, keywords):
    """Rank job descriptions based on combined similarity, keyword, and TF-IDF scores."""
    if jobs_df.empty or 'description' not in jobs_df.columns:
        logger.warning("Job DataFrame is empty or missing descriptions. Skipping ranking.")
        return jobs_df  # Return the empty DataFrame

    job_descriptions_list = jobs_df['description'].tolist()
    lang = detect_language(cv_text)
    job_descriptions_list_stopremoved = remove_stopwords(job_descriptions_list, lang)
    jobs_df['tfidf_score'] = calculate_tfidf_scores(cv_text, job_descriptions_list_stopremoved)
    jobs_df['sbert_similarity'] = calculate_semantic_scores(cv_text, job_descriptions_list)
    jobs_df['keyword_score'] = calculate_keyword_scores(keywords, job_descriptions_list)
    jobs_df['elite_score'] = calculate_elite_scores(cv_text, job_descriptions_list_stopremoved, 100)

    '''
    # Normalize scores and calculate combined score
    for score in ['sbert_similarity', 'keyword_score', 'tfidf_score']:
        score_range = (jobs_df[score].max() - jobs_df[score].min())
        if score_range == 0:
            jobs_df[score] = 0.0
        else:
            jobs_df[score] = (jobs_df[score] - jobs_df[score].min()) / score_range
    '''

    # Historical score max values
    tfidf_score_max_historical = 0.12664288035741172
    sbert_score_max_historical = 0.67
    keyword_score_max_historical = 0.5833333333333334
    elite_score_max_historical = 0.8806282
    print('Max scores:', jobs_df['tfidf_score'].max(), jobs_df['sbert_similarity'].max(), jobs_df['keyword_score'].max(), jobs_df['elite_score'].max())

    # Calculate max values based on historical and current data
    tfidf_score_max = max(tfidf_score_max_historical, jobs_df['tfidf_score'].max())
    sbert_score_max = max(sbert_score_max_historical, jobs_df['sbert_similarity'].max())
    keyword_score_max = max(keyword_score_max_historical, jobs_df['keyword_score'].max())
    elite_score_max = max(elite_score_max_historical, jobs_df['elite_score'].max())

    # Normalize scores
    jobs_df['tfidf_score'] = jobs_df['tfidf_score'] / tfidf_score_max
    jobs_df['sbert_similarity'] = jobs_df['sbert_similarity'] / sbert_score_max
    jobs_df['keyword_score'] = jobs_df['keyword_score'] / keyword_score_max
    jobs_df['elite_score'] = jobs_df['elite_score'] / elite_score_max

    # Calculate combined score (average of the top 2 highest scores)
    jobs_df['combined_score'] = jobs_df.apply(
        lambda row: np.mean(
            sorted([row['sbert_similarity'], row['keyword_score'], row['tfidf_score'], row['elite_score']], reverse=True)[:3]),
        axis=1
    )

    # Sort by tier first, then by combined score within each tier
    jobs_df = jobs_df.sort_values(by=['combined_score'], ascending=False)

    # Define tier bins and corresponding labels
    bins = [0, 0.4, 0.6, 0.8, 1]
    labels = ['Irrelevant', 'Low', 'Moderate', 'High']

    # Create the 'tier' column using pd.cut
    jobs_df['tier'] = pd.cut(jobs_df['combined_score'], bins=bins, labels=labels, include_lowest=True)

    return jobs_df
