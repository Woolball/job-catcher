import torch
from sentence_transformers import SentenceTransformer, SimilarityFunction
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from .utils import detect_language, remove_stopwords

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, sublinear_tf=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
semantic_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
semantic_model = SentenceTransformer(semantic_model_name).to(device)
semantic_model.similarity_fn_name = SimilarityFunction.DOT_PRODUCT  # COSINE, DOT_PRODUCT, EUCLIDEAN, MANHATTAN
tokenizer = semantic_model.tokenizer
max_seq_length = semantic_model.max_seq_length
cls_token_id = torch.tensor([tokenizer.cls_token_id])
sep_token_id = torch.tensor([tokenizer.sep_token_id])

#semantic_model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
#semantic_model_name = 'sentence-transformers/all-MiniLM-L12-v2'
#semantic_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
#semantic_model_name = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'

# Initialize logger
logger = logging.getLogger(__name__)

def filter_jobs_by_keywords(jobs_df, required_keywords, exclude_keywords):
    """
    Filters jobs based on the presence of 'must-have' and 'exclusion' keywords.
    """
    # Check if job descriptions contain all must-have keywords
    if required_keywords:
        jobs_df = jobs_df[jobs_df['description'].apply(lambda desc: all(keyword in desc for keyword in required_keywords))]

    # Exclude job descriptions containing any exclusion keywords
    if exclude_keywords:
        jobs_df = jobs_df[~jobs_df['description'].apply(lambda desc: any(keyword in desc for keyword in exclude_keywords))]

    return jobs_df


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


def calculate_semantic_scores(cv_text, job_texts, embedding_seq_length=None):
    """Calculate semantic similarity scores between a CV and multiple job descriptions."""
    if not cv_text or not job_texts or job_texts == ['']:
        return [0.0] * len(job_texts)

    def tokenize_and_chunk(text):
        """Tokenize text, truncate to embedding_seq_length, and chunk within max sequence length."""
        tokens = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=embedding_seq_length or max_seq_length,
            padding=False
        )
        input_ids, attention_mask = tokens['input_ids'][0], tokens['attention_mask'][0]

        # Define chunk size, accounting for [CLS] and [SEP] tokens
        chunk_size = max_seq_length - 2
        input_id_chunks = [input_ids[i:i + chunk_size] for i in range(0, len(input_ids), chunk_size)]
        attention_mask_chunks = [attention_mask[i:i + chunk_size] for i in range(0, len(attention_mask), chunk_size)]

        # Add [CLS] and [SEP] tokens to each chunk
        input_id_chunks = [torch.cat([cls_token_id, chunk, sep_token_id]) for chunk in input_id_chunks]
        attention_mask_chunks = [torch.cat([torch.tensor([1]), chunk, torch.tensor([1])]) for chunk in attention_mask_chunks]

        return input_id_chunks, attention_mask_chunks

    def get_normalized_average_embedding(text):
        """Calculate normalized mean embedding from text chunks."""
        input_id_chunks, attention_mask_chunks = tokenize_and_chunk(text)
        if not input_id_chunks:
            return np.zeros(semantic_model.get_sentence_embedding_dimension())

        # Accumulate embeddings to calculate the mean
        embedding_sum = 0
        for input_ids, attention_mask in zip(input_id_chunks, attention_mask_chunks):
            inputs = {'input_ids': input_ids.unsqueeze(0), 'attention_mask': attention_mask.unsqueeze(0)}
            with torch.no_grad():
                model_output = semantic_model(inputs, output_hidden_states=False, output_attentions=False)
                chunk_embedding = model_output['sentence_embedding']
            embedding_sum += chunk_embedding.cpu().numpy().squeeze()

        # Calculate mean embedding and normalize
        mean_embedding = embedding_sum / len(input_id_chunks)
        norm = np.linalg.norm(mean_embedding)
        return mean_embedding / norm if norm != 0 else mean_embedding

    # Obtain normalized embeddings for CV and job descriptions
    cv_embedding = get_normalized_average_embedding(cv_text)
    job_embeddings = [get_normalized_average_embedding(job_text) for job_text in job_texts]

    # Compute cosine similarity scores as dot products
    similarity_scores = [np.dot(cv_embedding, job_emb) for job_emb in job_embeddings]
    return similarity_scores


# Extract unique elite keywords based on term frequency and entropy
def calculate_elite_scores(cv_text, job_texts, top_n=30, boost_factor=1.5):
    if not cv_text or not job_texts or len(job_texts) < 2:
        return [0.0] * len(job_texts)

    # Combine all documents for TF-IDF vectorization
    documents = [cv_text] + job_texts

    # Vectorize with TF-IDF
    vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    # Term frequencies for the entire matrix
    term_frequencies = np.asarray(tfidf_matrix.sum(axis=0)).flatten()

    # Extract CV term indices from the TF-IDF matrix
    cv_tfidf_vector = tfidf_matrix[0].toarray().flatten()  # CV text is the first row
    cv_term_indices = cv_tfidf_vector.nonzero()[0]  # Get indices of terms in CV text

    # Apply boost factor to term frequencies of terms present in CV
    term_frequencies[cv_term_indices] *= boost_factor

    # Entropy Calculation
    prob_dist_matrix = tfidf_matrix.toarray() / np.sum(tfidf_matrix.toarray(), axis=0)
    entropy_per_term = -np.sum(prob_dist_matrix * np.log2(prob_dist_matrix + 1e-9), axis=0)

    # Combined Ranking with boosted CV terms
    combined_ranking = term_frequencies * (1 / (entropy_per_term + 1e-9))
    elite_keywords_idx = np.argsort(combined_ranking)[-top_n:]
    elite_keywords = [feature_names[idx] for idx in elite_keywords_idx]

    # Reduce documents to elite keywords
    elite_keywords_set = set(elite_keywords)
    reduced_documents = [
        ' '.join([word for word in doc.split() if word in elite_keywords_set])
        for doc in documents
    ]

    # Calculate Semantic Similarity
    elite_scores = calculate_semantic_scores(reduced_documents[0], reduced_documents[1:])
    return elite_scores


def rank_job_descriptions(jobs_df, cv_text, preferred_keywords, required_keywords=[], exclude_keywords=[]):
    """Rank job descriptions based on combined similarity, keyword, and TF-IDF scores."""
    if jobs_df.empty or 'description' not in jobs_df.columns:
        logger.warning("Job DataFrame is empty or missing descriptions. Skipping ranking.")
        return jobs_df  # Return the empty DataFrame
    if not cv_text:
        logger.warning("CV text is empty or non-existent. Skipping ranking.")
        return jobs_df

    # Filter jobs based on 'must-have' and 'exclusion' keywords
    jobs_df = filter_jobs_by_keywords(jobs_df, required_keywords, exclude_keywords)

    job_descriptions_list = jobs_df['description'].tolist()
    lang = detect_language(cv_text)
    job_descriptions_list_stopremoved = remove_stopwords(job_descriptions_list, lang)
    jobs_df['tfidf_score'] = calculate_tfidf_scores(cv_text, job_descriptions_list_stopremoved)
    jobs_df['sbert_similarity'] = calculate_semantic_scores(cv_text, job_descriptions_list, 508)
    jobs_df['keyword_score'] = calculate_keyword_scores(preferred_keywords, job_descriptions_list)
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
    semantic_score_max_historical = 0.78217864
    keyword_score_max_historical = 0.5833333333333334
    elite_score_max_historical = 0.9056018
    print('Max scores:', jobs_df['tfidf_score'].max(), jobs_df['sbert_similarity'].max(), jobs_df['keyword_score'].max(), jobs_df['elite_score'].max())

    # Calculate max values based on historical and current data
    tfidf_score_max = max(tfidf_score_max_historical, jobs_df['tfidf_score'].max())
    sbert_score_max = max(semantic_score_max_historical, jobs_df['sbert_similarity'].max())
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
    labels = ['Irrelevant', 'Tiny maybe', 'Mid match', 'High match']

    # Create the 'tier' column using pd.cut
    jobs_df['tier'] = pd.cut(jobs_df['combined_score'], bins=bins, labels=labels, include_lowest=True)

    return jobs_df
