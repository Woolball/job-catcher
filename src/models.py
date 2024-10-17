import torch
from sentence_transformers import SentenceTransformer, SimilarityFunction
from sklearn.feature_extraction.text import TfidfVectorizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
#sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2').to(device)
sbert_model = SentenceTransformer('all-MiniLM-L12-v2').to(device)
#sbert_model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1').to(device)
sbert_model.similarity_fn_name = SimilarityFunction.COSINE  # COSINE, DOT_PRODUCT, EUCLIDEAN, MANHATTAN

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, sublinear_tf=True)