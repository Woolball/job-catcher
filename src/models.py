import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
#model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
#model = SentenceTransformer('paraphrase-MiniLM-L6-v2').to(device)
#model = SentenceTransformer('paraphrase-MiniLM-L12-v2').to(device)
#model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2').to(device)
sbert_model = SentenceTransformer('all-MiniLM-L12-v2').to(device)
tfidf_vectorizer = TfidfVectorizer()