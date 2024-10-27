from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from config import Config

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, sublinear_tf=True)


class EmbeddingModel:
    def __init__(self, full_model_name, device="cpu"):
        self.device = device
        self.model_type, self.model_name = full_model_name.split('/', 1)

        # Initialize model and tokenizer based on model type
        if self.model_type == "intfloat":
            self.tokenizer = AutoTokenizer.from_pretrained(full_model_name)
            self.model = AutoModel.from_pretrained(full_model_name).to(device)
        elif self.model_type == "sentence-transformers":
            self.model = SentenceTransformer(self.model_name).to(device)
        elif (self.model_type == "shibing624" or self.model_type == "Mihaiii"):
            self.model = SentenceTransformer(full_model_name).to(device)
        else:
            raise ValueError("Model type not supported")

    def preprocess(self, cv_text, job_texts):
        """Preprocess text based on model type."""
        if self.model_type == "intfloat":
            # Add E5-specific prefixes
            cv_text = f"query: {cv_text}"
            job_texts = [f"passage: {job}" for job in job_texts]
        # For SentenceTransformer, no preprocessing needed
        return cv_text, job_texts

    def encode(self, texts):
        """Encode texts to embeddings based on model type."""
        if self.model_type == "intfloat":
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                embeddings = self.model(**inputs).last_hidden_state[:, 0, :]  # CLS token
            return embeddings.cpu().numpy()
        elif (self.model_type == "sentence-transformers" or self.model_type == "shibing624"
              or self.model_type == "Mihaiii"):
            return self.model.encode(texts, convert_to_numpy=True)

    def calculate_semantic_scores(self, cv_text, job_texts):
        """Calculate similarity scores between CV and job descriptions."""
        cv_text, job_texts = self.preprocess(cv_text, job_texts)
        texts = [cv_text] + job_texts  # Combine for batch encoding
        embeddings = self.encode(texts)

        # Separate CV embedding from job embeddings
        cv_embedding = embeddings[0]  # First embedding is for the CV
        job_embeddings = embeddings[1:]  # The rest are job embeddings

        # Calculate cosine similarity scores
        return cosine_similarity([cv_embedding], job_embeddings).flatten()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
semantic_model = EmbeddingModel(Config.SEMANTIC_MODEL_NAME, device)


