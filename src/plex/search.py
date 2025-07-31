import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# try:
#     from sentence_transformers import SentenceTransformer
# except ImportError:
SentenceTransformer = None  # So we can still use TF-IDF mode

_LOGGER = logging.getLogger(__name__)

class BaseTextSearch:
    def __init__(self, fields_with_weights, mode='tfidf', model_name='all-MiniLM-L6-v2'):
        """
        :param fields_with_weights: dict of {field_name: weight}
        :param mode: 'tfidf' or 'embedding'
        :param model_name: model for embedding mode
        """
        self.fields = fields_with_weights
        self.mode = mode
        self.model_name = model_name
        self.vectorizers = {}
        self.embeddings = {}
        self.df = None

        if self.mode == 'embedding':
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers not installed. Run `pip install sentence-transformers`.")
            self.embedder = SentenceTransformer(model_name)

    def fit(self, df: pd.DataFrame):
        """
        Fit the model to a DataFrame with the fields specified in self.fields.
        """
        self.df = df.copy()

        _LOGGER.info(f"Fitting {self.mode} model to {len(self.df)} documents {json.dumps(self.fields, indent=2)}.")

        for field in self.fields:
            self.df[field] = self.df[field].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))

        if self.mode == 'tfidf':
            for field, weight in self.fields.items():
                vec = TfidfVectorizer()
                try:
                    matrix = vec.fit_transform(self.df[field])
                    self.vectorizers[field] = vec
                    self.embeddings[field] = matrix * weight
                except ValueError as e:
                    _LOGGER.warning(f"TF-IDF vectorization failed for field '{field}': {e}")
                    self.vectorizers[field] = None
                    self.embeddings[field] = np.zeros((self.df.shape[0], 1))

        elif self.mode == 'embedding':
            for field, weight in self.fields.items():
                vectors = self.embedder.encode(self.df[field].tolist(), convert_to_tensor=False)
                self.embeddings[field] = np.array(vectors) * weight

    def search(self, query_dict: dict, top_k: int=5) -> pd.DataFrame:
        """
        Perform a weighted search using the given query_dict.
        :param query_dict: dict like {'title': 'alien', 'summary': 'horror'}
        :param top_k: number of results to return
        :return: DataFrame with 'score' column added, sorted descending
        """
        if self.df is None:
            raise ValueError("Model not fitted. Call fit() before search().")
        combined_scores = np.zeros(self.df.shape[0])

        for field, weight in self.fields.items():
            if field in query_dict and query_dict[field]:
                if self.mode == 'tfidf':
                    try:
                        vec = self.vectorizers[field]
                        query_vector = vec.transform([query_dict[field]]) * weight
                        matrix = self.embeddings[field]
                        scores = cosine_similarity(query_vector, matrix)[0] # type: ignore
                    except ValueError as e:
                        _LOGGER.warning(f"TF-IDF search failed for field '{field}': {e}")
                        scores = np.zeros(self.df.shape[0])
                    except AttributeError:
                        _LOGGER.warning(
                            f"TF-IDF vectorizer not fitted for field '{field}'"
                        )
                        scores = np.zeros(self.df.shape[0])
                elif self.mode == 'embedding':
                    query_vec = self.embedder.encode([query_dict[field]], convert_to_tensor=False)[0] * weight
                    matrix = self.embeddings[field]
                    scores = cosine_similarity([query_vec], matrix)[0] # type: ignore
                else:
                    raise ValueError(f"Unknown mode: {self.mode}")
                combined_scores += scores

        result_df = self.df.copy()
        result_df["score"] = combined_scores
        return result_df.sort_values("score", ascending=False).head(top_k)
