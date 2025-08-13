import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
import ast
from tqdm import tqdm
import gensim
from gensim import corpora
import warnings
warnings.filterwarnings('ignore')

class ContentBasedFilter:
    """
    Content-based recommender system that suggests movies based on LDA topic features.
    Uses cosine similarity between movie topic vectors to generate recommendations.
    """
    def __init__(self):
        self.LDA_movies_data_df = None
        self.movie_features_matrix = None  # 2D NumPy array: each row is a movie's feature vector(weights of the 11 LDA topics derived from movie taglines, genres, and tags)
        self.features_norm = None # Precomputed L2 norms of the feature vectors to accelerate cosine similarity calculations.
        self.is_fitted = False
        self.user_rated_movie_ids_cb = {} 

    def fit(self, user_movie_rating_df, movies_tokenized_df=None):
        """
        Train an LDA model on tokenized movie taglines and generate topic features.
        After training, it also builds a matrix of feature vectors for faster computations.
        Also pre-calculates user-rated movie IDs for faster lookups later. 
        """
        self.user_movie_rating_df = user_movie_rating_df

        if movies_tokenized_df is not None:
            print("Processing tokenized taglines...")
            tokenized_taglines = movies_tokenized_df['tokenized_tagline_genres_tags'].apply(ast.literal_eval).values.tolist()

            print("Creating dictionary...")
            dct_tagline = corpora.Dictionary(tokenized_taglines)

            print("Converting to bag-of-words representation...")
            movies_tagline_corpus = [dct_tagline.doc2bow(text) for text in tokenized_taglines]

            print("Training LDA model with 11 topics...")
            best_lda_tagline_model = gensim.models.ldamodel.LdaModel(
                corpus=movies_tagline_corpus,
                id2word=dct_tagline,
                num_topics=11,
                random_state=100,
                update_every=1,
                chunksize=100,
                passes=10,
                alpha='auto'
            )

            print("\nGenerating topic features for each movie...")
            tagline_topic_features = [
                best_lda_tagline_model.get_document_topics(movie_corpus, minimum_probability=0.0)
                for movie_corpus in movies_tagline_corpus
            ]

            self.LDA_movies_data_df = movies_tokenized_df[['movieId', 'title']].copy()
            self.LDA_movies_data_df['LDA_topic_features'] = tagline_topic_features

            # Build a NumPy matrix from the topic features.
            self.LDA_movies_data_df['feature_vector'] = self.LDA_movies_data_df['LDA_topic_features'].apply(
                lambda feats: np.array([t[1] for t in feats])
            )
            self.movie_features_matrix = np.vstack(self.LDA_movies_data_df['feature_vector'].values)
            self.features_norm = np.linalg.norm(self.movie_features_matrix, axis=1)

            print("Training completed. Ready to generate recommendations.")

        # Pre-calculate user-rated movie IDs for faster lookups # NEW: Implementation for optimization
        print("Pre-calculating user-rated movie IDs for ContentBasedFilter...") # NEW: Print statement
        self.user_rated_movie_ids_cb = {}
        for user_id in tqdm(user_movie_rating_df['userId'].unique(), desc="Processing users for CB rated movies"): # NEW: Progress bar for user processing
            rated_movies = user_movie_rating_df[user_movie_rating_df['userId'] == user_id]['movieId'].unique()
            self.user_rated_movie_ids_cb[user_id] = set(rated_movies) # Store as a set for faster lookups

        self.is_fitted = True

    def fetch_movies_features(self, movies_info_path='../data/movie_lens_small/content_based_LDA_data.csv'):
        """
        Load movie features from CSV file containing LDA topic vectors.
        Also builds the matrix of feature vectors for fast vectorized similarity computation.
        """
        print(f'Fetching movies features from the PATH: {movies_info_path}\n')
        self.LDA_movies_data_df = pd.read_csv(movies_info_path)
        self.LDA_movies_data_df['LDA_topic_features'] = self.LDA_movies_data_df['LDA_topic_features'].apply(ast.literal_eval)

        # Create a feature vector column from the LDA topic features.
        self.LDA_movies_data_df['feature_vector'] = self.LDA_movies_data_df['LDA_topic_features'].apply(
            lambda feats: np.array([t[1] for t in feats])
        )
        self.movie_features_matrix = np.vstack(self.LDA_movies_data_df['feature_vector'].values)
        self.features_norm = np.linalg.norm(self.movie_features_matrix, axis=1)

    def compute_prediction(self, u, i):
        """
        Predict the rating for a target movie based on the user's rated movies using
        a similarity-weighted average.
        """
        if self.LDA_movies_data_df is None:
            self.fetch_movies_features()

        user_movies = self.user_movie_rating_df[
            (self.user_movie_rating_df['userId'] == u) &
            (self.user_movie_rating_df['rating'].notna())
        ]

        if user_movies.empty:
            raise ValueError("No valid rated movies found for similarity calculation.")

        rated_features = []
        ratings = []
        for _, row in user_movies.iterrows():
            movie_row = self.LDA_movies_data_df[self.LDA_movies_data_df['movieId'] == row['movieId']]
            if not movie_row.empty:
                rated_features.append(movie_row.iloc[0]['feature_vector'])
                ratings.append(row['rating'])
        rated_features = np.vstack(rated_features)

        target_row = self.LDA_movies_data_df[self.LDA_movies_data_df['movieId'] == i]
        if target_row.empty:
            raise ValueError("Target movie not found in LDA_movies_data file.")
        target_vector = target_row.iloc[0]['feature_vector']

        # Vectorized cosine similarity calculation:
        norm_target = np.linalg.norm(target_vector)
        rated_norms = np.linalg.norm(rated_features, axis=1)
        dot_products = rated_features.dot(target_vector)
        similarities = dot_products / (rated_norms * norm_target + 1e-10)

        numerator = np.sum(similarities * np.array(ratings))
        denominator = np.sum(similarities)
        if denominator == 0:
            raise ValueError("Check: sum of similarities equal to 0")
        return numerator / denominator

    def recommend(self, u, n_recommendations=10, rating_threshold=4.0, dict_map=None):
        """
        Generate personalized recommendations using vectorized computation of cosine similarity.
        This version avoids Python-level loops by computing similarities in bulk.
        """
        if self.LDA_movies_data_df is None:
            self.fetch_movies_features()

        # Identify movies the user liked.
        user_movies_ids = self.user_movie_rating_df[
            (self.user_movie_rating_df['userId'] == u) &
            (self.user_movie_rating_df['rating'] >= rating_threshold) &
            (self.user_movie_rating_df['movieId'].isin(self.LDA_movies_data_df['movieId']))
        ]['movieId'].unique()

        if len(user_movies_ids) == 0:
            return []  # No recommendations if no liked movies

        # Get indices of liked movies.
        liked_mask = self.LDA_movies_data_df['movieId'].isin(user_movies_ids)
        liked_indices = self.LDA_movies_data_df.index[liked_mask].tolist()
        liked_features = self.movie_features_matrix[liked_indices, :]
        liked_norms = self.features_norm[liked_indices]

        # Candidate movies are those not already rated highly by the user.
        candidate_mask = ~self.LDA_movies_data_df['movieId'].isin(user_movies_ids)
        candidate_indices = self.LDA_movies_data_df.index[candidate_mask].tolist()
        candidate_features = self.movie_features_matrix[candidate_indices, :]
        candidate_norms = self.features_norm[candidate_indices]

        # Compute dot products between each candidate and each liked movie.
        dot_products = candidate_features.dot(liked_features.T)  # shape: (n_candidates, n_liked)
        # Divide by norms (broadcasting over candidates and liked movies).
        denom = candidate_norms[:, None] * liked_norms[None, :] + 1e-10
        similarities = dot_products / denom  # shape: (n_candidates, n_liked)

        # Compute a robust (median) similarity for each candidate movie.
        median_similarities = np.median(similarities, axis=1)

        # Sort candidates by similarity in descending order.
        sorted_idx = np.argsort(median_similarities)[::-1]
        top_candidate_indices = [candidate_indices[i] for i in sorted_idx[:n_recommendations]]
        recommended_movies = self.LDA_movies_data_df.iloc[top_candidate_indices]['movieId'].tolist()

        return recommended_movies