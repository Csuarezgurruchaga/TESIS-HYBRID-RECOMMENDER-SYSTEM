import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import pdist


class MemoryCollaborativeFilter:
    def __init__(self, min_overlap=5, n_neighbors=35, n_recommendations=10, 
                 userId_col_name="userId", movieId_col_name='movieId',
                 rating_col_name="rating"):
        """
        Initialize the MemoryCollaborativeFilter object with specified parameters.

        Parameters:
        - min_overlap (int): Minimum number of overlapping users required for similarity calculation.
        - n_neighbors (int): Number of neighbors to consider in collaborative filtering.
        - n_recommendations (int): Number of top recommendations to generate.
        """
        self.min_overlap = min_overlap
        self.n_neighbors = n_neighbors
        self.n_recommendations = n_recommendations
        self.userId_col_name = userId_col_name
        self.movieId_col_name = movieId_col_name
        self.rating_col_name = rating_col_name
        self.items_similarities = None
        self.user_item_matrix = None


    def adjusted_cosine_similarity(self, itemA, itemB):
        """
        Calculate the adjusted cosine similarity of the items for a user-item matrix.
        """
        
        mu_users = self.user_item_matrix.mean(axis=1)

        # Extract ratings for the specified items
        result = self.user_item_matrix.loc[:, [itemA, itemB]]

        # Normalize ratings by subtracting mean for each user
        normalized = result.sub(mu_users, axis=0)

        # Filter out rows with NaN values
        non_nan_mask = ~normalized.isnull().any(axis=1)
        normalized_2 = normalized[non_nan_mask]

        if normalized_2.shape[0] > self.min_overlap:
            # Calculate cosine similarity for the normalized ratings
            items_similarity_scalar = 1 - pdist(normalized_2.T, 'cosine')
            return items_similarity_scalar[0]

        # Return None if the minimum overlap condition is not met
        return None

    def fit(self, data):
        """
        Computes item similarity based on the adjusted cosine similarity metric.
        """
        
        assert isinstance(data, pd.DataFrame), "user_item_matrix should be a Pandas DataFrame."
        
        # Transform data to UxI matrix
        self.user_item_matrix = pd.pivot_table(data,
                                index   = self.userId_col_name,
                                columns = self.movieId_col_name,
                                values  = self.rating_col_name)
        
        idx_items = list(self.user_item_matrix.columns)
        similarity_dict = {}
        
        for item1 in tqdm(idx_items, desc="Computing Similarities", unit="item"):
            similarity_dict[item1] = {}
            for item2 in idx_items:
                # the customer purchased both items?
                if np.any(np.logical_and(self.user_item_matrix.loc[:, item1], self.user_item_matrix.loc[:, item2])): 
                    
                    # calculate the adjusted cosine similarity between the two items
                    similarity = self.adjusted_cosine_similarity(item1, item2) 
                    
                    # we consider only the positive values of similarity
                    if similarity is not None and similarity >= 0:
                        similarity_dict[item1][item2] = similarity
                        
        #sort the dict with the values of similarity
        sorted_sim_dict = {k: {inner_key: inner_value for inner_key, inner_value in sorted(v.items(), key=lambda item: item[1], reverse=True) 
                            if inner_key != k} for k, v in similarity_dict.items()}

        self.items_similarities = sorted_sim_dict

    def compute_neighbors(self, u, i):
        """
        Compute the top neighbors of item i for a given user u based on item similarity.
        """
    
        sim_keys = self.items_similarities[i].keys()
        non_nan_mask = None
        try:
            non_nan_mask = self.user_item_matrix.loc[u, :].notna()
        except IndexError:
            print(f"Error: el usuario {u} no se encuentra registrado")
            
        if non_nan_mask is not None:
            non_nan_idx = non_nan_mask[non_nan_mask].index
            j = list(set(sim_keys) & set(list(non_nan_idx)))

            # Create a dictionary with keys from j and values from sim[i]
            sorted_similarities = {k: self.items_similarities[i][k] for k in j}
            # Sort the dictionary based on values in descending order
            sorted_similarities = dict(sorted(sorted_similarities.items(), key=lambda x: x[1], reverse=True))
            # Select the top neighbors values from the sorted dictionary
            neighbors_of_i = dict(list(sorted_similarities.items())[:self.n_neighbors])

            return neighbors_of_i
        
        
    def compute_prediction(self, u, i):
        """
        Compute the predicted rating for a given user and item.
        """
        
        neighbors = self.compute_neighbors(u, i)
        user_ratings_mean = np.mean(self.user_item_matrix, axis=1)
        neighbors_rating = self.user_item_matrix.loc[u, :].dropna()[list(neighbors.keys())] 
        num = sum(np.array(list(neighbors.values())) * np.array(neighbors_rating))
        denom = sum(list(neighbors.values()))
        try:
            hat_rating =  num / denom
            return hat_rating, user_ratings_mean[u]
        
        except ZeroDivisionError:
            print(f"Warning: Cold-Star problem detected: No neighbors found for item {i} to predict its rating")
            return np.nan, user_ratings_mean[u]


    def recommend(self, u, dict_map):
        """
        Generate top n recommendations for a given user based on item collaborative filtering.
        """

        try :
            rating_predictions = {}
            items_to_predict = list(self.user_item_matrix.loc[u, self.user_item_matrix.loc[u, :].isna()].index)
            for i in items_to_predict:
                rating_predictions[i] = self.compute_prediction(u, i)

            all_recommendations = dict(sorted(rating_predictions.items(), key=lambda item: item[1], reverse=True))
            n_recommendations = [k for k, _ in list(all_recommendations.items())[:self.n_recommendations]]
            rec = [dict_map[item] for item in n_recommendations]
            return rec
        
        except KeyError:
            print(f"Warning Cold-Star Problem detected: User {u} is not registered")