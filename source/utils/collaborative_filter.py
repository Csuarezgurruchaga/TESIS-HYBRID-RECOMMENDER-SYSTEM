import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from scipy.spatial.distance import pdist
from scipy.optimize import minimize_scalar
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from .metrics_and_utils import mean_absolute_error


class MemoryCollaborativeFilter:
    def __init__(self, min_overlap=5, n_neighbours=40, userId_col_name="userId", 
                 movieId_col_name='movieId', rating_col_name="rating"):
        """
        Initialize the MemoryCollaborativeFilter object with specified parameters.

        Parameters:
        - min_overlap (int): Minimum number of overlapping users required for similarity calculation.
        - n_neighbours (int): Number of neighbors to consider in collaborative filtering.
        - n_recommendations (int): Number of top recommendations to generate.
        """
        self.min_overlap = min_overlap
        self.n_neighbours = n_neighbours
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
        
        for item1 in tqdm(idx_items, desc="Computing Similarities", unit=" item", dynamic_ncols=True, leave=True):
            similarity_dict[item1] = {}
            for item2 in idx_items:
                # the user interacted both items?
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
        Compute the top neighbours of item i for a given user u based on item similarity.
        """
        # if there aren't any neighbors, return an empty key.
        sim_keys = self.items_similarities.get(i, {}).keys()
        non_nan_mask = None
        try:
            non_nan_mask = self.user_item_matrix.loc[u, :].notna()
        except IndexError:
            print(f"Error: User {u} is not registered")
            
        if non_nan_mask is not None:
            non_nan_idx = non_nan_mask[non_nan_mask].index
            j = list(set(sim_keys) & set(list(non_nan_idx)))

            # Create a dictionary with keys from j and values from sim[i]
            sorted_similarities = {k: self.items_similarities[i][k] for k in j}
            # Sort the dictionary based on values in descending order
            sorted_similarities = dict(sorted(sorted_similarities.items(), key=lambda x: x[1], reverse=True))
            # Select the top neighbours values from the sorted dictionary
            neighbours_of_i = dict(list(sorted_similarities.items())[:self.n_neighbours])
            return neighbours_of_i

    # def compute_neighbors(self, u, i):
    #     """
    #     Compute the top neighbours of item i for a given user u based on item similarity.
    #     """
    
    #     # if there aren't any neighbors, return an empty key.
    #     sim_keys = self.items_similarities.get(i, {}).keys()
    #     non_nan_mask = None
    #     try:
    #         non_nan_mask = self.user_item_matrix.loc[u, :].notna()
    #     except IndexError:
    #         print(f"Error: User {u} is not registered")
    #         return {}

    #     if non_nan_mask is not None:
    #         non_nan_idx = non_nan_mask[non_nan_mask].index
    #         j = list(set(sim_keys) & set(list(non_nan_idx)))

    #         # Create a dictionary with keys from j and values from sim[i]
    #         sorted_similarities = {k: self.items_similarities[i][k] for k in j}
    #         # Sort the dictionary based on values in descending order
    #         sorted_similarities = dict(sorted(sorted_similarities.items(), key=lambda x: x[1], reverse=True))
    #         # Select the top neighbours values from the sorted dictionary
    #         neighbours_of_i = dict(list(sorted_similarities.items())[:self.n_neighbours])
    #         return neighbours_of_i
    def compute_neighbors(self, u, i):
        """
        Compute the top neighbours of item i for a given user u based on item similarity.
        """
        # if there aren't any neighbors, return an empty key.
        sim_keys = self.items_similarities.get(i, {}).keys()
        non_nan_mask = None
        try:
            non_nan_mask = self.user_item_matrix.loc[u, :].notna()
        except (KeyError, IndexError):  # Cambiar IndexError por (KeyError, IndexError)
            print(f"Error: User {u} is not registered")
            return {}  # AGREGAR RETURN EXPLÍCITO

        if non_nan_mask is not None:
            non_nan_idx = non_nan_mask[non_nan_mask].index
            j = list(set(sim_keys) & set(list(non_nan_idx)))

            # Create a dictionary with keys from j and values from sim[i]
            sorted_similarities = {k: self.items_similarities[i][k] for k in j}
            # Sort the dictionary based on values in descending order
            sorted_similarities = dict(sorted(sorted_similarities.items(), key=lambda x: x[1], reverse=True))
            # Select the top neighbours values from the sorted dictionary
            neighbours_of_i = dict(list(sorted_similarities.items())[:self.n_neighbours])
            return neighbours_of_i
        
        # Si non_nan_mask es None, retornar diccionario vacío
        return {}

        
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
            hat_rating = num / denom
            return hat_rating#, user_ratings_mean[u]
        
        except ZeroDivisionError:
            print(f"Warning: Cold-Star problem detected: No neighbours found for item {i} to predict its rating")
            return np.nan#, user_ratings_mean[u]
    
    def recommend(self, u, dict_map=None, n_recommendations=10):
        """
        Generate top n recommendations for a given user based on collaborative filtering.
        
        Parameters:
            u (int): User ID.
            dict_map (dict): Dictionary mapping internal item IDs to external item IDs (e.g., movie titles).
            n_recommendations (int): Number of top recommendations to generate.
        
        Returns:
            list: A list of top n recommended item IDs (or names if dict_map is provided).
        """
        try:
            rating_predictions = {}
            # Identify items the user hasn't rated yet
            items_to_predict = list(self.user_item_matrix.loc[u, self.user_item_matrix.loc[u, :].isna()].index)
            
            # Compute predicted ratings for each unseen item
            for i in items_to_predict:
                rating_predictions[i] = self.compute_prediction(u, i)
            
            # Sort items by predicted rating (highest first)
            all_recommendations = dict(sorted(rating_predictions.items(), key=lambda item: item[1], reverse=True))
            recommendations_ids = [k for k, _ in list(all_recommendations.items())[:n_recommendations]]
            
            # If a mapping is provided, convert internal IDs to external names/IDs
            if dict_map is not None:
                rec = [dict_map[item] for item in recommendations_ids]
            else:
                rec = recommendations_ids

            return rec

        except KeyError:
            print(f"Warning: User {u} is not registered (Cold-Star Problem)")
            return []

class TWMemoryCollaborativeFilter:
    """
    Parameters:
    - min_overlap (int): Minimum number of overlapping users required for similarity calculation.
    - n_neighbours (int): Number of neighbors to consider in collaborative filtering. A number of neighbors between 20 to 50 is most often recommended.
    - n_recommendations (int): Number of top recommendations to generate.
    - max_n_clusters (int): Maximum number of clusters to consider. A larger number may result in overfitting.
    - refine_n_clusters (bool): If True, refines the number of clusters based on silhouette scores.
    - userId_col_name (str): Name of the column containing user IDs in the input data.
    - movieId_col_name (str): Name of the column containing movie IDs in the input data.
    - rating_col_name (str): Name of the column containing ratings in the input data.
    - time_col_name (str): Name of the column containing timestamps in the input data.
    - verbose (bool): If True, provides detailed information about the process.
    - random_state (int): Used for deterministic results.

    Attributes:
    - min_overlap (int): Minimum number of overlapping users required for similarity calculation.
    - n_neighbours (int): Number of neighbors to consider in collaborative filtering.
    - userId_col_name (str): Name of the column containing user IDs in the input data.
    - movieId_col_name (str): Name of the column containing movie IDs in the input data.
    - rating_col_name (str): Name of the column containing ratings in the input data.
    - time_col_name (str): Name of the column containing timestamps in the input data.
    - max_n_clusters (int): Maximum number of clusters to consider. A larger number may result in overfitting.
    - refine_n_clusters (bool): If True, refines the number of clusters based on silhouette scores.
    - rescale_parameter (float): Rescaling parameter for time-weighted calculations.
    - items_similarities (dict): Dictionary containing item similarities based on adjusted cosine similarity.
    - user_item_matrix (pd.DataFrame): User-item matrix representation of the input data.
    - data (pd.DataFrame): Original input data used to train the collaborative filter.
    - train_LOO (pd.DataFrame): Leave-One-Out train set with NaN ratings for validation movies.
    - sim_count_0_user_item(dict): userId and movieId for the items that cannot compute the prediction of T0
    """
    def __init__(self, min_overlap=5, n_neighbours=40, max_n_clusters=3, 
                 refine_n_clusters=True, userId_col_name="userId", movieId_col_name='movieId', 
                 rating_col_name="rating", time_col_name='timestamp', verbose=False, random_state=1203):
        self.min_overlap = min_overlap
        self.n_neighbours = n_neighbours
        self.userId_col_name = userId_col_name
        self.movieId_col_name = movieId_col_name
        self.rating_col_name = rating_col_name
        self.time_col_name = time_col_name
        self.max_n_clusters = max_n_clusters
        self.refine_n_clusters = refine_n_clusters
        self.verbose = verbose
        self.random_state = random_state
        self.is_fitted = False  # Added to track if the model is fitted

        self.items_similarities = None
        self.user_item_matrix = None
        self.data = None
        self.train_LOO = None
        self.y_true_LOO_by_cluster = None
        self.test_items_LOO = None
        self.train_LOO_scenario = None
        self.test_items_LOO_scenario_sim_count_greater_1 = None
        self.T0_by_user_cluster_df = None
        self.user_cluster_T0_map_dict = None
        self.train_T0 = None
        self.sim_count_0_user_item = None
        
        # Cache for user neighbors and time weights
        self.cached_neighbors = {}
        self.cached_time_weights = {}

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
        Fits the TWMemoryCollaborativeFilter model to the provided user-item interaction data considering the time.

        Parameters:
            data (pd.DataFrame): Input DataFrame containing user-item interactions with ratings and timestamps.

        Updates:
            - Computes item similarity based on the adjusted cosine similarity metric.
            - Performs leave-one-out (LOO) splitting for training and validation sets, grouping items into clusters.
            - Classifies scenarios for each user and cluster of items, based on the count of similar items.
            - Optimizes and computes T0 values for each user and cluster pair using LOO predictions.
            - Builds the training set with T0 values and updates the instance attributes.

        Notes:
            - The adjusted cosine similarity is computed using the user-item matrix derived from the input data.
            - LOO splitting is performed, and scenarios are classified to handle different cases.
            - MAE is the metric used to find the better T0 for each user and each cluster of items.
            - T0 values are optimized using the leave-one-out (LOO) prediction and stored for future use.
            - The training set is constructed with T0 values, considering different scenarios for predictions.
        """
    
        assert isinstance(data, pd.DataFrame), "user_item_matrix should be a Pandas DataFrame."
        
        self.data = data

        # CORRECTED: scaling timestamps for each user in days based on their LAST interaction with the system.
        # This gives MORE weight to RECENT interactions (better for recommendation systems)
        self.data[self.time_col_name] = pd.to_datetime(self.data[self.time_col_name], unit='s')
        self.data['last_timestamp'] = self.data.groupby(self.userId_col_name)[self.time_col_name].transform('max')
        self.data[self.time_col_name] = (self.data['last_timestamp'] - self.data[self.time_col_name]).dt.total_seconds() / (24 * 3600)
        self.data = self.data.drop(columns=['last_timestamp'])

        
        # Transform data to UxI matrix
        self.user_item_matrix = pd.pivot_table(self.data,
                                index   = self.userId_col_name,
                                columns = self.movieId_col_name,
                                values  = self.rating_col_name)
        
        # Store user means for later use
        self.user_means = self.user_item_matrix.mean(axis=1)
        
        idx_items = list(self.user_item_matrix.columns)
        similarity_dict = {}
        
        for item1 in tqdm(idx_items, desc="Computing Similarities", unit="item", dynamic_ncols=True, leave=True):
            similarity_dict[item1] = {}
            for item2 in idx_items:
                # the user interacted both items?
                if np.any(np.logical_and(self.user_item_matrix.loc[:, item1].notna(), self.user_item_matrix.loc[:, item2].notna())): 
                    
                    # calculate the adjusted cosine similarity between the two items
                    similarity = self.adjusted_cosine_similarity(item1, item2) 
                    
                    # we consider only the positive values of similarity
                    if similarity is not None and similarity >= 0:
                        similarity_dict[item1][item2] = similarity
                        
        #sort the dict with the values of similarity
        sorted_sim_dict = {k: {inner_key: inner_value for inner_key, inner_value in sorted(v.items(), key=lambda item: item[1], reverse=True) 
                            if inner_key != k} for k, v in similarity_dict.items()}
        self.items_similarities = sorted_sim_dict
        
        # for each user split the items in clusters using the ratings.
        self.LOO_split_by_cluster()
        
        # build different scenarios for each item's cluster
        self.classify_scenario(self.train_LOO, self.test_items_LOO)
        
        # considering the scenario of that item and that user find the T0 value.
        self.get_T0_by_user_cluster()
        
        # build the final dataset UxIxT0
        self.build_trainT0()
        
        # Create a pivot table from train_T0 for faster lookups
        self.user_item_matrix_final = pd.pivot_table(self.train_T0,
                                index   = self.userId_col_name,
                                columns = self.movieId_col_name,
                                values  = self.rating_col_name)
        
        # Mark the model as fitted
        self.is_fitted = True
        
        # Clear the caches
        self.cached_neighbors = {}
        self.cached_time_weights = {}
        
    def clusterize_user(self, u, to_clusterize):
        """
        Clusterizes items for a given user based on ratings using KMeans clustering.

        Parameters:
            u (int): User ID.
            to_clusterize (pd.DataFrame): DataFrame containing 'rating' and 'movieId' columns to be clustered.

        Returns:
            pd.DataFrame: Clusterized DataFrame with columns 'movieId', 'rating', and 'cluster_label'.
        """
        
        if self.refine_n_clusters:
            n_clusters_range = range(2, self.max_n_clusters+1)
            # Dictionary to store silhouette scores for each number of clusters
            sil_scores = {}
            for n_clusters in n_clusters_range:
                # Create and fit KMeans model
                kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init = 'auto')
                cluster_labels = kmeans.fit_predict(np.array(to_clusterize[self.rating_col_name]).reshape(-1, 1))
                
                # Calculate Silhouette Score
                if len(set(cluster_labels)) > 1:
                    sil_score = silhouette_score(np.array(to_clusterize[self.rating_col_name]).reshape(-1, 1), cluster_labels)
                    sil_scores[n_clusters] = sil_score

                    # Find the best number of clusters
                    best_n_clusters = max(sil_scores, key=sil_scores.get)
            
            if len(set(cluster_labels)) == 1:
                # the user rated similarly all the movies that he saw.
                best_n_clusters = 1
                if self.verbose:
                    print(f"n_clusters for user {u} is 1, the user rated similarly all the movies that he saw.")
                    
            # Create and fit KMeans model with the best number of clusters
            best_kmeans = KMeans(n_clusters=best_n_clusters, random_state=self.random_state, n_init = 'auto')
            best_cluster_labels = best_kmeans.fit_predict(np.array(to_clusterize[self.rating_col_name]).reshape(-1, 1))

            clusterized_df = pd.DataFrame({self.movieId_col_name: to_clusterize[self.movieId_col_name].values,
                                            self.rating_col_name: to_clusterize[self.rating_col_name].values,  
                                            'cluster_label': best_cluster_labels})
            return clusterized_df    
        
        # if refine_n_clusters is False, the computation cost its reduced and the number of clusters is
        # set as max_n_clusters
        best_kmeans = KMeans(n_clusters=self.max_n_clusters, random_state=self.random_state, n_init = 'auto')
        best_cluster_labels = best_kmeans.fit_predict(np.array(to_clusterize['rating']).reshape(-1, 1))

        clusterized_df = pd.DataFrame({'movieId': to_clusterize['movieId'].values,
                                        'rating': to_clusterize['rating'].values,  
                                        'cluster_label': best_cluster_labels})
        return clusterized_df
    
    def LOO_split_by_cluster(self):
        """
        Splits the data into leave-one-out (LOO) train and validation sets based on item clusters for each user.

        Updates:
        - self.train_LOO: Pandas DataFrame containing the LOO train set with NaN ratings for validation movies.
        - self.y_true_LOO_by_cluster: Dictionary mapping user IDs to lists of true ratings for each cluster (validation set).
        - self.test_items_LOO: Dictionary mapping user IDs to lists of movie IDs for the validation set.
        """
        
        users = set(self.data[self.userId_col_name])
        train_LOO = pd.DataFrame()
        y_true_LOO_df = pd.DataFrame()
            
        #for each user I build clusters of items and for each cluster build train_LOO and test_LOO time sensitive
        for u in users:
            to_clusterize = self.data[(self.data[self.userId_col_name]==u) & \
                (self.data[self.rating_col_name].notna())][[self.rating_col_name, self.movieId_col_name]]
            
            clusters = self.clusterize_user(u, to_clusterize)
            items_timestamp = self.data[self.data[self.userId_col_name] == u]
            aux_df = pd.merge(clusters, items_timestamp, on=[self.movieId_col_name, self.rating_col_name], how='left')
            
            # get the movies that will be the validation set to fetch the better T0 for each cluster and each user
            max_timestamp_movies_idx = aux_df.groupby('cluster_label')[self.time_col_name].idxmax()
            col_list = [self.userId_col_name, self.movieId_col_name, self.rating_col_name, self.time_col_name, 'cluster_label']
            y_true_LOO_aux = aux_df.loc[max_timestamp_movies_idx, col_list].sort_values("cluster_label")
            # set np.nan for train_LOO
            y_true_LOO_movieid=list(y_true_LOO_aux[self.movieId_col_name])
            train_df_LOO_aux = aux_df.copy()
            train_df_LOO_aux.loc[train_df_LOO_aux[self.movieId_col_name].isin(y_true_LOO_movieid), self.rating_col_name] = np.nan
            train_LOO     = pd.concat([train_LOO, train_df_LOO_aux])
            y_true_LOO_df = pd.concat([y_true_LOO_df, y_true_LOO_aux]) 
        # transform to dict for mapping after
        self.y_true_LOO_by_cluster = y_true_LOO_df.groupby(self.userId_col_name)[self.rating_col_name].apply(list).to_dict()
        self.test_items_LOO = y_true_LOO_df.groupby(self.userId_col_name)[self.movieId_col_name].apply(list).to_dict()
        
        #sorting columns in train_LOO
        self.train_LOO = train_LOO[[self.userId_col_name, self.movieId_col_name, self.rating_col_name, self.time_col_name, 'cluster_label']]

    def classify_scenario(self, train_LOO, test_items_LOO):
        """
        Assigns different scenarios ('sim_count=0', 'sim_count=1', 'sim_count>1') to each row in the training DataFrame based on
        the count of similar items in the given similarity dictionary.

        Parameters:
            train_LOO (pd.DataFrame): Training DataFrame containing user-item interactions and cluster labels.
            test_items_LOO (dict): Dictionary mapping user IDs to lists of test items, grouped by item clusters.
        """
        
        train_LOO_scenario = train_LOO.copy()
        train_LOO_scenario['scenario'] = None
        test_items_LOO_scenario_sim_count_greater_1 = {}
        for u, cluster_test_items_LOO in test_items_LOO.items():
            test_items_LOO_scenario_sim_count_greater_1[u] = {cluster: item for cluster, item in enumerate(cluster_test_items_LOO)}
            for cluster, test_item in enumerate(cluster_test_items_LOO):
                train_items_user_cluster = train_LOO_scenario.loc[(train_LOO_scenario['cluster_label'] == cluster) & 
                                                                    (train_LOO_scenario[self.userId_col_name] == u), self.movieId_col_name]
                sim_count = len(set(self.items_similarities.get(test_item, {})) & set(train_items_user_cluster))
                
                if sim_count == 0:
                    train_LOO_scenario.loc[(train_LOO_scenario['cluster_label'] == cluster) & (train_LOO_scenario[self.userId_col_name] == u), "scenario"] = 'sim_count=0'
                    # remove the test_items_LOO that not have any sim value between items inside the cluster and the item to predict the rating
                    if cluster in test_items_LOO_scenario_sim_count_greater_1[u]:
                        del test_items_LOO_scenario_sim_count_greater_1[u][cluster]
                elif sim_count == 1:
                    train_LOO_scenario.loc[(train_LOO_scenario['cluster_label'] == cluster) & (train_LOO_scenario[self.userId_col_name] == u), "scenario"] = 'sim_count=1'
                    # remove the test_items_LOO that have only one sim value between items inside the cluster and the item to predict the rating
                    if cluster in test_items_LOO_scenario_sim_count_greater_1[u]:
                        del test_items_LOO_scenario_sim_count_greater_1[u][cluster]
                elif sim_count > 1:
                    train_LOO_scenario.loc[(train_LOO_scenario['cluster_label'] == cluster) & (train_LOO_scenario[self.userId_col_name] == u), "scenario"] = 'sim_count>1'
        
            self.train_LOO_scenario = train_LOO_scenario
            self.test_items_LOO_scenario_sim_count_greater_1 = test_items_LOO_scenario_sim_count_greater_1

    def get_time_weight_LOO(self, t0, data_by_cluster_user):
        """
        Calculate time weights for that T0 candidate.

        Parameters:
        - t0(int): T0 candidate for that user and that cluster of items.
        - data_by_cluster_user: Pandas DataFrame containing user-cluster specific data, including a 'timestamp' column.

        Returns:
        - time_weight: Numpy array containing time weights calculated.
        """     
        LAMBDA = 1 / t0
        timestamp = data_by_cluster_user[self.time_col_name].values
        time_weight = np.exp(-LAMBDA * timestamp)
        return time_weight

    def LOO_prediction(self, u, i, cluster, t0):
        """
        Predicts the rating for a given user-item pair using the Leave-One-Out (LOO) approach.

        Parameters:
            u (int): User ID for whom the rating is predicted.
            i (int): Item ID for which the rating is predicted.
            cluster (int): Cluster label indicating the group of items to consider for the prediction.
            t0 (float): T0 candidate value for the user and cluster, used in time-sensitive weighting.

        Returns:
            Tuple: A tuple containing the predicted rating and the mean rating of the user for the
            validation set of the cluster.
        """
        # Only use the cluster of items that have more than one neighbor (sim_count>1)
        data_by_cluster_user = self.train_LOO_scenario.loc[(self.train_LOO_scenario["scenario"]=='sim_count>1') & 
                                                           (self.train_LOO_scenario["cluster_label"]==cluster)  & 
                                                           (self.train_LOO_scenario[self.userId_col_name]==u)   &
                                                           (self.train_LOO_scenario[self.rating_col_name].notna())
                                                            ]

        ratings_neighbors_LOO = data_by_cluster_user[self.rating_col_name].values
        user_ratings_mean = np.mean(ratings_neighbors_LOO)
        items_rated = data_by_cluster_user[self.movieId_col_name].values

        sim_weight_LOO = [self.items_similarities[i].get(rated, 0) for rated in items_rated]
        time_weight_LOO = self.get_time_weight_LOO(t0, data_by_cluster_user)
        num = sum(np.array(sim_weight_LOO) * np.array(ratings_neighbors_LOO) * np.array(time_weight_LOO))
        denom = sum(np.array(sim_weight_LOO) * time_weight_LOO)
        
        if denom == 0:
            return user_ratings_mean, user_ratings_mean
        
        hat_rating = num / denom    
        
        return hat_rating, user_ratings_mean

    def get_T0_by_user_cluster(self):
        """
        Compute T0 values for each user-cluster pair using the Leave-One-Out (LOO) prediction and optimization.

        Updates:
        - self.T0_by_user_cluster_df: Pandas DataFrame containing computed T0 values for each user-cluster pair.
        - self.user_cluster_T0_map_dict: Dictionary mapping user IDs and cluster labels to their corresponding T0 values.
        """
        T0_by_user_cluster = []  
        user_cluster_T0_map_dict = {}
        for u, clusters_items in tqdm(self.test_items_LOO_scenario_sim_count_greater_1.items(), desc="Computing T0 values", unit=" T0 values"):
            for cluster, i in clusters_items.items():
                # Define the objective function to minimize
                def objective_function(T0_candidate):
                    hat_rating, _ = self.LOO_prediction(u, i, cluster, T0_candidate)
                    mae = mean_absolute_error(self.y_true_LOO_by_cluster[u][cluster], hat_rating)
                    return mae
                # lower and upper bounds for T0 candidates
                lower_bound_T0 = 1  #24   #min bound 1 day
                upper_bound_T0 = 90 #2190 #max bound 3 months

                result = minimize_scalar(objective_function, bounds=(lower_bound_T0, upper_bound_T0))
                T0 = result.x
                user_cluster_t0_dict = {self.userId_col_name: u, self.movieId_col_name:i, 'cluster_label': cluster, 'T0': T0}
                T0_by_user_cluster.append(user_cluster_t0_dict)
                
                # crafting user_cluster_T0_map_dict to map T0 value
                if u not in user_cluster_T0_map_dict:
                    user_cluster_T0_map_dict[u] = {cluster: T0}
                # Update the T0 value for the user-cluster pair
                user_cluster_T0_map_dict[u][cluster] = T0
                
        self.T0_by_user_cluster_df = pd.DataFrame(T0_by_user_cluster)
        self.user_cluster_T0_map_dict = user_cluster_T0_map_dict
        
    def build_trainT0(self):
        """
        Builds the training set with time-sensitive T0 values that were calculated for each user and cluster.

        Updates:
        - Fills the NaN rating values for items in the validation set in 'test_items_LOO', that were used to find the T0 values.
        - Merges the training set with the computed T0 values for each user-cluster pair.
        - Sets T0 values for items with 'sim_count=1' using the "T0 user-cluster mean" strategy.
        """
        # fill the nan rating values of test_items_LOO
        for u, cluster_items in self.test_items_LOO.items():
            for i in cluster_items:
                self.train_LOO_scenario.loc[(self.train_LOO_scenario[self.userId_col_name]==u) & 
                    (self.train_LOO_scenario[self.movieId_col_name]==i), self.rating_col_name] = self.data.loc[(self.data[self.userId_col_name]==u) & 
                                                                                                               (self.data[self.movieId_col_name]==i), 
                                                                                                               self.rating_col_name].values
            
        merge_1 = pd.merge(left = self.train_LOO_scenario, right = self.T0_by_user_cluster_df, how = 'left', on = [self.userId_col_name, 
                                                                                                                self.movieId_col_name,'cluster_label'])

        train_df_T0 = pd.merge(left = self.data, right = merge_1, how = 'left', on = [self.userId_col_name, 
                                                                            self.movieId_col_name, 
                                                                            self.rating_col_name, 
                                                                            self.time_col_name])

        # find T0 value for items with sim_count = 1, using "T0 cluster mean" strategy
        T0_cluster_mean_by_user = self.T0_by_user_cluster_df.groupby(self.userId_col_name)["T0"].mean().to_dict()

        # store the userId and movieId for the items that cannot compute the prediction of T0
        self.sim_count_0_user_item = train_df_T0[train_df_T0['scenario']=='sim_count=0'][[self.userId_col_name, self.movieId_col_name]].groupby(self.userId_col_name)[self.movieId_col_name].apply(list).to_dict()
        # clean sim_count=0 values
        train_df_T0 = train_df_T0[(train_df_T0['scenario']=='sim_count>1') | (train_df_T0['scenario']=='sim_count=1')]

        # fill T0 for "sim_count = 1" 
        mask1 = train_df_T0["scenario"] == "sim_count=1"
        mask2 = train_df_T0["scenario"] == "sim_count>1"
        
        train_df_T0.loc[mask1, "T0"] = train_df_T0[self.userId_col_name].map(T0_cluster_mean_by_user)
        train_df_T0.loc[mask2, "T0"] = train_df_T0.apply(lambda row: self.user_cluster_T0_map_dict.get(row[self.userId_col_name], {})\
            .get(row['cluster_label'], None), axis=1)
        # cleaning auxiliary columns
        train_df_T0.drop(columns=['cluster_label', 'scenario'], inplace = True)
        self.train_T0 = train_df_T0
        
    def compute_neighbors(self, u, i):
        """
        Compute the top neighbors of item i for a given user u based on item similarity.
        Uses caching to avoid re-computing neighbors for the same user-item pair.
        
        Parameters:
        - u (int): User ID
        - i (int): Item ID
        
        Returns:
        - dict: Dictionary of neighbor items with their similarity scores
        """
        # Check the cache first
        cache_key = (u, i)
        if cache_key in self.cached_neighbors:
            return self.cached_neighbors[cache_key]
            
        # if there aren't any neighbors, return an empty key.
        sim_keys = self.items_similarities.get(i, {}).keys()
        non_nan_mask = None
        try:
            # Use the final user-item matrix for lookups instead of recreating it
            non_nan_mask = self.user_item_matrix_final.loc[u, :].notna()
        except (KeyError, IndexError):
            if self.verbose:
                print(f"Error: User {u} is not registered")
            self.cached_neighbors[cache_key] = {}
            return {}
            
        if non_nan_mask is not None:
            non_nan_idx = non_nan_mask[non_nan_mask].index
            # from the items that were interacted by user u, give me all the ones that have similarity with the item i
            j = list(set(sim_keys) & set(list(non_nan_idx)))
            
            if not j:  # No similar items found
                self.cached_neighbors[cache_key] = {}
                return {}
                
            # Save the similarities of the items in the dict sorted_similarities
            # Create a dictionary with keys from j and values from sim[i]
            sorted_similarities = {k: self.items_similarities[i][k] for k in j}
            # Sort the dictionary based on values in descending order
            sorted_similarities = dict(sorted(sorted_similarities.items(), key=lambda x: x[1], reverse=True))
            # Select the top neighbors values from the sorted dictionary
            neighbors_of_i = dict(list(sorted_similarities.items())[:self.n_neighbours])
            
            # Cache the result
            self.cached_neighbors[cache_key] = neighbors_of_i
            return neighbors_of_i
            
        # If we get here, no valid neighbors were found
        self.cached_neighbors[cache_key] = {}
        return {}

    def get_time_weight(self, u, i):
        """
        Calculate the final time weights for a given user and item based on their neighbors.
        Uses caching to avoid re-computing time weights for the same user-item pair.

        Parameters:
        - u (int): User ID.
        - i (int): Item ID.

        Returns:
        - time_weight (np.array): Numpy array containing time weights calculated based on T0 values.
        """
# Check the cache first
        cache_key = (u, i)
        if cache_key in self.cached_time_weights:
            return self.cached_time_weights[cache_key]
        
        # Get the neighbors for this user-item pair
        neighbours_items = self.compute_neighbors(u, i).keys()
        
        if not neighbours_items:
            # No neighbors, return empty array
            self.cached_time_weights[cache_key] = np.array([])
            return np.array([])
        
        # Create a copy of the relevant rows from train_T0
        df_aux = self.train_T0.loc[(self.train_T0[self.userId_col_name] == u) & \
            (self.train_T0[self.movieId_col_name].isin(neighbours_items)),:].copy()
        
        if df_aux.empty:
            # No data found for these neighbors
            self.cached_time_weights[cache_key] = np.array([])
            return np.array([])
        
        # Convert movieId column to categorical for proper sorting
        df_aux.loc[:, self.movieId_col_name] = \
            df_aux[self.movieId_col_name].astype(pd.CategoricalDtype(categories=neighbours_items, ordered=True))
        
        # Sort the DataFrame based on the items neighbors order to match sim weights
        df_aux = df_aux.sort_values(by=self.movieId_col_name)
        
        # Calculate time weights using T0 values
        t0 = df_aux["T0"].values
        LAMBDA = 1 / t0
        timestamp = df_aux[self.time_col_name].values
        time_weight = np.exp(-LAMBDA * timestamp)
        
        # Cache and return the result
        self.cached_time_weights[cache_key] = time_weight
        return time_weight

    def compute_prediction(self, u, i):
        """
        Predict the rating for a given user and item using time weight memory item collaborative filtering.
        Optimized to avoid rebuilding the user-item matrix on each prediction.

        Parameters:
        - u (int): User ID.
        - i (int): Item ID.

        Returns:
        - hat_rating (float): Predicted rating for the specified user-item pair.
        """
        # Check if the model has been fitted
        if not self.is_fitted:
            raise ValueError("The model must be fitted before making predictions.")
        
        # Get the neighbors for this user-item pair
        neighbors = self.compute_neighbors(u, i)
        
        if not neighbors:
            # No neighbors found - return the user's mean rating
            try:
                return self.user_means[u]
            except (KeyError, TypeError):
                if self.verbose:
                    print(f"Warning: Cold-start problem detected: User {u} not found in training data")
                return np.nan
        
        # Get the user's ratings for the neighbor items
        try:
            neighbors_rating = self.user_item_matrix_final.loc[u, list(neighbors.keys())].values
        except (KeyError, ValueError):
            if self.verbose:
                print(f"Warning: Cold-start problem detected: Unable to get ratings for user {u}")
            return np.nan
        
        # Get time weights for this user-item pair
        time_weight = self.get_time_weight(u, i)
        
        if len(time_weight) == 0 or np.isnan(time_weight).all():
            if self.verbose:
                print(f"Warning: T0 Cold-Start problem detected: user {u} has no valid time weights for item {i}")
            try:
                return self.user_means[u]
            except (KeyError, TypeError):
                return np.nan
        
        # Calculate the weighted prediction
        neighbors_values = np.array(list(neighbors.values()))
        num = sum(neighbors_rating * neighbors_values * time_weight)
        denom = sum(neighbors_values * time_weight)
        
        try:
            hat_rating = num / denom
            return hat_rating
        except ZeroDivisionError:
            if self.verbose:
                print(f"Warning: Cold-start problem detected: No valid neighbors found for item {i} for user {u}")
            try:
                return self.user_means[u]
            except (KeyError, TypeError):
                return np.nan
    
    def recommend(self, u, dict_map=None, n_recommendations=10):
        """
        Generate top n recommendations for a given user based on collaborative filtering.
        
        Parameters:
            u (int): User ID.
            dict_map (dict): Dictionary mapping internal item IDs to external item IDs (e.g., movie titles).
            n_recommendations (int): Number of top recommendations to generate.
        
        Returns:
            list: A list of top n recommended item IDs (or names if dict_map is provided).
        """
        if not self.is_fitted:
            raise ValueError("The model must be fitted before making recommendations.")
            
        try:
            # Get items the user hasn't rated yet
            try:
                user_unrated_mask = self.user_item_matrix_final.loc[u, :].isna()
                items_to_predict = list(user_unrated_mask[user_unrated_mask].index)
            except (KeyError, AttributeError):
                # If user not in matrix, consider all items as unrated
                items_to_predict = list(self.user_item_matrix_final.columns)
            
            # Compute predicted ratings for each unseen item
            rating_predictions = {}
            for i in items_to_predict:
                try:
                    prediction = self.compute_prediction(u, i)
                    if not np.isnan(prediction):
                        rating_predictions[i] = prediction
                except Exception as e:
                    if self.verbose:
                        print(f"Error predicting for item {i}: {str(e)}")
                    continue
            
            if not rating_predictions:
                if self.verbose:
                    print(f"Warning: No valid predictions for user {u}")
                return []
                
            # Sort items by predicted rating (highest first) and take top n
            all_recommendations = dict(sorted(rating_predictions.items(), key=lambda item: item[1], reverse=True))
            recommendations_ids = [k for k, _ in list(all_recommendations.items())[:n_recommendations]]
            
            # If a mapping is provided, convert internal IDs to external names/IDs
            if dict_map is not None:
                rec = [dict_map.get(item, f"Unknown item {item}") for item in recommendations_ids]
            else:
                rec = recommendations_ids

            return rec

        except Exception as e:
            if self.verbose:
                print(f"Error generating recommendations for user {u}: {str(e)}")
            return []
    
    def clear_cache(self):
        """
        Clear the caches to free up memory.
        Useful for long-running applications.
        """
        self.cached_neighbors = {}
        self.cached_time_weights = {}

class LFCollaborativeFilter:
    """
    LFCollaborativeFilter is a collaborative filtering recommendation system based on matrix factorization.

    Parameters:
        - steps (int): Number of steps for the gradient descent optimization (default is 5000).
        - lr (float): Learning rate for the gradient descent optimization (default is 0.0002).
        - reg (float): Regularization parameter to control overfitting (default is 3).
        - random_state (int): Seed for random number generation to ensure reproducibility (default is 1203).

    Attributes:
        - steps (int): Number of steps for the gradient descent optimization.
        - lr (float): Learning rate for the gradient descent optimization.
        - reg (float): Regularization parameter to control overfitting.
        - U (torch.Tensor): User matrix learned during the training process.
        - V (torch.Tensor): Item matrix learned during the training process.
        - users_map (dict): Mapping from user ID to torch matrix index.
        - items_map (dict): Mapping from item ID to torch matrix column.
        - R_copy (pd.DataFrame): Copy of the input data in a pivot table format.
        - random_state (int): Seed for random number generation.
        """
    def __init__(self, steps=5000, lr=0.0002, reg=3, random_state=1203):
        """
        Initialize the LFCollaborativeFilter with specified parameters.

        Parameters:
            - steps (int): Number of steps for the gradient descent optimization (default is 5000).
            - lr (float): Learning rate for the gradient descent optimization (default is 0.0002).
            - reg (float): Regularization parameter to control overfitting (default is 3).
            - random_state (int): Seed for random number generation to ensure reproducibility (default is 1203).
        """
        self.is_fitted = False
        self.steps = steps
        self.lr = lr
        self.reg = reg
        self.U = None
        self.V = None
        self.users_map = None
        self.items_map = None
        self.R_copy = None
        self.random_state = random_state
        torch.manual_seed(random_state)
    
    def fit(self, X, F=2):
        """
        Train the collaborative filter on the input data X with latent factor F.

        Parameters:
            - X (pd.DataFrame): Input data containing user, item, and rating information.
            - F (int): Number of latent factors (default is 2).
        """
        R = pd.pivot_table(X,
                        index   = "userId",
                        columns = "movieId",
                        values  = "rating")
        self.R_copy = R.copy()
        #map from userID to torch.matrix idx
        self.users_map = dict(zip(R.index, range(0, len(R.index))))
        #map from itemID to torch.matrix column
        self.items_map = dict(zip(R.columns, range(0, len(R.columns))))        
        
        R = R.map(lambda x: -1 if pd.isna(x) else x)
        # Move data to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        R = torch.tensor(R.values, dtype=torch.float32, device=device)
        # Initialize U and V random matrices
        self.U = torch.randn(R.shape[0], F, device=device)
        self.V = torch.randn(R.shape[1], F, device=device)
        
        prev_error = float('inf')
        
        for epoch in tqdm(range(self.steps)):
            R_hat = torch.mm(self.U, self.V.t())
            rated_mask = (R > 0).float()

            # Calculate error
            error = rated_mask * (R - R_hat)
            
            # Update U and V using gradient descent
            U_grad = -2 * torch.mm(error, self.V) + 2 * self.reg * self.U
            V_grad = -2 * torch.mm(error.t(), self.U) + 2 * self.reg * self.V
            self.U = self.U - self.lr * U_grad
            self.V = self.V - self.lr * V_grad

            # early stopping
            current_error = torch.sum(error**2).item() + self.reg * (torch.sum(self.U**2).item() + torch.sum(self.V**2).item())
            if current_error < 0.001 or abs(prev_error - current_error) < 1e-5:
                print(f"Early stopping at epoch {epoch + 1}. Final error: {current_error}")
                break
            prev_error = current_error
        
        self.is_fitted = True
        
    def compute_prediction(self, u, i):
        """
        Compute the predicted rating for a user u and an item i.

        Parameters:
            - u: User ID.
            - i: Item ID.

        Returns:
            - float: Predicted rating for the user-item pair.
        """
        r_hat = torch.matmul(self.U[self.users_map[u],:], self.V[self.items_map[i],:].t()).cpu().numpy()
        return round(float(r_hat), 1)

    def recommend(self, u, dict_map=None, n_recommendations=10):
        """
        Provide recommendations for a user u.

        Parameters:
            - u: User ID.
            - dict_map (dict): Dictionary mapping item IDs to corresponding names (default is None).
            - n_recommendations (int): Number of recommendations to provide (default is 10).
        """
        try:
            rating_predictions = {}
            items_to_predict = self.R_copy.loc[u, self.R_copy.loc[u, :].isna()].index
            
            for i in items_to_predict:
                rating_predictions[i] = self.compute_prediction(u, i)
            
            # Sort items by predicted rating (highest first) and take top n
            all_recommendations = dict(sorted(rating_predictions.items(), key=lambda item: item[1], reverse=True))
            recommendations_ids = [k for k, _ in list(all_recommendations.items())[:n_recommendations]]
            
            # If a mapping is provided, convert internal IDs to external names/IDs
            if dict_map is not None:
                rec = [dict_map.get(item, f"Unknown item {item}") for item in recommendations_ids]
            else:
                rec = recommendations_ids

            return rec

        except KeyError:
            print(f"Warning: User {u} is not registered (Cold-Star Problem)")
            return []