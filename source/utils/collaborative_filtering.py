import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import pdist
from scipy.optimize import minimize_scalar
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from .metrics import mean_absolute_error


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
        
        for item1 in tqdm(idx_items, desc="Computing Similarities", unit=" item"):
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
            print(f"Warning: Cold-Star problem detected: No neighbours found for item {i} to predict its rating")
            return np.nan, user_ratings_mean[u]
            
    def recommend(self, u, dict_map=None, test_items=None, n_recommendations=10):
        """     
        Generate top n recommendations for a given user based on time-weighted item collaborative filtering.

        Parameters:
        - u (int): User ID.
        - dict_map (dict): Dictionary mapping internal item IDs to external item IDs. Defaults to None.
        - test_items (list, optional): List of movies IDs set for test. Defaults to None.
        - n_recommendations (int): Number of top recommendations to generate.

        Returns:
        - rec (list): List of top n recommended item IDs for the specified user.
        
        If the method receives `test_items`, it enters test mode and generates recommendations based on the provided list
        of movies IDs, in this mode it returns the ids of the movies that are recommended. Otherwise, it generates 
        recommendations based on the items that the user has not interacted with before.
        """
        try :
            rating_predictions = {}
            items_to_predict = list(self.user_item_matrix.loc[u, self.user_item_matrix.loc[u, :].isna()].index)
            
            #if the method receives test_items, it enters test mode.
            if test_items is not None:
                items_to_predict = test_items[u]
                for i in items_to_predict:
                    rating_predictions[i] = self.compute_prediction(u, i)

                    all_recommendations = dict(sorted(rating_predictions.items(), key=lambda item: item[1], reverse=True))
                    recommendations_ids = [k for k, _ in list(all_recommendations.items())[:n_recommendations]]
                return recommendations_ids
            
            for i in items_to_predict:
                rating_predictions[i] = self.compute_prediction(u, i)

            all_recommendations = dict(sorted(rating_predictions.items(), key=lambda item: item[1], reverse=True))
            recommendations_ids = [k for k, _ in list(all_recommendations.items())[:n_recommendations]]
            rec = [dict_map[item] for item in recommendations_ids]
            return rec
        
        except KeyError:
            print(f"Warning Cold-Star Problem detected: User {u} is not registered")

class TWMemoryCollaborativeFilter:
    def __init__(self, min_overlap=5, n_neighbours=40, max_n_clusters=3, 
                 refine_n_clusters= True, userId_col_name="userId", movieId_col_name='movieId', 
                 rating_col_name="rating", time_col_name = 'timestamp', verbose = False, random_state = 1203):
        """
        Initialize the TWMemoryCollaborativeFilter object with specified parameters.

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

        # scaling timestamps for each user in days based on their first interaction with the system.
        self.data[self.time_col_name] = pd.to_datetime(self.data[self.time_col_name], unit='s')
        self.data['initial_timestamp'] = self.data.groupby(self.userId_col_name)[self.time_col_name].transform('min')
        self.data[self.time_col_name] = (self.data[self.time_col_name] - self.data['initial_timestamp']).dt.total_seconds() / (24 * 3600)
        self.data = self.data.drop(columns=['initial_timestamp'])

        
        # Transform data to UxI matrix
        self.user_item_matrix = pd.pivot_table(self.data,
                                index   = self.userId_col_name,
                                columns = self.movieId_col_name,
                                values  = self.rating_col_name)
        
        idx_items = list(self.user_item_matrix.columns)
        similarity_dict = {}
        
        for item1 in tqdm(idx_items, desc="Computing Similarities", unit="item"):
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
        # for each user split the items in clusters using the ratings.
        self.LOO_split_by_cluster()
        # build different scenarios for each item's cluster
        self.classify_scenario(self.train_LOO, self.test_items_LOO)
        # considering the scenario of that item and that user find the T0 value.
        self.get_T0_by_user_cluster()
        # build the final dataset UxIxT0
        self.build_trainT0()
        
    def clusterize_user(self, u, to_clusterize):
        """
        Clusterizes items for a given user based on ratings using KMeans clustering.

        Parameters:
            u (int): User ID.
            to_clusterize (pd.DataFrame): DataFrame containing 'rating' and 'movieId' columns to be clustered.
            max_n_clusters (int): Maximum number of clusters to consider. A larger number may result in overfitting.
            refine_n_clusters (bool): If True, refines the number of clusters based on silhouette scores.

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
        best_kmeans = KMeans(n_clusters = self.max_n_clusters, random_state=self.random_state, n_init = 'auto')
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

        Notes:
        - This method iterates through each user, builds item clusters, and creates LOO train and validation sets.
        - The clusters are obtained using the clusterize_user method within the class.
        - The validation set contains the movies with the maximum timestamp for each cluster, used to find the best T0.
        - The ratings for validation set movies in the train set are set to NaN.
        - The results are stored in the instance attributes self.train_LOO, self.y_true_LOO_by_cluster, and self.test_items_LOO.
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
            sim (dict): Dictionary containing item similarities.

        Updates:
            self.train_LOO_scenario (pd.DataFrame): Training DataFrame with an additional 'scenario' column indicating the scenario
                            based on the count of similar items for each user and cluster.
            self.test_items_LOO_scenario_sim_count_greater_1 (dict): Dictionary mapping user IDs to dictionaries of test items,
                                grouped by item clusters, with only the movieId of the clusters of items with sim_count>1.
        Psudocode:
        FOR each user (u) in test_items_LOO:
            FOR each item (test_item) in the list associated with user (u):
                Get all movie IDs (train_items_user_cluster) for user (u) in the same cluster (cluster) as test_item from train_LOO
                Count the number of similar items (sim_count) between test_item and train_items_user_cluster
                IF sim_count is equal to 0:
                    Set the case column of the corresponding row in train_LOO to "sim_count=0"
                ELSE IF sim_count is equal to 1:
                    Set the case column of the corresponding row in train_LOO to "sim_count=1"
                ELSE:
                    Set the case column of the corresponding row in train_LOO to "sim_count>1"
        """
        
        train_LOO_scenario = train_LOO.copy()
        train_LOO_scenario['scenario'] = None
        test_items_LOO_scenario_sim_count_greater_1 = {}
        for u, cluster_test_items_LOO in test_items_LOO.items():
            test_items_LOO_scenario_sim_count_greater_1[u] = {cluster: item for cluster, item in enumerate(cluster_test_items_LOO)}
            for cluster, test_item in enumerate(cluster_test_items_LOO):
                train_items_user_cluster = train_LOO_scenario.loc[(train_LOO_scenario['cluster_label'] == cluster) & 
                                                                    (train_LOO_scenario[self.userId_col_name] == u), self.movieId_col_name]
                sim_count = len(set(self.items_similarities[test_item]) & set(train_items_user_cluster))
                
                if sim_count == 0:
                    train_LOO_scenario.loc[(train_LOO_scenario['cluster_label'] == cluster) & (train_LOO_scenario[self.userId_col_name] == u), "scenario"] = 'sim_count=0'
                    # remove the test_items_LOO that not have any sim value between items inside the cluster and the item to predict the rating
                    del test_items_LOO_scenario_sim_count_greater_1[u][cluster]
                elif sim_count == 1:
                    train_LOO_scenario.loc[(train_LOO_scenario['cluster_label'] == cluster) & (train_LOO_scenario[self.userId_col_name] == u), "scenario"] = 'sim_count=1'
                    # remove the test_items_LOO that have only one sim value between items inside the cluster and the item to predict the rating
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
        timestamp = data_by_cluster_user['timestamp'].values
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

        Notes:
            - The method considers items with 'sim_count>1' in the training set to make predictions.
            - Utilizes adjusted cosine similarity and time-sensitive weighting in the prediction.
            - The prediction is based on the weighted sum of ratings from similar items within the cluster.
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
        denom = sum(sim_weight_LOO * time_weight_LOO)
        hat_rating =  num / denom
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
                    hat_rating = self.LOO_prediction(u, i, cluster, T0_candidate)[0]
                    mae = mean_absolute_error(self.y_true_LOO_by_cluster[u][cluster], hat_rating)
                    return mae
                # lower and upper bounds for T0 candidates
                lower_bound_T0 = 1e3
                upper_bound_T0 = 5e3

                result = minimize_scalar(objective_function, bounds=(lower_bound_T0, upper_bound_T0))
                T0 = result.x
                user_cluster_t0_dict = {self.userId_col_name: u, self.movieId_col_name:i, 'cluster_label': cluster, 'T0': T0}
                T0_by_user_cluster.append(user_cluster_t0_dict)
                
                # crafting user_cluster_T0_map_dict to map T0 value
                if u not in user_cluster_T0_map_dict:
                    user_cluster_T0_map_dict[u] = {cluster: T0}
                # Check if the user_id is already in the nested dictionary
                user_cluster_T0_map_dict[u][cluster] = T0
                T0_by_user_cluster_df = pd.DataFrame(T0_by_user_cluster)
        self.T0_by_user_cluster_df = T0_by_user_cluster_df
        self.user_cluster_T0_map_dict = user_cluster_T0_map_dict
        
    def build_trainT0(self):
        """
        Builds the training set with time-sensitive T0 values that were calculated for each user and cluster.

        Updates:
        - Fills the NaN rating values for items in the validation set in 'test_items_LOO', that were used to find the T0 values.
        - Merges the training set with the computed T0 values for each user-cluster pair.
        - Sets T0 values for items with 'sim_count=1' using the "T0 user-cluster mean" strategy.

        Notes:
        - Uses the T0 values obtained from the optimization process in the 'get_T0_by_user_cluster' method.
        - Considers items with 'sim_count>1' for building the training set.
        - Utilizes user-cluster specific T0 values for time-sensitive weighting in collaborative filtering.
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
        self.sim_count_0_user_item = train_df_T0[train_df_T0['scenario']=='sim_count=0'][["userId", 'movieId']].groupby('userId')['movieId'].apply(list).to_dict()
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
        """
        # if there aren't any neighbors, return an empty key.
        sim_keys = self.items_similarities.get(i, {}).keys()
        non_nan_mask = None
        try:
            non_nan_mask = self.user_item_matrix.loc[u, :].notna()
        except IndexError:
            if self.verbose:
                print(f"Error: User {u} is not registered")
            
        if non_nan_mask is not None:
            non_nan_idx = non_nan_mask[non_nan_mask].index
            # from the items that were interacted by user u, give me all the ones that have similarity with the item i
            j = list(set(sim_keys) & set(list(non_nan_idx)))
            # Save the similarities of the items in the dict sorted_similarities
            # Create a dictionary with keys from j and values from sim[i]
            sorted_similarities = {k: self.items_similarities[i][k] for k in j}
            # Sort the dictionary based on values in descending order
            sorted_similarities = dict(sorted(sorted_similarities.items(), key=lambda x: x[1], reverse=True))
            # Select the top neighbors values from the sorted dictionary
            neighbors_of_i = dict(list(sorted_similarities.items())[:self.n_neighbours])
            return neighbors_of_i

    def get_time_weight(self, u, i):
        """
        Calculate the final time weights for a given user and item based on their neighbors.

        Parameters:
        - u (int): User ID.
        - i (int): Item ID.

        Returns:
        - time_weight (np.array): Numpy array containing time weights calculated based on T0 values that were calculated before.

        Notes:
        - Utilizes the neighbors of the given item for time-weighted collaborative filtering.
        - Computes time weights using the exponential decay function with T0 values.
        - The time weights are calculated based on the timestamps of interactions with neighbors.
        """        
        neighbours_items = self.compute_neighbors(u, i).keys()
        df_aux = self.train_T0.loc[(self.train_T0[self.userId_col_name] == u) & \
            (self.train_T0[self.movieId_col_name].isin(neighbours_items)),:].copy()
        df_aux.loc[:, self.movieId_col_name] = \
            df_aux[self.movieId_col_name].astype(pd.CategoricalDtype(categories=neighbours_items, ordered=True))
        # Sort the DataFrame based on the items neighbours order to be like sim weight
        df_aux = df_aux.sort_values(by=self.movieId_col_name)
        t0 = df_aux["T0"].values
        LAMBDA = 1 / t0
        timestamp = df_aux[self.time_col_name].values
        time_weight = np.exp(-LAMBDA * timestamp) # reescale
        return time_weight

    def compute_prediction(self, u, i):
        """
        Predict the rating for a given user and item using time weight memory item collaborative filtering.

        Parameters:
        - u (int): User ID.
        - i (int): Item ID.

        Returns:
        - hat_rating (float): Predicted rating for the specified user-item pair.
        - user_ratings_mean (float): Mean rating of the user for normalization.

        Notes:
        - Computes the prediction based on item collaborative filtering using neighbors.
        - Utilizes the similarity between items and their ratings to predict the target rating.
        - Considers time-weighted collaborative filtering using exponential decay with T0 values.
        - Handles cases where neighbors are not available or when the prediction encounters a division by zero.
        - Provides a warning for the Cold-Star problem when no neighbors are found for the item.
        """
        # Transform data to UxI matrix
        self.user_item_matrix = pd.pivot_table(self.train_T0,
                                index   = self.userId_col_name,
                                columns = self.movieId_col_name,
                                values  = self.rating_col_name)
        
        neighbors = self.compute_neighbors(u, i)
        user_ratings_mean = np.mean(self.user_item_matrix, axis=1)[u]
        neighbors_rating = self.user_item_matrix.loc[u, :].dropna()[list(neighbors.keys())] 
        time_weight = self.get_time_weight(u, i)
        if np.isnan(time_weight).all() & self.verbose:
            print(f"Warning: T0 Cold-Star problem detected: user {u} has sim_cout = 1 and don't have any T0 value to use for mean strategy")
        num = sum(np.array(neighbors_rating) * np.array(list(neighbors.values())) * time_weight)
        denom = sum(list(neighbors.values()) * time_weight)
        try:
            hat_rating = num / denom
            return hat_rating, user_ratings_mean
        
        except ZeroDivisionError:
            if self.verbose:
                print(f"Warning: Cold-Star problem detected: No neighbours found for item {i} to predict its rating for user {u}")
            return np.nan, user_ratings_mean
    
    def recommend(self, u, dict_map=None, test_items=None, n_recommendations=10):
        """     
        Generate top n recommendations for a given user based on time-weighted item collaborative filtering.

        Parameters:
        - u (int): User ID.
        - dict_map (dict): Dictionary mapping internal item IDs to external item IDs. Not used in test mode.
        - test_items (list, optional): List of movies IDs set for test. Defaults to None.
        - n_recommendations (int): Number of top recommendations to generate.

        Returns:
        - rec (list): List of top n recommended item IDs for the specified user.
        
        If the method receives `test_items`, it enters test mode and generates recommendations based on the provided list
        of movies IDs, in this mode it returns the ids of the movies that are recommended. Otherwise, it generates 
        recommendations based on the items that the user has not interacted with before.
        
        Note: In test mode, the method dont use the dict_map hiperparameter.
        """
        try :
            rating_predictions = {}
            items_to_predict = list(self.user_item_matrix.loc[u, self.user_item_matrix.loc[u, :].isna()].index)
            
            #if the method receives test_items, it enters test mode.
            if test_items is not None:
                items_to_predict = test_items[u]
                for i in items_to_predict:
                    rating_predictions[i] = self.compute_prediction(u, i)

                    all_recommendations = dict(sorted(rating_predictions.items(), key=lambda item: item[1], reverse=True))
                    recommendations_ids = [k for k, _ in list(all_recommendations.items())[:n_recommendations]]
                return recommendations_ids
            
            for i in items_to_predict:
                rating_predictions[i] = self.compute_prediction(u, i)

            all_recommendations = dict(sorted(rating_predictions.items(), key=lambda item: item[1], reverse=True))
            recommendations_ids = [k for k, _ in list(all_recommendations.items())[:n_recommendations]]
            rec = [dict_map[item] for item in recommendations_ids]
            return rec
        
        except KeyError:
            print(f"Warning Cold-Star Problem detected: User {u} is not registered")
