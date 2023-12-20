import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import pdist
from scipy.optimize import minimize_scalar
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans


from .metrics import mean_absolute_error


class MemoryCollaborativeFilter:
    def __init__(self, min_overlap=5, n_neighbours=35, n_recommendations=10, 
                 userId_col_name="userId", movieId_col_name='movieId',
                 rating_col_name="rating"):
        """
        Initialize the MemoryCollaborativeFilter object with specified parameters.

        Parameters:
        - min_overlap (int): Minimum number of overlapping users required for similarity calculation.
        - n_neighbours (int): Number of neighbors to consider in collaborative filtering.
        - n_recommendations (int): Number of top recommendations to generate.
        """
        self.min_overlap = min_overlap
        self.n_neighbours = n_neighbours
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
        Compute the top neighbors of item i for a given user u based on item similarity.
        """
    
        sim_keys = self.items_similarities[i].keys()
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
            # Select the top neighbors values from the sorted dictionary
            neighbors_of_i = dict(list(sorted_similarities.items())[:self.n_neighbours])

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
            
            
            





class TWMemoryCollaborativeFilter:
    def __init__(self, min_overlap=5, n_neighbours=35, n_recommendations=10, max_n_clusters = 3, 
                 rescale_parameter = 1e9, userId_col_name="userId", movieId_col_name='movieId',
                 rating_col_name="rating", time_col_name = 'timestamp'):
        """
        Initialize the MemoryCollaborativeFilter object with specified parameters.

        Parameters:
        - min_overlap (int): Minimum number of overlapping users required for similarity calculation.
        - n_neighbours (int): Number of neighbors to consider in collaborative filtering.
        - n_recommendations (int): Number of top recommendations to generate.
        """
        self.min_overlap = min_overlap
        self.n_neighbours = n_neighbours
        self.n_recommendations = n_recommendations
        self.userId_col_name = userId_col_name
        self.movieId_col_name = movieId_col_name
        self.rating_col_name = rating_col_name
        self.time_col_name = time_col_name
        self.max_n_clusters = max_n_clusters
        self.refine_n_clusters = True
        self.rescale_parameter = rescale_parameter

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
        
        self.data = data
        
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
        
        self.LOO_split_by_cluster()
        self.classify_scenario(self.train_LOO, self.test_items_LOO)
        self.get_T0_by_user_cluster()
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
                kmeans = KMeans(n_clusters=n_clusters, random_state=1203, n_init = 'auto')
                cluster_labels = kmeans.fit_predict(np.array(to_clusterize['rating']).reshape(-1, 1))
                
                # Calculate Silhouette Score
                if len(set(cluster_labels))>1:
                    sil_score = silhouette_score(np.array(to_clusterize['rating']).reshape(-1, 1), cluster_labels)
                    sil_scores[n_clusters] = sil_score

                    # Find the best number of clusters
                    best_n_clusters = max(sil_scores, key=sil_scores.get)
            
            if len(set(cluster_labels))==1:
                # the user rated similarly all the movies that he saw.
                best_n_clusters = 1
                print(f"n_clusters for user {u} is 1, the user rated similarly all the movies that he saw.")
                    
            # Create and fit KMeans model with the best number of clusters
            best_kmeans = KMeans(n_clusters=best_n_clusters, random_state=1203, n_init = 'auto')
            best_cluster_labels = best_kmeans.fit_predict(np.array(to_clusterize['rating']).reshape(-1, 1))

            clusterized_df = pd.DataFrame({'movieId': to_clusterize['movieId'].values,
                                            'rating': to_clusterize['rating'].values,  
                                            'cluster_label': best_cluster_labels})
            return clusterized_df    
        
        # if refine_n_clusters is False, the computation cost its reduced and the number of clusters is
        # set as max_n_clusters
        best_kmeans = KMeans(n_clusters = self.max_n_clusters, random_state=1203, n_init = 'auto')
        best_cluster_labels = best_kmeans.fit_predict(np.array(to_clusterize['rating']).reshape(-1, 1))

        clusterized_df = pd.DataFrame({'movieId': to_clusterize['movieId'].values,
                                        'rating': to_clusterize['rating'].values,  
                                        'cluster_label': best_cluster_labels})
        return clusterized_df
    
    
    
    def LOO_split_by_cluster(self):
        """
        Splits the data into leave-one-out (LOO) train and validation sets based on item clusters for each user.

        Parameters:
            data (pd.DataFrame): Input DataFrame containing user-item interactions.
            max_n_clusters (int): Maximum number of clusters to consider for item clustering.
            userId_col_name (str): Name of the column containing user IDs.
            movieId_col_name (str): Name of the column containing movie IDs.
            time_col_name (str): Name of the column containing timestamps.
            rating_col_name (str): Name of the column containing ratings.

        Returns:
            pd.DataFrame: LOO train DataFrame with NaNs for validation items.
            dict: Dictionary mapping user IDs to true ratings in the validation set, grouped by item clusters.
            dict: Dictionary mapping user IDs to lists of movie IDs in the validation set, grouped by item clusters.
        """
        
        users = set(self.data['userId'])
        train_LOO = pd.DataFrame()
        y_true_LOO_df = pd.DataFrame()
            
        #for each user I build clusters of items and for each cluster build train_LOO and test_LOO time sensitive
        for u in users:
            to_clusterize = self.data[(self.data[self.userId_col_name]==u) & \
                (self.data[self.rating_col_name].notna())][[self.rating_col_name, self.movieId_col_name]]
            
            clusters=self.clusterize_user(u, to_clusterize)
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
        self.y_true_LOO_by_cluster = y_true_LOO_df.groupby(self.userId_col_name)[self.rating_col_name].apply(list).to_dict()
        self.test_items_LOO = y_true_LOO_df.groupby(self.userId_col_name)[self.movieId_col_name].apply(list).to_dict()
        
        #sorting columns 
        self.train_LOO = train_LOO[[self.userId_col_name, self.movieId_col_name, self.rating_col_name, self.time_col_name, 'cluster_label']]





    def classify_scenario(self, train_LOO, test_items_LOO):
        """
        Assigns different scenarios ('sim_count=0', 'sim_count=1', 'sim_count>1') to each row in the training DataFrame based on
        the count of similar items in the given similarity dictionary.

        Parameters:
            train_LOO (pd.DataFrame): Training DataFrame containing user-item interactions and cluster labels.
            test_items_LOO (dict): Dictionary mapping user IDs to lists of test items, grouped by item clusters.
            sim (dict): Dictionary containing item similarities.

        Returns:
            train_LOO_scenario (pd.DataFrame): Training DataFrame with an additional 'scenario' column indicating the scenario
                            based on the count of similar items for each user and cluster.
            test_items_LOO_scenario_sim_count_greater_1 (dict): Dictionary mapping user IDs to dictionaries of test items,
                                grouped by item clusters, with only the movieId of the clusters of items with sim_count>1.
        """
        
        train_LOO_scenario = train_LOO.copy()
        train_LOO_scenario['scenario'] = None
        test_items_LOO_scenario_sim_count_greater_1 = {}

        # for user, items in test_items_LOO.items():
            # test_items_LOO_scenario_sim_count_greater_1[user] = {cluster: item for cluster, item in enumerate(items)}
        
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





    def get_time_weight(self, t0, data_by_cluster_user):
        LAMBDA = 1 / t0
        timestamp = data_by_cluster_user['timestamp'].values
        time_weight = np.exp(-LAMBDA * timestamp / self.rescale_parameter) # reescale
        return time_weight


    def LOO_prediction(self, u, i, cluster, t0):
        
        # Only use the cluster of items that have more than one neighbor (sim_count>1)
        data_by_cluster_user = self.train_LOO_scenario.loc[(self.train_LOO_scenario["scenario"]=='sim_count>1') & 
                                                (self.train_LOO_scenario["cluster_label"]==cluster) & 
                                                (self.train_LOO_scenario[self.userId_col_name]==u) &
                                                (self.train_LOO_scenario[self.rating_col_name].notna())
                                                ]

        ratings_neighbors_LOO = data_by_cluster_user['rating'].values
        user_ratings_mean = np.mean(ratings_neighbors_LOO)
        items_rated = data_by_cluster_user['movieId'].values

        sim_weight = [self.items_similarities[i].get(rated, 0) for rated in items_rated]
        time_weight = self.get_time_weight(t0, data_by_cluster_user)
        num = sum(np.array(sim_weight) * np.array(ratings_neighbors_LOO) * np.array(time_weight))
        denom = sum(sim_weight * time_weight)
        try:
            hat_rating =  num / denom
            return hat_rating, user_ratings_mean
            
        except ZeroDivisionError:
            
            print(f"\nError item Cold-Star Problem: No hay items vecinos del item {i} para estimar el rating")
            return np.nan, user_ratings_mean




    def get_T0_by_user_cluster(self):
        
        T0_by_user_cluster = []  
        user_cluster_T0_map_dict = {}
 

        for u, clusters_items in tqdm(self.test_items_LOO_scenario_sim_count_greater_1.items(), desc="Computing T0 values", unit=" T0 values"):
            for cluster, i in clusters_items.items():
                # Define the objective function to minimize
                def objective_function(T0_candidate):
                    hat_rating = self.LOO_prediction(u, i, cluster, T0_candidate)[0]
                    mae = mean_absolute_error(self.y_true_LOO_by_cluster[u][cluster], hat_rating)
                    return mae

                # Set lower and upper bounds
                lower_bound_T0 = 10
                upper_bound_T0 = 200

                # Use minimize_scalar with bounds
                result = minimize_scalar(objective_function, bounds=(lower_bound_T0, upper_bound_T0))

                T0 = result.x
                # Create a new dictionary for each iteration and append to the list
                user_cluster_t0_dict = {'userId': u, 'movieId':i, 'cluster_label': cluster, 'T0': T0}
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
        T0_cluster_mean_by_user=self.T0_by_user_cluster_df.groupby(self.userId_col_name)["T0"].mean().to_dict()

        # fill T0 for "sim_count = 1" 
        train_df_T0.loc[train_df_T0["scenario"]=="sim_count=1", "T0"] = train_df_T0[self.userId_col_name].map(T0_cluster_mean_by_user)

        train_df_T0.loc[train_df_T0["scenario"] == "sim_count>1", "T0"] = train_df_T0.apply(lambda row: self.user_cluster_T0_map_dict\
            .get(row[self.userId_col_name], {}).get(row['cluster_label'], None), axis=1)

        # cleaning auxiliary columns
        train_df_T0.drop(columns=['cluster_label', 'scenario'], inplace= True)
        
        self.train_T0 = train_df_T0
        
        
    def compute_prediction(self, u, i):
        
        user_ratings = self.train_T0.loc[self.train_T0[self.userId_col_name]==u, self.rating_col_name].values
        user_ratings_mean = np.nanmean(user_ratings)
        all_neighbours = [neigh for neigh in self.items_similarities[i].keys()]
        items_rated_by_user = self.train_T0.loc[(self.train_T0[self.userId_col_name]==u) & (self.train_T0[self.rating_col_name].notna()) & (self.train_T0[self.movieId_col_name].isin(all_neighbours))]

        # Create a categorical type with the desired order
        cat_type = pd.CategoricalDtype(categories=all_neighbours, ordered=True)

        # Apply the categorical type to the "movieId" column and sort the DataFrame
        sorted_items_rated_by_user = items_rated_by_user.astype({self.movieId_col_name: cat_type}).sort_values(by=self.movieId_col_name).head(self.n_neighbours)

        # get the sim_weight of the neighbours
        sim_weight = [self.items_similarities[i][neighbour] for neighbour in sorted_items_rated_by_user[self.movieId_col_name].values]

        # get the neighbors rating
        neighbours_ratings = sorted_items_rated_by_user[self.rating_col_name].values
        
        
        
        T0 = sorted_items_rated_by_user["T0"].values
        time_weight = self.get_time_weight(T0, sorted_items_rated_by_user)
        
        num = sum(np.array(sim_weight) * np.array(neighbours_ratings) * np.array(time_weight))
        denom = sum(sim_weight * time_weight)
        try:
            hat_rating =  num / denom
            return hat_rating, user_ratings_mean
            
        except ZeroDivisionError:
            
            print(f"\nError item Cold-Star Problem: No hay items vecinos del item {i} para estimar el rating")
            return np.nan, user_ratings_mean
        
        
    def recommend(self, u, dict_map):
        """
        Generate top n recommendations for a given user based on item collaborative filtering with time weight.
        """

        try :
            rating_predictions = {}
            items_to_predict = self.train_T0.loc[(self.train_T0[self.userId_col_name] == u) & (self.train_T0[self.rating_col_name].isna()), self.movieId_col_name].values
            for i in items_to_predict:
                rating_predictions[i] = self.compute_prediction(u, i)

            all_recommendations = dict(sorted(rating_predictions.items(), key=lambda item: item[1], reverse=True))
            n_recommendations = [k for k, _ in list(all_recommendations.items())[:self.n_recommendations]]
            rec = [dict_map[item] for item in n_recommendations]
            return rec
        
        except KeyError:
            print(f"Warning Cold-Star Problem detected: User {u} is not registered")