import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans


""" 
pros y contras sobre este metodo de split

1er tecnica: uso la historia de las ultimas 20 peliculas
2da tecnica: una funcion q baje el peso a medida q avanza el tiempo.


comparar teniendo en cuenta el tiempo y no, para mostrar el peso de tener en cuenta el tiempo.
"""

def train_test_split(data, userId_col_name="userId", movieId_col_name='movieId',
                     rating_col_name="rating", time_col_name="timestamp",
                     proportion_test_set=0.2, min_overlap=False,
                     min_interactions_users=25, min_interactions_items=35):
    """
    Divide the dataset into training and test sets, reserving the most recent records for testing based on the timestamp column.

    Parameters:
    - data (DataFrame): The input DataFrame containing user-item interactions.
    - userId_col_name (str, optional): The name of the user ID column. Defaults to "userId".
    - movieId_col_name (str, optional): The name of the movie ID column. Defaults to 'movieId'.
    - rating_col_name (str, optional): The name of the rating column. Defaults to "rating".
    - time_col_name (str, optional): The name of the timestamp column. Defaults to "timestamp".
    - proportion_test_set (float, optional): The proportion of the dataset to be used as the test set. Defaults to 0.2.
    - min_overlap (bool, optional): If True, prune the data by requiring minimum values of overlapping interactions for users and items.
    - min_interactions_users (int, optional): The minimum number of interactions required per user for inclusion. Defaults to 25.
    - min_interactions_items (int, optional): The minimum number of interactions required per item (movie) for inclusion. Defaults to 35.

    Returns:
    - train_df (DataFrame): The training DataFrame with NaNs replacing test set interactions.
    - y_true (dict): A dictionary where keys are user IDs, and values are lists of true ratings for the test set.
    - test_items (dict): A dictionary where keys are user IDs, and values are lists of movie IDs for the test set.
    """
    
    # Memorybased CF is very expensive, so we need to prune some data requesting min values of overlapping in users and items.
    if min_overlap:
        user_overlap_idx = data.groupby(userId_col_name).count()[rating_col_name][data.groupby(userId_col_name).count()[rating_col_name]\
            > min_interactions_users].index
        item_overlap_idx = data.groupby(movieId_col_name).count()[rating_col_name][data.groupby(movieId_col_name).count()[rating_col_name]\
            > min_interactions_items].index
        
        data = data[data[userId_col_name].isin(user_overlap_idx) & data[movieId_col_name].isin(item_overlap_idx)]

    #test set size for each user
    test_set_size_per_user = data.groupby(userId_col_name).apply(lambda group: max(1, int(len(group) * proportion_test_set)))

    #get the top 20% rows for each user
    def get_test_candidates(group):
        return group.sort_values(time_col_name, ascending=False).head(test_set_size_per_user[group.name])

    #apply the function to each user group
    test_candidates = data.groupby(userId_col_name).apply(get_test_candidates)
    test_candidates.reset_index(drop=True, inplace=True)

    test_items = test_candidates.groupby(userId_col_name)[movieId_col_name].apply(list).to_dict()

    y_true = test_candidates.groupby(userId_col_name)[rating_col_name].apply(list).to_dict()

    #create train_df with NaNs in the rating column for test set positions
    train_df = data.copy()
    
    for u,items in test_items.items():
        train_df.loc[(train_df[userId_col_name] == u) & (train_df[movieId_col_name].isin(items)), rating_col_name] = np.nan
    
    return train_df, y_true, test_items


def clusterize_user(to_clusterize, max_n_clusters, movieId_col_name='movieId', rating_col_name="rating"):
    """
    Clusterizes movies for a given user based on their ratings using KMeans clustering.

    Parameters:
    - to_clusterize (DataFrame): The DataFrame containing user's movie ratings to be clustered.
    - max_n_clusters (int): The maximum number of clusters to be created for the user u. A larger number may result in overfitting.
    - movieId_col_name (str, optional): The name of the movie ID column. Defaults to 'movieId'.
    - rating_col_name (str, optional): The name of the rating column. Defaults to 'rating'.

    Returns:
    - clusterized_df (DataFrame): A DataFrame containing movie IDs, ratings, and cluster labels for the user.
    """
    
    n_clusters_range = range(2, max_n_clusters+1)

    # Dictionary to store silhouette scores for each number of clusters
    sil_scores = {}

    for n_clusters in n_clusters_range:
        # Create and fit KMeans model
        kmeans = KMeans(n_clusters=n_clusters, random_state=1203, n_init = 'auto')
        cluster_labels = kmeans.fit_predict(np.array(to_clusterize[rating_col_name]).reshape(-1, 1))
        
        # Calculate Silhouette Score
        if len(set(cluster_labels))>1:
            sil_score = silhouette_score(np.array(to_clusterize[rating_col_name]).reshape(-1, 1), cluster_labels)
            sil_scores[n_clusters] = sil_score
            # Find the best number of clusters
            best_n_clusters = max(sil_scores, key=sil_scores.get)
    
        # the user rated similarly all the movies that he saw.
        if len(set(cluster_labels))==1:
            best_n_clusters = 1

    # Create and fit KMeans model with the best number of clusters
    best_kmeans = KMeans(n_clusters=best_n_clusters, random_state=1203, n_init = 'auto')
    best_cluster_labels = best_kmeans.fit_predict(np.array(to_clusterize[rating_col_name]).reshape(-1, 1))

    clusterized_df = pd.DataFrame({movieId_col_name: to_clusterize[movieId_col_name].values,
                                    rating_col_name: to_clusterize[rating_col_name].values,  
                                    'cluster_label': best_cluster_labels})

    return clusterized_df




def LOO_split_by_cluster(data, max_n_clusters=3, userId_col_name = "userId", movieId_col_name="movieId", 
           time_col_name = "timestamp", rating_col_name = "rating"):
    
    """
    Leave-One-Out (LOO) split of the input data based on clustering for each user and timestamp.

    This function clusters movies for each user and creates a train-test split for each cluster using LOO
    time sensitive strategy.

    Parameters:
    - data (DataFrame): The input DataFrame containing user-item interactions.
    - max_n_clusters (int, optional): The maximum number of clusters to be created for each user. Defaults to 3.
    - userId_col_name (str, optional): The name of the user ID column. Defaults to "userId".
    - movieId_col_name (str, optional): The name of the movie ID column. Defaults to "movieId".
    - time_col_name (str, optional): The name of the timestamp column. Defaults to "timestamp".
    - rating_col_name (str, optional): The name of the rating column. Defaults to "rating".

    Returns:
    - train_LOO (DataFrame): The training DataFrame with NaNs replacing test set interactions.
    - y_true_LOO (DataFrame): A DataFrame containing the true ratings for the test set.
    - outliers_rating_movies (DataFrame): A DataFrame containing outliers' ratings for movies in clusters with fewer than 2 movies.
    """
    
    
    users = set(data[userId_col_name])
    train_LOO = pd.DataFrame()
    y_true_LOO = pd.DataFrame()
    outliers_rating_movies = pd.DataFrame()
    
    #for each user I build the clusters of items and for each cluster build train_LOO and test_LOO time sensitive
    for u in users:
        to_clusterize = data[(data[userId_col_name]==u) & (data[rating_col_name].notna())][[rating_col_name, movieId_col_name]]
        clusters = clusterize_user(to_clusterize, max_n_clusters)
        
        items_timestamp = data[data[userId_col_name] == u]
        df = pd.merge(clusters, items_timestamp, on=[movieId_col_name, rating_col_name], how='left')
        total_movies_by_cluster = df.groupby('cluster_label')[movieId_col_name].count().to_dict()
        
        # if the cluster has more than 2 movies
        representative_clusters = [cluster for cluster, n_movies in total_movies_by_cluster.items() if n_movies>2]
        df = df[df["cluster_label"].isin(representative_clusters)]
        # get the movies that will be the validation set to fetch the better T0 for each cluster and each user
        max_timestamp_indices = df.groupby('cluster_label')[time_col_name].idxmax()
        col_list = ['cluster_label', userId_col_name, movieId_col_name, rating_col_name, time_col_name]
        y_true_LOO_aux = df.loc[max_timestamp_indices, col_list].sort_values("cluster_label")
        # set np.nan for train_LOO
        y_true_movieid = list(y_true_LOO_aux[movieId_col_name])
        train_df_LOO_aux = df.copy()
        train_df_LOO_aux.loc[train_df_LOO_aux[movieId_col_name].isin(y_true_movieid), rating_col_name] = np.nan

        #################################### OUTLIERS RATING MOVIES ####################################
        no_representative_clusters = [cluster for cluster, n_movies in total_movies_by_cluster.items() if n_movies<=2]
        #if the cluster has less than 2 movies, we 'll use the mean T0 of the user "u".
        #outliers_dict = {cluster:movieId}
        outliers_rating_movies_aux = df[df['cluster_label'].isin(no_representative_clusters)][['cluster_label',movieId_col_name]]\
            .groupby('cluster_label')[movieId_col_name].apply(list)
        
        train_LOO  = pd.concat([train_LOO, train_df_LOO_aux], axis = 0)
        y_true_LOO = pd.concat([y_true_LOO, y_true_LOO_aux] , axis = 0)
        outliers_rating_movies = pd.concat([outliers_rating_movies_aux, train_df_LOO_aux], axis = 0)
        
    return train_LOO, y_true_LOO, outliers_rating_movies


# def train_test_split(data, userId_col_name="userId" , movieId_col_name = 'movieId', 
#                            rating_col_name="rating" , time_col_name = "timestamp", 
#                            proportion_test_set=0.2  , min_overlap = False, 
#                            min_interactions_users=25, min_interactions_items=35):
#     """Divide en train y test, mandando a test los registros mas recientes, a partir de la columna 
#     timestamp COMPLETAR COMPLETAR COMPLETAR

#     Args:
#         data (_type_): _description_
#         userId_col_name (str, optional): _description_. Defaults to "userId".
#         movieId_col_name (str, optional): _description_. Defaults to 'movieId'.
#         rating_col_name (str, optional): _description_. Defaults to "rating".
#         time_col_name (str, optional): _description_. Defaults to "timestamp".
#         proportion_test_set (float, optional): _description_. Defaults to 0.2.
#         min_overlap (bool, optional): _description_. Defaults to False.
#         min_interactions_users (int, optional): _description_. Defaults to 25.
#         min_interactions_items (int, optional): _description_. Defaults to 35.

#     Returns:
#         _type_: _description_
#     """
    
#     user_item_matrix = pd.pivot_table(data,
#                                 index=userId_col_name,
#                                 columns=movieId_col_name,
#                                 values=rating_col_name)

#     # Memorybased CF is very expensive, so we need to prune some data requesting min values of overlapping in users and items.
#     if min_overlap:

#         item_overlap_idx = user_item_matrix.apply(lambda x: np.sum(x.notna()), axis = 0) >= min_interactions_items
#         user_overlap_idx = user_item_matrix.apply(lambda x: np.sum(x.notna()), axis = 1) >= min_interactions_users


#         user_item_matrix = user_item_matrix.loc[user_overlap_idx, item_overlap_idx]
    
    
#     y_true = {}
#     test_items = {}
    
#     train_df = user_item_matrix.copy()
#     test_df = user_item_matrix.copy()
    
#     users = list(user_item_matrix.index)
    
#     for user in users:
#         notna_mask = train_df.loc[user, :].notna()
#         notna_items_idx = list(train_df.loc[user, notna_mask].index)
        
#         # at least 1 item in test for each user
#         test_set_size = max(1, int(len(notna_items_idx)*proportion_test_set))
        
#         unsorted_test_candidates = data.loc[(data['userId'] == user) & (data['movieId'].isin(notna_items_idx))]
        
#         test_set_idx=list(unsorted_test_candidates.sort_values(time_col_name, ascending=True)["movieId"].values[-test_set_size:])
        
#         train_df.loc[user, test_set_idx] = np.nan
#         y_true[user] = test_df.loc[user, test_set_idx].values
#         test_items[user] = list(test_df.loc[user, test_set_idx].index)
    

    
#     return train_df, y_true, test_items




# def train_test_split(data, userId_col_name="userId" , movieId_col_name = 'movieId', 
#                            rating_col_name="rating" , time_col_name = "timestamp", 
#                            proportion_test_set=0.2  , min_overlap = False, 
#                            min_interactions_users=25, min_interactions_items=35):
#     """
#     Splits the input data into training and testing sets for collaborative filter time sensitive
    
#     Parameters:
#     - data (pd.DataFrame): The input DataFrame containing user-item interactions.
#     - userId_col_name (str): The name of the column representing user IDs.
#     - movieId_col_name (str): The name of the column representing movie IDs.
#     - rating_col_name (str): The name of the column representing ratings.
#     - time_col_name (str): The name of the column representing timestamps.
#     - proportion_test_set (float): The proportion of items to be included in the test set for each user.

#     Returns:
#     - train_df (pd.DataFrame): The training set represented as a user-item matrix.
#     - y_true (dict): A dictionary containing true ratings for each user in the test set.
#     - test_items (dict): A dictionary containing the list of items in the test set for each user.
#     """
    
#     user_item_matrix = pd.pivot_table(data,
#                                 index=userId_col_name,
#                                 columns=movieId_col_name,
#                                 values=rating_col_name)

#     # Memorybased CF is very expensive, so we need to prune some data requesting min values of overlapping in users and items.
#     if min_overlap:

#         item_overlap_idx = user_item_matrix.apply(lambda x: np.sum(x.notna()), axis = 0) >= min_interactions_items
#         user_overlap_idx = user_item_matrix.apply(lambda x: np.sum(x.notna()), axis = 1) >= min_interactions_users


#         user_item_matrix = user_item_matrix.loc[user_overlap_idx, item_overlap_idx]
    
    
#     y_true = {}
#     test_items = {}
    
    
#     train_df = user_item_matrix.copy()
#     test_df = user_item_matrix.copy()
    
#     users = list(user_item_matrix.index)
    
#     for user in users:
#         notna_mask = train_df.loc[user, :].notna()
#         notna_items_idx = list(train_df.loc[user, notna_mask].index)
        
#         # at least 1 item in test for each user
#         test_set_size = max(1, int(len(notna_items_idx)*proportion_test_set))
        
#         unsorted_test_candidates = data.loc[(data['userId'] == user) & (data['movieId'].isin(notna_items_idx))]
        
#         test_set_idx=list(unsorted_test_candidates.sort_values(time_col_name, ascending=True)["movieId"].values[-test_set_size:])
#         train_df.loc[user, test_set_idx] = np.nan
#         y_true[user] = test_df.loc[user, test_set_idx].values
#         test_items[user] = list(test_df.loc[user, test_set_idx].index)
    
#     return train_df, y_true, test_items


# def train_test_split(user_item_matrix, proportion_test_set=0.2, random_seed=1203):
#     """
#     Split the user-item interaction matrix into training and test sets for each user.

#     Parameters:
#     - user_item_matrix (pd.DataFrame): Pandas DataFrame representing the user-item interaction matrix,
#                                        where rows are users, columns are items, and values indicate
#                                        user-item interactions (e.g., ratings).
#     - proportion_test_set (float, optional): Proportion of items to include in the test set for each user.
#                                             Default is 0.2 (20% of items in the test set).
#     - random_seed (int, optional): Seed for random number generation for reproducibility. Default is 1203.

#     Returns:
#     - tuple: A tuple containing:
#         - pd.DataFrame: Training set with a subset of items set to NaN for each user.
#         - dict: Dictionary where keys are user IDs, and values are arrays of true ratings from the test set.
#         - dict: Dictionary where keys are user IDs, and values are lists of items included in the test set.

#     Notes:
#     - The user_item_matrix should be a Pandas DataFrame.
#     - The function randomly selects a subset of items for the test set for each user, setting them to NaN in the training set.
#     - The true ratings from the test set are stored in a dictionary for each user.
#     - The items included in the test set are stored in a dictionary for each user.
#     - Proportion of items to include in the test set is determined by the 'proportion_test_set' parameter.
#     """
    
#     y_true = {}
#     test_items = {}
    
#     random.seed(random_seed)
        
#     train_df = user_item_matrix.copy()
#     test_df = user_item_matrix.copy()
    
#     users = list(user_item_matrix.index)
    
#     for user in users:
#         notna_mask = train_df.loc[user, :].notna()
#         notna_items_idx = list(train_df.loc[user, notna_mask].index)
        
#         # at least 1 item in test for each user
#         test_set_size = max(1, int(len(notna_items_idx)*proportion_test_set))
#         test_set_idx = random.sample(notna_items_idx, test_set_size)
        
#         train_df.loc[user, test_set_idx] = np.nan
#         y_true[user] = test_df.loc[user, test_set_idx].values
#         test_items[user] = list(test_df.loc[user, test_set_idx].index)
    
#     return train_df, y_true, test_items
