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


def clusterize_user(u, to_clusterize, max_n_clusters, refine_n_clusters = True):
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
    
    if refine_n_clusters:
        n_clusters_range = range(2, max_n_clusters+1)
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
    best_kmeans = KMeans(n_clusters = max_n_clusters, random_state=1203, n_init = 'auto')
    best_cluster_labels = best_kmeans.fit_predict(np.array(to_clusterize['rating']).reshape(-1, 1))

    clusterized_df = pd.DataFrame({'movieId': to_clusterize['movieId'].values,
                                    'rating': to_clusterize['rating'].values,  
                                    'cluster_label': best_cluster_labels})

    return clusterized_df






def LOO_split_by_cluster(data, max_n_clusters=3, userId_col_name = "userId", movieId_col_name="movieId", 
           time_col_name = "timestamp", rating_col_name = "rating"):
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
    
    users = set(train_df['userId'])
    train_LOO = pd.DataFrame()
    y_true_LOO_df = pd.DataFrame()
        
    #for each user i build the clusters of items and for each cluster build train_LOO and test_LOO time sensitive
    for u in users:
        to_clusterize = data[(data[userId_col_name]==u) & (data[rating_col_name].notna())][[rating_col_name, movieId_col_name]]
        clusters=clusterize_user(u, to_clusterize, max_n_clusters, refine_n_clusters = True)

        
        items_timestamp = data[data[userId_col_name] == u]
        df = pd.merge(clusters, items_timestamp, on=[movieId_col_name, rating_col_name], how='left')
        # get the movies that will be the validation set to fetch the better T0 for each cluster and each user
        max_timestamp_movies_idx = df.groupby('cluster_label')[time_col_name].idxmax()
        col_list = [userId_col_name, movieId_col_name, rating_col_name, time_col_name, 'cluster_label']
        y_true_LOO_aux = df.loc[max_timestamp_movies_idx, col_list].sort_values("cluster_label")
        # set np.nan for train_LOO
        y_true_LOO_movieid=list(y_true_LOO_aux[movieId_col_name])
        train_df_LOO_aux = df.copy()
        train_df_LOO_aux.loc[train_df_LOO_aux[movieId_col_name].isin(y_true_LOO_movieid), rating_col_name] = np.nan
        
        train_LOO     = pd.concat([train_LOO, train_df_LOO_aux])
        y_true_LOO_df = pd.concat([y_true_LOO_df, y_true_LOO_aux]) 
    
    y_true_LOO_by_cluster = y_true_LOO_df.groupby(userId_col_name)[rating_col_name].apply(list).to_dict()
    test_items_LOO = y_true_LOO_df.groupby(userId_col_name)[movieId_col_name].apply(list).to_dict()
    
    return train_LOO, y_true_LOO_by_cluster, test_items_LOO