import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans


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


def split_scenarios(train_LOO, test_items_LOO, sim):
    """
    Assigns scenarios ('sim_count=0', 'sim_count=1', 'sim_count>1') to each row in the training DataFrame based on
    the count of similar items in the given similarity dict.
          
    Parameters:
        train_LOO (pd.DataFrame): Training DataFrame containing user-item interactions and cluster labels.
        test_items_LOO (dict): Dictionary mapping user IDs to lists of test items, grouped by item clusters.
        sim (dict): Dictionary containing item similarities.

    Returns:
        pd.DataFrame: Updated training DataFrame with an additional 'scenario' column indicating the scenario
                      based on the count of similar items for each user and cluster.
    """
    
    for u, cluster_test_items_LOO in test_items_LOO.items():
        for cluster, test_item in enumerate(cluster_test_items_LOO):
            # Extract cluster items for the user
            train_items_user_cluster = train_LOO[(train_LOO['cluster_label'] == cluster) & (train_LOO['userId'] == u)]['movieId']
            
            # Calculate the count of similar items
            sim_count = len(set(sim[test_item]) & set(train_items_user_cluster))

            # Assign scenarios based on sim_count
            if sim_count == 0:
                # Cold-Star Problem
                train_LOO.loc[(train_LOO['cluster_label'] == cluster) & (train_LOO['userId'] == u), "escenario"] = 'sim_count=0'
            elif sim_count == 1:
                # Only one similarity value
                train_LOO.loc[(train_LOO['cluster_label'] == cluster) & (train_LOO['userId'] == u), "escenario"] = 'sim_count=1'
            elif sim_count > 1:
                # Many similarity values
                train_LOO.loc[(train_LOO['cluster_label'] == cluster) & (train_LOO['userId'] == u), "escenario"] = 'sim_count>1'

    return train_LOO
            
            
# PSUDOCODE:
# FOR each user (u) in test_items_LOO:
#     FOR each item (test_item) in the list associated with user (u):
#         Get all movie IDs (train_items_user_cluster) for user (u) in the same cluster (cluster) as test_item from train_LOO
#         Count the number of similar items (sim_count) between test_item and train_items_user_cluster
#         IF sim_count is equal to 0:
#             Set the "escenario" of the corresponding row in train_LOO to "sim_count=0"
#         ELSE IF sim_count is equal to 1:
#             Set the "escenario" of the corresponding row in train_LOO to "sim_count=1"
#         ELSE:
#             Set the "escenario" of the corresponding row in train_LOO to "sim_count>1"



def split_escenarios(train_LOO, test_items_LOO, sim):
    """
    Assigns scenarios ('sim_count=0', 'sim_count=1', 'sim_count>1') to each row in the training DataFrame based on
    the count of similar items in the given similarity matrix.

    Parameters:
        train_LOO (pd.DataFrame): Training DataFrame containing user-item interactions and cluster labels.
        test_items_LOO (dict): Dictionary mapping user IDs to lists of test items, grouped by item clusters.
        sim (dict): Similarity matrix containing item similarities.

    Returns:
        pd.DataFrame: Updated training DataFrame with an additional 'escenario' column indicating the scenario
                      based on the count of similar items for each user and cluster.
    """


