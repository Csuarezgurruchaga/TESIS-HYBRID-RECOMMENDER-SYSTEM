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


def split_cases(train_LOO, test_items_LOO, sim):
    """
    Assigns different cases/scenarios ('sim_count=0', 'sim_count=1', 'sim_count>1') to each row in the training DataFrame based on
    the count of similar items in the given similarity dictionary.

    Parameters:
        train_LOO (pd.DataFrame): Training DataFrame containing user-item interactions and cluster labels.
        test_items_LOO (dict): Dictionary mapping user IDs to lists of test items, grouped by item clusters.
        sim (dict): Dictionary containing item similarities.

    Returns:
        train_LOO_cased (pd.DataFrame): Training DataFrame with an additional 'case' column indicating the scenario
                        based on the count of similar items for each user and cluster.
        test_items_LOO_cased (dict): Dictionary mapping user IDs to dictionaries of test items,
                             grouped by item clusters, with only the movieId of the clusters of items with sim_count>1.
    """
    
    train_LOO_cased = train_LOO.copy()
    train_LOO_cased['case'] = None
    test_items_LOO_cased = {}

    for key, values in test_items_LOO.items():
        test_items_LOO_cased[key] = {i: value for i, value in enumerate(values)}
    
    for u, cluster_test_items_LOO in test_items_LOO.items():
        for cluster, test_item in enumerate(cluster_test_items_LOO):
            train_items_user_cluster = train_LOO_cased[(train_LOO_cased['cluster_label'] == cluster) & 
                                                       (train_LOO_cased['userId'] == u)]['movieId']
            sim_count = len(set(sim[test_item]) & set(train_items_user_cluster))
            
            if sim_count == 0:
                train_LOO_cased.loc[(train_LOO_cased['cluster_label'] == cluster) & (train_LOO_cased['userId'] == u), "case"] = 'sim_count=0'
                # remove the test_items_LOO that not have any sim value between items inside the cluster and the item to predict the rating
                del test_items_LOO_cased[u][cluster]
            elif sim_count == 1:
                train_LOO_cased.loc[(train_LOO_cased['cluster_label'] == cluster) & (train_LOO_cased['userId'] == u), "case"] = 'sim_count=1'
                # remove the test_items_LOO that have only one sim value between items inside the cluster and the item to predict the rating
                del test_items_LOO_cased[u][cluster]
            elif sim_count > 1:
                train_LOO_cased.loc[(train_LOO_cased['cluster_label'] == cluster) & (train_LOO_cased['userId'] == u), "case"] = 'sim_count>1'
    
    return train_LOO_cased, test_items_LOO_cased



            
# PSUDOCODE:
# FOR each user (u) in test_items_LOO:
#     FOR each item (test_item) in the list associated with user (u):
#         Get all movie IDs (train_items_user_cluster) for user (u) in the same cluster (cluster) as test_item from train_LOO
#         Count the number of similar items (sim_count) between test_item and train_items_user_cluster
#         IF sim_count is equal to 0:
#             Set the case column of the corresponding row in train_LOO to "sim_count=0"
#         ELSE IF sim_count is equal to 1:
#             Set the case column of the corresponding row in train_LOO to "sim_count=1"
#         ELSE:
#             Set the case column of the corresponding row in train_LOO to "sim_count>1"

