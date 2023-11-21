import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform


def leave_one_last_item(df, uid_col_name, proportion_test_items=0.1):
    """
    Splits a DataFrame into a test set containing a proportion of the last items for each user and a training set.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing ratings data.
        uid_col_name (str): The name of the column representing user IDs.
        proportion_test_items (float, optional): The proportion of last items to include in the test set. Default is 10%.

    Returns:
        train (pd.DataFrame): DataFrame for training, containing all the items that are not in test.
        test (pd.DataFrame): DataFrame for testing, containing the last proportion_test_items items for each user.
    """

    test = pd.DataFrame()

    for _, data_by_user in df.groupby(uid_col_name):
        n_test_items = int(df.shape[0]*proportion_test_items)
        test = pd.concat([test, data_by_user.iloc[-n_test_items:]])

    train = df[~df.index.isin(test.index)]
    
    return train, test

    

def adjusted_cosine_similarity(user_item_matrix, itemA, itemB, min_overlap=5):
    """
    Calculate the adjusted cosine similarity of the items for a user-item matrix.

    Parameters:
    - user_item_matrix (pd.DataFrame): User-item matrix.
    - itemA (int): Index of the first item.
    - itemB (int): Index of the second item.
    - min_overlap (int): Minimum overlap of users who have rated both items.

    Returns:
    - float: Adjusted cosine similarity between the two items.
    """

    mu_users = user_item_matrix.mean(axis=1)

    # Extract ratings for the specified items
    result = user_item_matrix.loc[:, [itemA, itemB]]

    # Normalize ratings by subtracting mean for each user
    normalized = result.sub(mu_users, axis=0)

    # Filter out rows with NaN values
    non_nan_mask = ~normalized.isnull().any(axis=1)
    normalized_2 = normalized[non_nan_mask]

    if normalized_2.shape[0] > min_overlap:
        # Calculate cosine similarity for the normalized ratings
        items_similarity_scalar = 1 - pdist(normalized_2.T, 'cosine')
        return items_similarity_scalar[0]

    # Return None if the minimum overlap condition is not met
    return None




def compute_item_similarity(user_item_matrix):
    """
    Computes item similarity based on the adjusted cosine similarity metric using the 
    Amazon technic described in his paper.

    Parameters:
    - user_item_matrix (pd.DataFrame): A 2D matrix representing user-item interactions,
      where rows correspond to users, and columns correspond to items. The values
      indicate user-item interactions (e.g., purchase history or ratings for movies).

    Returns:
    - dict: A nested dictionary representing item similarity. The outer dictionary
      has item indices as keys, and the associated values are inner dictionaries.
      Inner dictionaries have other item indices as keys and adjusted cosine similarity
      values as values.

    Note:
    - The adjusted cosine similarity is computed for each pair of items based on the
      user-item matrix. Similarity is calculated only if at least one user has interacted
      with both items.
    - The negative values of similarity are ignored.
    """
    
    assert isinstance(user_item_matrix, pd.DataFrame), "user_item_matrix should be a Pandas DataFrame."
    
    idx_items = list(user_item_matrix.columns)
    similarity_dict = {}
    
    for item1 in tqdm(idx_items, desc="Computing Similarities", unit="item"):
        similarity_dict[item1] = {}
        for item2 in idx_items:
            # the customer purchased both items?
            if np.any(np.logical_and(user_item_matrix.loc[:, item1], user_item_matrix.loc[:, item2])): 
                
                # calculate the adjusted cosine similarity between the two items
                similarity = adjusted_cosine_similarity(user_item_matrix, item1, item2) 
                
                # we consider only the positive values of similarity
                if similarity is not None and similarity >= 0:
                    similarity_dict[item1][item2] = similarity
                    
    #sort the dict with the values of similarity
    sorted_sim_dict = {k: {inner_key: inner_value for inner_key, inner_value in sorted(v.items(), key=lambda item: item[1], reverse=True) 
                           if inner_key != k} for k, v in similarity_dict.items()}

    return sorted_sim_dict

def compute_neighbors(u, i, user_item_matrix, sim, n_neighbors=35):
    """
    Compute the top neighbors of item i for a given user u based on item similarity.

    Parameters:
    - u: User index
    - i: Item index for which neighbors are to be computed
    - sim: Dictionary of the similarity between items
    - n_neighbors: Number of neighbors to retrieve (default is 35)
    - user_item_matrix: User-item interaction matrix

    Returns:
    - neighbors_of_i: Dictionary containing the top neighbors of item i for user u
    """
    
    if type(user_item_matrix) != pd.DataFrame:
        user_item_matrix = pd.DataFrame(user_item_matrix)
    
    sim_keys = sim[i].keys()
    non_nan_mask = None
    try:
        non_nan_mask = user_item_matrix.loc[u, :].notna()
    except IndexError:
        print(f"Error: el usuario no se encuentra registrado")
        
    if non_nan_mask is not None:
        non_nan_idx = non_nan_mask[non_nan_mask].index
        j = list(set(sim_keys) & set(list(non_nan_idx)))

        # Create a dictionary with keys from j and values from sim[i]
        sorted_similarities = {k: sim[i][k] for k in j}
        # Sort the dictionary based on values in descending order
        sorted_similarities = dict(sorted(sorted_similarities.items(), key=lambda x: x[1], reverse=True))
        # Select the top neighbors values from the sorted dictionary
        neighbors_of_i = dict(list(sorted_similarities.items())[:n_neighbors])

        return neighbors_of_i
    
def compute_prediction(u, i, user_item_matrix, sim, n_neighbors=35):
    """
    Compute the predicted rating for a given user and item.

    Parameters:
    - u (int): User index.
    - i (int): Item index.
    - user_item_matrix (pd.DataFrame): User-item matrix.
    - sim (pd.DataFrame): Item-item similarity matrix.
    - n_neighbors (int): Number of neighbors to consider.

    Returns:
    - float: Predicted rating for the user-item pair.
    """
    
    if type(user_item_matrix) != pd.DataFrame:
        user_item_matrix = pd.DataFrame(user_item_matrix)
    
    neighbors = compute_neighbors(u, i, user_item_matrix, sim, n_neighbors)
    user_ratings_mean = np.mean(user_item_matrix, axis=1)
    norm_rating = user_item_matrix.loc[u, :].dropna()[list(neighbors.keys())] - user_ratings_mean[u]
    num = sum(np.array(list(neighbors.values())) * np.array(norm_rating))
    denom = sum(list(neighbors.values()))
    try:
        rating_hat = user_ratings_mean[u] + num / denom
    except ZeroDivisionError:
        print("Error: No hay items vecinos para el item a predecir el ranking")
        return np.nan

    return rating_hat

def compute_recommendations(u, user_item_matrix, dict_map, sim, n_recommendations = 10):
    """
    Generate top n recommendations for a given user based on item collaborative filtering.

    Parameters:
    - u (int): User ID for whom recommendations are to be generated.
    - user_item_matrix (pd.DataFrame): Pandas DataFrame representing the user-item interaction matrix,
                                       where rows are users, columns are items, and values indicate
                                       user-item interactions (e.g., ratings).
    - dict_map (dict): A dictionary mapping item indices to their corresponding item identifiers/names.
    - sim (pd.DataFrame): Similarity matrix between items (e.g., item-item similarity).
    - n_recommendations (int, optional): Number of recommendations to generate for the user. Default is 10.

    Returns:
    - list: A list of recommended item identifiers/names for the given user.

    Notes:
    - The user_item_matrix should be a Pandas DataFrame.
    - The function uses item collaborative filtering to predict user-item interactions and generate recommendations.
    - The final recommendations are returned as a list of item identifiers/names using the provided dict_map.
    """

    assert isinstance(user_item_matrix, pd.DataFrame), "user_item_matrix should be a Pandas DataFrame."
    
    predictions = {}
    items_to_predict=list(user_item_matrix.loc[u, user_item_matrix.loc[u,:].isna()].index)
    for i in items_to_predict:
        predictions[i]=compute_prediction(u, i, user_item_matrix, sim)
    
    all_recommendations = dict(sorted(predictions.items(), key=lambda item: item[1], reverse= True))
    n_recommendations = [k for k,_ in list(all_recommendations.items())[:n_recommendations]]
    
    rec = [dict_map[item] for item in n_recommendations]
    
    return rec