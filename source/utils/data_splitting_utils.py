import random
import numpy as np
import pandas as pd



""" 
pros y contras sobre este metodo de split

1er tecnica: uso la historia de las ultimas 20 peliculas
2da tecnica: una funcion q baje el peso a medida q avanza el tiempo.


comparar teniendo en cuenta el tiempo y no, para mostrar el peso de tener en cuenta el tiempo.
"""

def train_test_split(data, userId_col_name="userId", movieId_col_name = 'movieId', 
                           rating_col_name = "rating",time_col_name = "timestamp", 
                           proportion_test_set=0.2):
    """
    Splits the input data into training and testing sets for collaborative filtering timestamp-aware
    
    Parameters:
    - data (pd.DataFrame): The input DataFrame containing user-item interactions.
    - userId_col_name (str): The name of the column representing user IDs.
    - movieId_col_name (str): The name of the column representing movie IDs.
    - rating_col_name (str): The name of the column representing ratings.
    - time_col_name (str): The name of the column representing timestamps.
    - proportion_test_set (float): The proportion of items to be included in the test set for each user.

    Returns:
    - train_df (pd.DataFrame): The training set represented as a user-item matrix.
    - y_true (dict): A dictionary containing true ratings for each user in the test set.
    - test_items (dict): A dictionary containing the list of items in the test set for each user.
    """
    
    user_item_matrix = pd.pivot_table(data,
                                index=userId_col_name,
                                columns=movieId_col_name,
                                values=rating_col_name)

    
    
    y_true = {}
    test_items = {}
    
        
    train_df = user_item_matrix.copy()
    test_df = user_item_matrix.copy()
    
    users = list(user_item_matrix.index)
    
    for user in users:
        notna_mask = train_df.loc[user, :].notna()
        notna_items_idx = list(train_df.loc[user, notna_mask].index)
        
        # at least 1 item in test for each user
        test_set_size = max(1, int(len(notna_items_idx)*proportion_test_set))
        
        unsorted_test_candidates = data.loc[(data['userId'] == user) & (data['movieId'].isin(notna_items_idx))]
        
        test_set_idx=list(unsorted_test_candidates.sort_values("timestamp", ascending = True)["movieId"].values[-test_set_size:])
        train_df.loc[user, test_set_idx] = np.nan
        y_true[user] = test_df.loc[user, test_set_idx].values
        test_items[user] = list(test_df.loc[user, test_set_idx].index)
    
    return train_df, y_true, test_items


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
