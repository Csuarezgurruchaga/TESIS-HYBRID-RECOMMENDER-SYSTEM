import random
import numpy as np
import pandas as pd

def train_test_split(user_item_matrix, proportion_test_set=0.2, random_seed=1203):
    """
    Split the user-item interaction matrix into training and test sets for each user.

    Parameters:
    - user_item_matrix (pd.DataFrame): Pandas DataFrame representing the user-item interaction matrix,
                                       where rows are users, columns are items, and values indicate
                                       user-item interactions (e.g., ratings).
    - proportion_test_set (float, optional): Proportion of items to include in the test set for each user.
                                            Default is 0.2 (20% of items in the test set).
    - random_seed (int, optional): Seed for random number generation for reproducibility. Default is 1203.

    Returns:
    - tuple: A tuple containing:
        - pd.DataFrame: Training set with a subset of items set to NaN for each user.
        - dict: Dictionary where keys are user IDs, and values are arrays of true ratings from the test set.
        - dict: Dictionary where keys are user IDs, and values are lists of items included in the test set.

    Notes:
    - The user_item_matrix should be a Pandas DataFrame.
    - The function randomly selects a subset of items for the test set for each user, setting them to NaN in the training set.
    - The true ratings from the test set are stored in a dictionary for each user.
    - The items included in the test set are stored in a dictionary for each user.
    - Proportion of items to include in the test set is determined by the 'proportion_test_set' parameter.
    """
    
    y_true = {}
    test_items = {}
    
    random.seed(random_seed)
        
    train_df = user_item_matrix.copy()
    test_df = user_item_matrix.copy()
    
    users = list(user_item_matrix.index)
    
    for user in users:
        notna_mask = train_df.loc[user, :].notna()
        notna_items_idx = list(train_df.loc[user, notna_mask].index)
        
        # at least 1 item in test for each user
        test_set_size = max(1, int(len(notna_items_idx)*proportion_test_set))
        test_set_idx = random.sample(notna_items_idx, test_set_size)
        
        train_df.loc[user, test_set_idx] = np.nan
        y_true[user] = test_df.loc[user, test_set_idx].values
        test_items[user] = list(test_df.loc[user, test_set_idx].index)
    
    return train_df, y_true, test_items
