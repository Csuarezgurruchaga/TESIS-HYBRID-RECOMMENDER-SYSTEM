import random
import numpy as np
import pandas as pd

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
