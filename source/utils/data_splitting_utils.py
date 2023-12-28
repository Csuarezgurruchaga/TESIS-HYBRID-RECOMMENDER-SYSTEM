import numpy as np
import pandas as pd
from utils.utils import clusterize_user

""" 
pros y contras sobre este metodo de split

1er tecnica: uso la historia de las ultimas 20 peliculas
2da tecnica: una funcion q baje el peso a medida q avanza el tiempo.


comparar teniendo en cuenta el tiempo y no, para mostrar el peso de tener en cuenta el tiempo.
"""

def train_test_split(data, userId_col_name="userId", movieId_col_name='movieId',
                     rating_col_name="rating", time_col_name="timestamp",
                     proportion_test_set=0.2, min_overlap=False,
                     min_interactions_users=25, min_interactions_items=35):  #25 35
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
    - test (dict): A nested dict where the keys are the user IDs and the item IDs and the value is the true rating assigned by the user.
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
    
    test = {}
    for user, items in test_items.items():
        ratings = y_true.get(user, [])
        test[user] = {item: rating for item, rating in zip(items, ratings)}
        
    return train_df, test
    



