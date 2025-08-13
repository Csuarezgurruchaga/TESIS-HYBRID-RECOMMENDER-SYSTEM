import numpy as np
from tqdm import tqdm


def mean_absolute_error(y_true, y_pred):
    """
    Calculate Mean Absolute Error (MAE) between true and predicted values.

    Parameters:
    - y_true (array-like): Array or list of true values.
    - y_pred (array-like): Array or list of predicted values.

    Returns:
    - float: Mean Absolute Error.
    """ 
    if type(y_true) != float and type(y_pred) != float:
        assert len(y_true) == len(y_pred), "Input arrays must have the same length."
    
    return np.mean(np.abs(np.subtract(y_true, y_pred)))

def normalized_mean_absolute_error(y_true, y_pred, rmax, rmin):
    """
    Calculate Normalized Mean Absolute Error (NMAE) between true and predicted values.

    Parameters:
    - y_true (array): Array or list of true values.
    - y_pred (array): Array or list of predicted values.
    - rmax (scalar): Max rating value.
    - rmin (scalar): Min rating value.

    Returns:
    - float: Normalized Mean Absolute Error.
    """
    assert len(y_true) == len(y_pred), "Input arrays must have the same length."
    
    mae = np.mean(np.abs(np.subtract(y_true, y_pred)))
    normalization = np.substract(rmax, rmin)
    
    nmae = mae/normalization
    
    return nmae
    

def mean_squared_error(y_true, y_pred):
    """
    Calculate Mean Squared Error (MSE) between true and predicted values.

    Parameters:
    - y_true (array-like): Array or list of true values.
    - y_pred (array-like): Array or list of predicted values.

    Returns:
    - float: Mean Squared Error.
    """
    if type(y_true) != float and type(y_pred) != float:
        assert len(y_true) == len(y_pred), "Input arrays must have the same length."
    return np.mean(np.square(np.subtract(y_true, y_pred)))

def root_mean_squared_error(y_true, y_pred):
    """
    Calculate Root Mean Squared Error (RMSE) between true and predicted values.

    Parameters:
    - y_true (array-like): Array or list of true values.
    - y_pred (array-like): Array or list of predicted values.

    Returns:
    - float: Root Mean Squared Error.
    """
    assert len(y_true) == len(y_pred), "Input arrays must have the same length."
    return np.sqrt(mean_squared_error(y_true, y_pred))


# Una aclaracion sobre recall at k es que suele penalizar mas al modelo, al tener valores "k", mas pequeños y tiende a favorecer
# mas al modelo al aumentar el valor de "k", lo cual puede llevar a una lectura erronea de la performance del modelo. 

# Otro problema que tiene es que no toma en cuenta el orden en que fueron devueltos los items en la recomendacion. La metrica asigna el mismo 
# resultado, sin importar el orden del Top N. 

def calculate_coverage(filter_obj, all_users, all_items, k=10):
    """
    Calculate user and item coverage for a recommender system.
    
    Parameters:
    - filter_obj: Recommender model with recommend method
    - all_users: List of all users in the system
    - all_items: List of all items in the system
    - k: Number of recommendations to generate per user
    
    Returns:
    - dict: Coverage metrics including user and item coverage percentages
    """
    recommended_items = set()
    users_with_recs = 0
    
    for user in all_users:
        try:
            recs = filter_obj.recommend(user, k=k)
            if recs:
                users_with_recs += 1
                recommended_items.update(recs)
        except:
            continue
    
    user_coverage = (users_with_recs / len(all_users)) * 100
    item_coverage = (len(recommended_items) / len(all_items)) * 100
    
    return {
        'user_coverage': round(user_coverage, 2),
        'item_coverage': round(item_coverage, 2),
        'unique_items_recommended': len(recommended_items)
    }

def get_users_threshold(y_true, y_pred, percentile=80):
    """
    Calculate the "HIT threshold ratings" for users based on the specified percentile of predicted ratings.

    Parameters:
    - y_true (dict): Dictionary containing true ratings for each user.
    - y_pred (dict): Dictionary containing predicted ratings for each user.
    - percentile (int, optional): The percentile used to calculate the HIT rating threshold. Default is 80.

    Returns:
    - user_hit_threshold (dict): Dictionary mapping user IDs to their respective rating thresholds.
    """    
    user_hit_threshold = {}
    for u in y_true.keys():
        rating_predictions= list(y_pred[u].values())
        # we set a HIT when the rating of that item is above from the percentile 80th.
        user_hit_threshold[u] = np.nanpercentile(rating_predictions, percentile)
    return user_hit_threshold

def get_actual_relevant_items(ratings, threshold):
    """
    Get the items with ratings above a specified threshold.

    Parameters:
    - ratings (dict): Dictionary containing item ratings.
    - threshold (float): The threshold rating.

    Returns:
    - relevant_items (list): List of items with ratings above the threshold.
    """
    relevant_items = [item for item, rating in ratings.items() if rating > threshold]
    return relevant_items

def mean_average_precision_at_k(y_true, y_pred, model, k=10, verbose=True):
    """
    Calculate the Mean Average Precision (MAP) at a specified value of k for a recommendation model.
    precision = number of recommendations that are relevant / number of items that are recommended
    
    Parameters:
    - y_true (dict): Dictionary containing true ratings for each user.
    - y_pred (dict): Dictionary containing predicted ratings for each user.
    - model (object): The trained recommendation model.
    - k (int, optional): The number of recommendations to consider. Default is 10.

    Returns:
    - map (float): Mean Average Precision at the specified value of k.
    """
    relevant_recommendations = {}
    relevant_items = {}
    precision_at_k = {}
    user_hit_threshold = get_users_threshold(y_true, y_pred)
    
    for u in tqdm(y_true.keys()):     
        model_recommendations = model.recommend(u, y_true, k)
        
        relevant_recommendations[u] = get_actual_relevant_items(y_pred[u], user_hit_threshold[u])
        relevant_items = get_actual_relevant_items(y_true[u], user_hit_threshold[u])
        if relevant_items == [] & verbose:
            print(f"Cannot calculate precision@{k} for user {u}. No relevant items in the test set for this user. This user is not considered for the metric.")
        count_relevant_recom_items = len(set(model_recommendations) & set(relevant_items))
        precision_at_k[u] = count_relevant_recom_items / len(model_recommendations)
        map=np.nanmean(list(precision_at_k.values()))
    return map


def mean_average_recall_at_k(y_true, y_pred, model, k=10, verbose = True):
    """
    Calculate the Mean Average Recall (MAR) at a specified value of k for a recommendation model.

    Parameters:
    - y_true (dict): Dictionary containing true ratings for each user.
    - y_pred (dict): Dictionary containing predicted ratings for each user.
    - model (object): The recommendation model with a 'recommend' method.
    - k (int, optional): The number of recommendations to consider. Default is 10.

    Returns:
    - recall_at_k (dict): Dictionary mapping user IDs to their respective recall values at the specified value of k.
    """
    user_hit_threshold = get_users_threshold(y_true, y_pred)
    recall_at_k = {}

    for u in y_true.keys():
        model_recommendations = model.recommend(u, y_true, k)
        relevant_items = get_actual_relevant_items(y_true[u], user_hit_threshold[u])
        count_relevant_recom_items = len(set(model_recommendations) & set(relevant_items))
        try:
            recall_at_k[u] = count_relevant_recom_items / len(relevant_items)
        except ZeroDivisionError:
            if verbose:
                print(f"Cannot calculate recall@{k} for user {u}. No relevant items in the test set for this user. This user is not considered for the metric.")
                
                
def f1_score(MAP, MAR):
    f_beta = (2 * MAP * MAR) / (MAP + MAR)
    return round(f_beta, 2)



"""
    If you have many users, you can calculate the F1 score for each user individually and then take the average to obtain a summary metric across all users. This approach is known as user-based F1.

Here's how you can do it:

    Calculate Precision, Recall, and F1 for Each User:
        For each user, calculate precision, recall, and F1 score based on the relevant items and recommendations for that specific user.

    User-Based F1 Score:
        Once you have individual F1 scores for all users, you can compute the average F1 score across all users. This provides an overall evaluation of your recommender system's performance.

    User-Based F1=∑i=1nF1inUser-Based F1=n∑i=1n​F1i​​

    Where:
        nn is the total number of users.
        F1iF1i​ is the F1 score for the ii-th user.

This user-based F1 score gives you an aggregated measure of how well your recommender system is performing across a diverse set of users. It considers both precision and recall on a per-user basis, helping you understand the system's overall effectiveness in providing relevant recommendations to different users.
    
"""

def evaluate_recommender(recommender, test_data):
    """
    Evaluate a recommender system using multiple metrics
    
    Parameters:
    -----------
    recommender : object
        Fitted recommender system with compute_prediction method
    test_data : dict
        Dictionary containing test data {user_id: {item_id: rating}}
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    y_pred = {}
    # users_mean = {}
    metrics = {
        'mae': {},
        'mse': {},
        'rmse': {}
    }
    
    for user_id, items_ratings in tqdm(test_data.items(), desc="Evaluating"):
        user_predictions = {}
        
        for item_id, _ in items_ratings.items():
            try:
                if isinstance(recommender.compute_prediction(user_id, item_id), tuple):
                    prediction, user_mean = recommender.compute_prediction(user_id, item_id)
                    # users_mean[user_id] = user_mean
                else:
                    prediction = recommender.compute_prediction(user_id, item_id)
            except ValueError:
                print("check ValueError")

                
            user_predictions[item_id] = prediction
            
        # Store predictions
        y_pred[user_id] = user_predictions
        
        # Calculate metrics
        true_ratings = list(items_ratings.values())
        pred_ratings = list(user_predictions.values())
        
        metrics['mae'][user_id] = mean_absolute_error(true_ratings, pred_ratings)
        metrics['mse'][user_id] = mean_squared_error(true_ratings, pred_ratings)
        metrics['rmse'][user_id] = root_mean_squared_error(true_ratings, pred_ratings)
    
    # Calculate average metrics
    avg_metrics = {
        'avg_mae': round(np.nanmean(list(metrics['mae'].values())), 3),
        'avg_mse': round(np.nanmean(list(metrics['mse'].values())), 3),
        'avg_rmse': round(np.nanmean(list(metrics['rmse'].values())), 3)
    }
    
    return {
        'predictions': y_pred,
        # 'user_means': users_mean,
        'metrics': metrics,
        'avg_metrics': avg_metrics
    }