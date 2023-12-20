import numpy as np


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


# y true will be the idx of the true hit items for user u
# y pred will be the idx of the predicted hit items for user u
def recall(y_true, y_pred, k):
    y_true_set = set(y_true)
    y_pred_set = set(y_pred[:k])
    result = round(len(y_true_set & y_pred_set) / float(len(y_true_set)), 2)
    return result

# Una aclaracion sobre recall at k es que suele penalizar mas al modelo, al tener valores "k", mas pequeños y tiende a favorecer
# mas al modelo al aumentar el valor de "k", lo cual puede llevar a una lectura erronea de la performance del modelo. 

# Otro problema que tiene es que no toma en cuenta el orden en que fueron devueltos los items en la recomendacion. La metrica asigna el mismo 
# resultado, sin importar el orden del Top N. 

# Para nuestro caso, nos interesa que el usuario encuentre 