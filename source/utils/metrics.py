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
