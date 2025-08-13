import pandas as pd
import numpy as np
from collections import defaultdict

class HybridFilter:
    """
    An hybrid recommender system that combines collaborative filtering and content-based filtering.
    
    Parameters:
        cf_model (LFCollaborativeFilter): Trained collaborative filtering model.
        cb_model (ContentBasedFilter): Trained content-based filtering model.
        alpha (float): Weight for collaborative filter prediction (0-1). Default is 0.5.
            If alpha = 1: Only the collaborative filter's prediction is used.
            If alpha = 0: Only the content-based filter's prediction is used.
            If alpha = 0.5: Equal weight is given to both models.
    """
    def __init__(self, cf_model, cb_model, alpha=0.5):
        self.cf = cf_model
        self.cb = cb_model
        self.alpha = alpha
        self._is_fitted = False
        
        # Prediction cache to avoid redundant calculations
        self.prediction_cache = {}
        
        # Pre-calculate candidate items for recommendations
        self.all_items = None
        self.user_rated_items = defaultdict(set)
        
    def fit(self, train_df_cf=None, train_df_cb=None, movies_tokenized_df=None):
        """
        Train both collaborative and content-based filters if datasets are provided,
        or verify they are already trained if no datasets are provided.
        
        Parameters:
            train_df_cf: Training dataset for collaborative filter
            train_df_cb: Training dataset for content-based filter
            movies_tokenized_df: Tokenized movies data for content-based filter
            
        Returns:
            self: The trained hybrid filter
            
        Raises:
            ValueError: If only one dataset is provided or if models aren't trained
                        and no datasets are provided
        """
        if not self._is_fitted:
            # Case 1: Both datasets provided - retrain both models
            if train_df_cf is not None and train_df_cb is not None:
                print("Training collaborative filter...")
                self.cf.fit(train_df_cf)
                
                print("Training content-based filter...")
                if movies_tokenized_df is not None:
                    self.cb.fit(train_df_cb, movies_tokenized_df)
                else:
                    self.cb.fit(train_df_cb)
                
                # Use the more complete dataset for pre-calculation
                train_df_for_precalc = train_df_cb
                
            # Case 2: No datasets provided - check if models are trained
            elif train_df_cf is None and train_df_cb is None:
                cf_trained = hasattr(self.cf, 'is_fitted') and self.cf.is_fitted
                cb_trained = hasattr(self.cb, 'is_fitted') and self.cb.is_fitted
                
                if not cf_trained or not cb_trained:
                    raise ValueError(
                        "Both models must be trained if no training datasets are provided. "
                        f"CF model trained: {cf_trained}, CB model trained: {cb_trained}"
                    )
                
                # We need some dataset for pre-calculation
                if hasattr(self.cf, 'R_copy') and self.cf.R_copy is not None:
                    # If CF model is trained, we can extract training data
                    print("Extracting training data from collaborative filter...")
                    train_df_for_precalc = pd.melt(
                        self.cf.R_copy.reset_index(), 
                        id_vars=['userId'], 
                        var_name='movieId', 
                        value_name='rating'
                    ).dropna()
                elif hasattr(self.cb, 'user_movie_rating_df') and self.cb.user_movie_rating_df is not None:
                    # If CB model is trained, we can use its training data
                    print("Using training data from content-based filter...")
                    train_df_for_precalc = self.cb.user_movie_rating_df
                else:
                    raise ValueError(
                        "Cannot find training data in either model. "
                        "Please provide training datasets."
                    )
                
            # Case 3: Only one dataset provided - raise error
            else:
                raise ValueError(
                    "Either provide both training datasets (train_df_cf and train_df_cb) "
                    "or none (if models are already trained)"
                )
                
            # Pre-calculate data for faster recommendations
            print("Pre-calculating data for faster recommendations...")
            self._precalculate_user_items(train_df_for_precalc)
            
            self._is_fitted = True
        return self
    
    def _precalculate_user_items(self, train_df):
        """
        Pre-calculate user-rated items and all available items for faster lookups.
        OPTIMIZED: Uses vectorized operations instead of loops for 10-100x speed improvement.
        
        Parameters:
            train_df: DataFrame with columns 'userId', 'movieId', and 'rating'
        """
        # Get all unique items from both models
        items_cf = set()
        items_cb = set()
        
        # Add items from CF model
        if hasattr(self.cf, 'R_copy') and self.cf.R_copy is not None:
            items_cf = set(self.cf.R_copy.columns)
        
        # Add items from CB model
        if hasattr(self.cb, 'LDA_movies_data_df') and self.cb.LDA_movies_data_df is not None:
            items_cb = set(self.cb.LDA_movies_data_df['movieId'].unique())
        
        # Combine all items
        self.all_items = items_cf.union(items_cb)
        
        # Pre-calculate user-rated items from training data
        # We only consider items with non-null ratings as "rated"
        rated_df = train_df[train_df['rating'].notna()]
        
        print(f"Processing {len(rated_df)} ratings for {rated_df['userId'].nunique()} users...")
        
        # OPTIMIZATION: Use vectorized groupby instead of iterrows (10-100x faster)
        user_items = rated_df.groupby('userId')['movieId'].apply(set)
        self.user_rated_items = defaultdict(set, user_items.to_dict())
        
        print("User-item precalculation completed!")

    def compute_prediction(self, u, i):
        """
        Compute combined prediction using both models.
        Use cache to avoid redundant calculations.
        
        Parameters:
            u: User ID
            i: Item ID
            
        Returns:
            Combined prediction score
        """
        # Check cache first
        key = (u, i)
        if key in self.prediction_cache:
            return self.prediction_cache[key]
        
        cf_pred = None
        cb_pred = None
        
        # Try collaborative filter prediction
        try:
            cf_pred = self.cf.compute_prediction(u, i)
        except Exception:
            pass  # Collaborative filtering couldn't predict
            
        # Try content-based filter prediction
        try:
            cb_pred = self.cb.compute_prediction(u, i)
        except Exception:
            pass  # Content-based filtering couldn't predict

        # Combine predictions if both available
        if cf_pred is not None and cb_pred is not None:
            prediction = self.alpha * cf_pred + (1 - self.alpha) * cb_pred
        elif cf_pred is not None:
            prediction = cf_pred
        elif cb_pred is not None:
            prediction = cb_pred
        else:
            raise ValueError(f"No predictions available for user {u} and item {i}")
        
        # Cache the result
        self.prediction_cache[key] = prediction
        return prediction

    def recommend(self, u, dict_map=None, n_recommendations=10, sample_size=None):
        """
        Generate hybrid recommendations for a user efficiently.
        
        OPTIMIZATION: Added intelligent sampling for faster recommendations in production.
        
        Parameters:
            u: User ID
            dict_map: Movie ID to name mapping dictionary
            n_recommendations: Number of recommendations to return
            sample_size: Number of unrated items to sample for evaluation.
                        If None, evaluates ALL unrated items (exact but slower).
                        If int, samples that many items (faster but approximate).
                        
        Returns:
            list: A list of recommended movie IDs or names (if dict_map is provided)
        """
        # Get rated items for this user
        user_rated = self.user_rated_items.get(u, set())
        
        # If no pre-calculated rated items, try to get from models
        if not user_rated:
            try:
                # Try to get rated items from CF model
                user_ratings = self.cf.R_copy.loc[u]
                user_rated.update(user_ratings[~user_ratings.isna()].index)
            except (KeyError, AttributeError):
                # If CF doesn't have the user, try CB
                try:
                    user_rated.update(self.cb.user_rated_movie_ids_cb.get(u, set()))
                except (AttributeError, KeyError):
                    pass
        
        # Get all possible items from both models
        all_items = set()
        
        try:
            # Get items from CF model
            if hasattr(self.cf, 'R_copy') and self.cf.R_copy is not None:
                all_items.update(self.cf.R_copy.columns)
        except (AttributeError, KeyError):
            pass
            
        try:
            # Get items from CB model
            if hasattr(self.cb, 'LDA_movies_data_df') and self.cb.LDA_movies_data_df is not None:
                all_items.update(self.cb.LDA_movies_data_df['movieId'].unique())
        except (AttributeError, KeyError):
            pass
        
        # If we have pre-calculated all items, use those instead
        if self.all_items:
            all_items = self.all_items
            
        # Remove items the user has already rated
        unrated_items = list(all_items - user_rated)
        
        if not unrated_items:
            print("No unrated items available for this user")
            return []

        # OPTIMIZATION: Sample unrated items if sample_size is specified
        if sample_size is not None and len(unrated_items) > sample_size:
            np.random.seed(u)  # Reproducible per user
            unrated_items = np.random.choice(unrated_items, size=sample_size, replace=False)

        # Compute predictions for unrated items
        predictions = {}
        failed_items = 0
        total_items = len(unrated_items)
        
        for item in unrated_items:
            try:
                score = self.compute_prediction(u, item)
                predictions[item] = score
            except Exception as e:
                failed_items += 1
                continue
        
        # Check for high failure rate
        if failed_items > total_items * 0.3:
            print(f"Warning: High failure rate in predictions ({failed_items}/{total_items} items failed)")
        
        if not predictions:
            # Try fallback to CF or CB recommendations directly
            try:
                if hasattr(self.cf, 'recommend'):
                    cf_recs = self.cf.recommend(u, n_recommendations=n_recommendations)
                    if cf_recs:
                        return cf_recs
            except Exception:
                pass
                
            try:
                if hasattr(self.cb, 'recommend'):
                    cb_recs = self.cb.recommend(u, n_recommendations=n_recommendations)
                    if cb_recs:
                        return cb_recs
            except Exception:
                pass
                
            # If all else fails
            return []
        
        # Sort and select top recommendations
        sorted_recs = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
        
        # Convert the sorted recommendations to a list (using mapping if provided)
        if dict_map is not None:
            rec = [dict_map.get(movie_id, 'Unknown') for movie_id, score in sorted_recs]
        else:
            rec = [movie_id for movie_id, score in sorted_recs]
        
        return rec
        
    def clear_cache(self):
        """Clear the prediction cache to free up memory."""
        self.prediction_cache = {}
        
    @property
    def is_fitted(self):
        """Check if the hybrid filter has been fitted."""
        return self._is_fitted


# # OLD VERSION OF CACHE
# class HybridFilter:
#     """
#     An optimized hybrid recommender system that combines collaborative filtering and content-based filtering.
    
#     Parameters:
#         cf_model (LFCollaborativeFilter): Trained collaborative filtering model.
#         cb_model (ContentBasedFilter): Trained content-based filtering model.
#         alpha (float): Weight for collaborative filter prediction (0-1). Default is 0.5.
#             If alpha = 1: Only the collaborative filter's prediction is used.
#             If alpha = 0: Only the content-based filter's prediction is used.
#             If alpha = 0.5: Equal weight is given to both models.
#     """
#     def __init__(self, cf_model, cb_model, alpha=0.5):
#         self.cf = cf_model
#         self.cb = cb_model
#         self.alpha = alpha
#         self._is_fitted = False
        
#         # Prediction cache to avoid redundant calculations
#         self.prediction_cache = {}
        
#         # Pre-calculate candidate items for recommendations
#         self.all_items = None
#         self.user_rated_items = defaultdict(set)
        
#     def fit(self, train_df_cf=None, train_df_cb=None, movies_tokenized_df=None):
#         """
#         Train both collaborative and content-based filters if datasets are provided,
#         or verify they are already trained if no datasets are provided.
        
#         Parameters:
#             train_df_cf: Training dataset for collaborative filter
#             train_df_cb: Training dataset for content-based filter
#             movies_tokenized_df: Tokenized movies data for content-based filter
            
#         Returns:
#             self: The trained hybrid filter
            
#         Raises:
#             ValueError: If only one dataset is provided or if models aren't trained
#                         and no datasets are provided
#         """
#         if not self._is_fitted:
#             # Case 1: Both datasets provided - retrain both models
#             if train_df_cf is not None and train_df_cb is not None:
#                 print("Training collaborative filter...")
#                 self.cf.fit(train_df_cf)
                
#                 print("Training content-based filter...")
#                 if movies_tokenized_df is not None:
#                     self.cb.fit(train_df_cb, movies_tokenized_df)
#                 else:
#                     self.cb.fit(train_df_cb)
                
#                 # Use the more complete dataset for pre-calculation
#                 train_df_for_precalc = train_df_cb
                
#             # Case 2: No datasets provided - check if models are trained
#             elif train_df_cf is None and train_df_cb is None:
#                 cf_trained = hasattr(self.cf, 'is_fitted') and self.cf.is_fitted
#                 cb_trained = hasattr(self.cb, 'is_fitted') and self.cb.is_fitted
                
#                 if not cf_trained or not cb_trained:
#                     raise ValueError(
#                         "Both models must be trained if no training datasets are provided. "
#                         f"CF model trained: {cf_trained}, CB model trained: {cb_trained}"
#                     )
                
#                 # We need some dataset for pre-calculation
#                 if hasattr(self.cf, 'R_copy') and self.cf.R_copy is not None:
#                     # If CF model is trained, we can extract training data
#                     print("Extracting training data from collaborative filter...")
#                     train_df_for_precalc = pd.melt(
#                         self.cf.R_copy.reset_index(), 
#                         id_vars=['userId'], 
#                         var_name='movieId', 
#                         value_name='rating'
#                     ).dropna()
#                 elif hasattr(self.cb, 'user_movie_rating_df') and self.cb.user_movie_rating_df is not None:
#                     # If CB model is trained, we can use its training data
#                     print("Using training data from content-based filter...")
#                     train_df_for_precalc = self.cb.user_movie_rating_df
#                 else:
#                     raise ValueError(
#                         "Cannot find training data in either model. "
#                         "Please provide training datasets."
#                     )
                
#             # Case 3: Only one dataset provided - raise error
#             else:
#                 raise ValueError(
#                     "Either provide both training datasets (train_df_cf and train_df_cb) "
#                     "or none (if models are already trained)"
#                 )
                
#             # Pre-calculate data for faster recommendations
#             print("Pre-calculating data for faster recommendations...")
#             self._precalculate_user_items(train_df_for_precalc)
            
#             self._is_fitted = True
#         return self
    
#     def _precalculate_user_items(self, train_df):
#         """
#         Pre-calculate user-rated items and all available items for faster lookups.
        
#         Parameters:
#             train_df: DataFrame with columns 'userId', 'movieId', and 'rating'
#         """
#         # Get all unique items from both models
#         items_cf = set()
#         items_cb = set()
        
#         # Add items from CF model
#         if hasattr(self.cf, 'R_copy') and self.cf.R_copy is not None:
#             items_cf = set(self.cf.R_copy.columns)
        
#         # Add items from CB model
#         if hasattr(self.cb, 'LDA_movies_data_df') and self.cb.LDA_movies_data_df is not None:
#             items_cb = set(self.cb.LDA_movies_data_df['movieId'].unique())
        
#         # Combine all items
#         self.all_items = items_cf.union(items_cb)
        
#         # Pre-calculate user-rated items from training data
#         # We only consider items with non-null ratings as "rated"
#         rated_df = train_df[train_df['rating'].notna()]
        
#         print(f"Processing {len(rated_df)} ratings for {rated_df['userId'].nunique()} users...")
        
#         for _, row in tqdm(rated_df.iterrows(), desc="Building user-item cache", total=len(rated_df)):
#             self.user_rated_items[row['userId']].add(row['movieId'])

#     def compute_prediction(self, u, i):
#         """
#         Compute combined prediction using both models.
#         Use cache to avoid redundant calculations.
        
#         Parameters:
#             u: User ID
#             i: Item ID
            
#         Returns:
#             Combined prediction score
#         """
#         # Check cache first
#         key = (u, i)
#         if key in self.prediction_cache:
#             return self.prediction_cache[key]
        
#         cf_pred = None
#         cb_pred = None
        
#         # Try collaborative filter prediction
#         try:
#             cf_pred = self.cf.compute_prediction(u, i)
#         except Exception:
#             pass  # Collaborative filtering couldn't predict
            
#         # Try content-based filter prediction
#         try:
#             cb_pred = self.cb.compute_prediction(u, i)
#         except Exception:
#             pass  # Content-based filtering couldn't predict

#         # Combine predictions if both available
#         if cf_pred is not None and cb_pred is not None:
#             prediction = self.alpha * cf_pred + (1 - self.alpha) * cb_pred
#         elif cf_pred is not None:
#             prediction = cf_pred
#         elif cb_pred is not None:
#             prediction = cb_pred
#         else:
#             raise ValueError(f"No predictions available for user {u} and item {i}")
        
#         # Cache the result
#         self.prediction_cache[key] = prediction
#         return prediction

#     def recommend(self, u, dict_map=None, n_recommendations=10):
#         """
#         Generate hybrid recommendations for a user efficiently.
#         Process ALL unrated items without limitations for most accurate coverage metrics.
        
#         Parameters:
#             u: User ID
#             dict_map: Movie ID to name mapping dictionary
#             n_recommendations: Number of recommendations to return
            
#         Returns:
#             list: A list of recommended movie IDs or names (if dict_map is provided)
#         """
#         # Get rated items for this user
#         user_rated = self.user_rated_items.get(u, set())
        
#         # If no pre-calculated rated items, try to get from models
#         if not user_rated:
#             try:
#                 # Try to get rated items from CF model
#                 user_ratings = self.cf.R_copy.loc[u]
#                 user_rated.update(user_ratings[~user_ratings.isna()].index)
#             except (KeyError, AttributeError):
#                 # If CF doesn't have the user, try CB
#                 try:
#                     user_rated.update(self.cb.user_rated_movie_ids_cb.get(u, set()))
#                 except (AttributeError, KeyError):
#                     pass
        
#         # Get all possible items from both models
#         all_items = set()
        
#         try:
#             # Get items from CF model
#             if hasattr(self.cf, 'R_copy') and self.cf.R_copy is not None:
#                 all_items.update(self.cf.R_copy.columns)
#         except (AttributeError, KeyError):
#             pass
            
#         try:
#             # Get items from CB model
#             if hasattr(self.cb, 'LDA_movies_data_df') and self.cb.LDA_movies_data_df is not None:
#                 all_items.update(self.cb.LDA_movies_data_df['movieId'].unique())
#         except (AttributeError, KeyError):
#             pass
        
#         # If we have pre-calculated all items, use those instead
#         if self.all_items:
#             all_items = self.all_items
            
#         # Remove items the user has already rated
#         unrated_items = list(all_items - user_rated)
        
#         if not unrated_items:
#             print("No unrated items available for this user")
#             return []

#         # Compute predictions for unrated items
#         predictions = {}
#         failed_items = 0  # Contador de ítems que fallan
#         total_items = len(unrated_items)  # Total de ítems no calificados
        
#         for item in unrated_items:
#             try:
#                 score = self.compute_prediction(u, item)
#                 predictions[item] = score
#             except Exception as e:
#                 failed_items += 1
#                 continue
        
#         # Verificar si hay una tasa alta de fallos
#         if failed_items > total_items * 0.3:  # Más del 30% falló
#             print(f"Warning: High failure rate in predictions ({failed_items}/{total_items} items failed)")
        
#         if not predictions:
#             # Try fallback to CF or CB recommendations directly
#             try:
#                 if hasattr(self.cf, 'recommend'):
#                     cf_recs = self.cf.recommend(u, n_recommendations=n_recommendations)
#                     if cf_recs:
#                         return cf_recs
#             except Exception:
#                 pass
                
#             try:
#                 if hasattr(self.cb, 'recommend'):
#                     cb_recs = self.cb.recommend(u, n_recommendations=n_recommendations)
#                     if cb_recs:
#                         return cb_recs
#             except Exception:
#                 pass
                
#             # If all else fails
#             return []
        
#         # Sort and select top recommendations
#         sorted_recs = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
        
#         # Convert the sorted recommendations to a list (using mapping if provided)
#         if dict_map is not None:
#             rec = [dict_map.get(movie_id, 'Unknown') for movie_id, score in sorted_recs]
#         else:
#             rec = [movie_id for movie_id, score in sorted_recs]
        
#         return rec
        
#     def clear_cache(self):
#         """Clear the prediction cache to free up memory."""
#         self.prediction_cache = {}


# NEW CACHE VERSION, BUT IS NOT FINISHED YET
# class HybridFilter:
#     """
#     Un sistema de recomendación híbrido optimizado que combina filtrado colaborativo y 
#     filtrado basado en contenido con implementación de caché para mejorar el rendimiento.
    
#     Parameters:
#         cf_model (object): Modelo de filtrado colaborativo entrenado.
#         cb_model (object): Modelo de filtrado basado en contenido entrenado.
#         alpha (float): Peso para la predicción del filtro colaborativo (0-1). 
#                        Default es 0.5.
#             - alpha = 1: Solo se usa la predicción del filtro colaborativo.
#             - alpha = 0: Solo se usa la predicción del filtro basado en contenido.
#             - alpha = 0.5: Se da igual peso a ambos modelos.
#     """
#     def __init__(self, cf_model, cb_model, alpha=0.5):
#         self.cf = cf_model
#         self.cb = cb_model
#         self.alpha = alpha
#         self.is_fitted = False
        
#         # Caché para evitar cálculos redundantes
#         self.prediction_cache = {}
        
#         # Estructuras de datos precalculadas para recomendaciones más eficientes
#         self.user_rated_items = defaultdict(set)
#         self.all_items = set()
        
#     def fit(self, train_df_cf=None, train_df_cb=None, movies_tokenized_df=None):
#         """
#         Entrena el filtro híbrido o verifica que los componentes ya estén entrenados.
#         Precalcula información para acelerar las recomendaciones futuras.
        
#         Parameters:
#             train_df_cf (pd.DataFrame, optional): Dataset de entrenamiento para el filtro colaborativo
#             train_df_cb (pd.DataFrame, optional): Dataset de entrenamiento para el filtro basado en contenido
#             movies_tokenized_df (pd.DataFrame, optional): Datos tokenizados de películas para CB
            
#         Returns:
#             self: El filtro híbrido entrenado
            
#         Raises:
#             ValueError: Si sólo se proporciona un dataset o si los modelos no están entrenados
#                         y no se proporcionan datasets
#         """
#         if not self.is_fitted:
#             # Caso 1: Se proporcionan ambos datasets - reentrenar ambos modelos
#             if train_df_cf is not None and train_df_cb is not None:
#                 print("Entrenando filtro colaborativo...")
#                 self.cf.fit(train_df_cf)
                
#                 print("Entrenando filtro basado en contenido...")
#                 if movies_tokenized_df is not None:
#                     self.cb.fit(train_df_cb, movies_tokenized_df)
#                 else:
#                     self.cb.fit(train_df_cb)
                
#                 # Usar el dataset más completo para precálculo
#                 train_df_for_precalc = train_df_cb
                
#             # Caso 2: No se proporcionan datasets - verificar si los modelos están entrenados
#             elif train_df_cf is None and train_df_cb is None:
#                 cf_trained = hasattr(self.cf, 'is_fitted') and self.cf.is_fitted
#                 cb_trained = hasattr(self.cb, 'is_fitted') and self.cb.is_fitted
                
#                 if not cf_trained or not cb_trained:
#                     raise ValueError(
#                         "Ambos modelos deben estar entrenados si no se proporcionan datasets. "
#                         f"Modelo CF entrenado: {cf_trained}, Modelo CB entrenado: {cb_trained}"
#                     )
                
#                 # Necesitamos algún dataset para precálculo
#                 if hasattr(self.cf, 'R_copy') and self.cf.R_copy is not None:
#                     # Si el modelo CF está entrenado, podemos extraer los datos de entrenamiento
#                     print("Extrayendo datos de entrenamiento del filtro colaborativo...")
#                     train_df_for_precalc = pd.melt(
#                         self.cf.R_copy.reset_index(), 
#                         id_vars=['userId'], 
#                         var_name='movieId', 
#                         value_name='rating'
#                     ).dropna()
#                 elif hasattr(self.cb, 'user_movie_rating_df') and self.cb.user_movie_rating_df is not None:
#                     # Si el modelo CB está entrenado, podemos usar sus datos de entrenamiento
#                     print("Usando datos de entrenamiento del filtro basado en contenido...")
#                     train_df_for_precalc = self.cb.user_movie_rating_df
#                 else:
#                     raise ValueError(
#                         "No se encontraron datos de entrenamiento en ninguno de los modelos. "
#                         "Por favor, proporcione datasets de entrenamiento."
#                     )
                
#             # Caso 3: Solo se proporciona un dataset - error
#             else:
#                 raise ValueError(
#                     "Proporcione ambos datasets de entrenamiento (train_df_cf y train_df_cb) "
#                     "o ninguno (si los modelos ya están entrenados)"
#                 )
                
#             # Precalcular datos para recomendaciones más rápidas
#             print("Precalculando datos para recomendaciones más rápidas...")
#             self._precalculate_user_items(train_df_for_precalc)
            
#             self.is_fitted = True
#         return self
    
#     def _precalculate_user_items(self, train_df):
#         """
#         Precalcula ítems valorados por usuario y todos los ítems disponibles para búsquedas más rápidas.
#         Utiliza operaciones vectorizadas para mayor eficiencia.
        
#         Parameters:
#             train_df: DataFrame con columnas 'userId', 'movieId', y 'rating'
#         """
#         # Obtener todos los ítems únicos de ambos modelos
#         items_cf = set()
#         items_cb = set()
        
#         # Añadir ítems del modelo CF
#         if hasattr(self.cf, 'R_copy') and self.cf.R_copy is not None:
#             items_cf = set(self.cf.R_copy.columns)
        
#         # Añadir ítems del modelo CB
#         if hasattr(self.cb, 'LDA_movies_data_df') and self.cb.LDA_movies_data_df is not None:
#             items_cb = set(self.cb.LDA_movies_data_df['movieId'].unique())
        
#         # Combinar todos los ítems
#         self.all_items = items_cf.union(items_cb)
        
#         # Precalcular ítems valorados por usuario utilizando operaciones vectorizadas
#         rated_df = train_df[train_df['rating'].notna()]
        
#         print(f"Procesando {len(rated_df)} valoraciones para {rated_df['userId'].nunique()} usuarios...")
        
#         # Usar groupby en lugar de bucles para mayor eficiencia
#         user_items = rated_df.groupby('userId')['movieId'].apply(set)
#         self.user_rated_items = defaultdict(set, user_items.to_dict())

#     def compute_prediction(self, u, i):
#         """
#         Calcula la predicción combinada utilizando ambos modelos.
#         Utiliza caché para evitar cálculos repetidos.
        
#         Parameters:
#             u: ID de usuario
#             i: ID de ítem
            
#         Returns:
#             float: Valor de predicción combinada
#         """
#         # Verificar caché primero
#         key = (u, i)
#         if key in self.prediction_cache:
#             return self.prediction_cache[key]
        
#         cf_pred = None
#         cb_pred = None
        
#         # Intentar predicción con filtro colaborativo
#         try:
#             cf_pred = self.cf.compute_prediction(u, i)
#         except Exception:
#             pass  # El filtrado colaborativo no pudo predecir
            
#         # Intentar predicción con filtro basado en contenido
#         try:
#             cb_pred = self.cb.compute_prediction(u, i)
#         except Exception:
#             pass  # El filtrado basado en contenido no pudo predecir

#         # Combinar predicciones si ambas están disponibles
#         if cf_pred is not None and cb_pred is not None:
#             prediction = self.alpha * cf_pred + (1 - self.alpha) * cb_pred
#         elif cf_pred is not None:
#             prediction = cf_pred
#         elif cb_pred is not None:
#             prediction = cb_pred
#         else:
#             raise ValueError(f"No hay predicciones disponibles para usuario {u} e ítem {i}")
        
#         # Guardar en caché
#         self.prediction_cache[key] = prediction
#         return prediction

#     def recommend(self, u, dict_map=None, n_recommendations=10):
#         """
#         Genera recomendaciones híbridas para un usuario de manera eficiente.
#         Procesa TODOS los ítems no valorados para una cobertura precisa.
        
#         Parameters:
#             u: ID de usuario
#             dict_map: Diccionario de mapeo de ID de ítem a nombre
#             n_recommendations: Número de recomendaciones a devolver
            
#         Returns:
#             list: Lista de IDs de ítems recomendados o nombres (si se proporciona dict_map)
#         """
#         # Obtener ítems valorados por este usuario
#         user_rated = self.user_rated_items.get(u, set())
        
#         # Si no hay ítems valorados precalculados, intentar obtenerlos de los modelos
#         if not user_rated:
#             try:
#                 # Intentar obtener ítems valorados del modelo CF
#                 user_ratings = self.cf.R_copy.loc[u]
#                 user_rated.update(user_ratings[~user_ratings.isna()].index)
#             except (KeyError, AttributeError):
#                 # Si CF no tiene al usuario, intentar CB
#                 try:
#                     user_rated.update(self.cb.user_rated_movie_ids_cb.get(u, set()))
#                 except (AttributeError, KeyError):
#                     pass
        
#         # Obtener todos los ítems posibles de ambos modelos
#         if self.all_items:
#             all_items = self.all_items
#         else:
#             all_items = set()
#             try:
#                 # Obtener ítems del modelo CF
#                 if hasattr(self.cf, 'R_copy') and self.cf.R_copy is not None:
#                     all_items.update(self.cf.R_copy.columns)
#             except (AttributeError, KeyError):
#                 pass
                
#             try:
#                 # Obtener ítems del modelo CB
#                 if hasattr(self.cb, 'LDA_movies_data_df') and self.cb.LDA_movies_data_df is not None:
#                     all_items.update(self.cb.LDA_movies_data_df['movieId'].unique())
#             except (AttributeError, KeyError):
#                 pass
            
#         # Eliminar ítems que el usuario ya ha valorado
#         unrated_items = list(all_items - user_rated)
        
#         # Calcular predicciones para TODOS los ítems no valorados
#         predictions = {}
        
#         for item in unrated_items:
#             try:
#                 score = self.compute_prediction(u, item)
#                 predictions[item] = score
#             except Exception:
#                 continue
        
#         if not predictions:
#             # Intentar recurrir a recomendaciones de CF o CB directamente
#             try:
#                 if hasattr(self.cf, 'recommend'):
#                     cf_recs = self.cf.recommend(u, dict_map=dict_map, n_recommendations=n_recommendations)
#                     if cf_recs:
#                         return cf_recs
#             except Exception:
#                 pass
                
#             try:
#                 if hasattr(self.cb, 'recommend'):
#                     cb_recs = self.cb.recommend(u, dict_map=dict_map, n_recommendations=n_recommendations)
#                     if cb_recs:
#                         return cb_recs
#             except Exception:
#                 pass
                
#             # Si todo falla
#             return []
        
#         # Ordenar y seleccionar las mejores recomendaciones
#         sorted_recs = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
        
#         # Convertir las recomendaciones ordenadas a una lista (usando el mapeo si se proporciona)
#         if dict_map is not None:
#             rec = [dict_map.get(movie_id, 'Unknown') for movie_id, score in sorted_recs]
#         else:
#             rec = [movie_id for movie_id, score in sorted_recs]
        
#         return rec
        
#     def clear_cache(self):
#         """Limpia la caché de predicciones para liberar memoria."""
#         self.prediction_cache = {}