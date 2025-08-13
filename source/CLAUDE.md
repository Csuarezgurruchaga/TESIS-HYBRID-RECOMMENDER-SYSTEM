# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a hybrid recommender system research project implementing and evaluating multiple collaborative filtering and content-based filtering algorithms. The project focuses on comparing different approaches including:

- **Memory-based Collaborative Filtering**: Item-based CF with adjusted cosine similarity
- **Time-weighted Collaborative Filtering**: Temporal dynamics-aware CF with user clustering
- **Latent Factor Collaborative Filtering**: Matrix factorization using gradient descent optimization
- **Content-based Filtering**: LDA topic modeling on movie features (taglines, genres, tags)
- **Hybrid Approaches**: Linear combination of CF and CB methods with configurable alpha weighting

## Core Architecture

### Utils Module Structure

The `utils/` directory contains the core recommender system implementations:

- **`collaborative_filter.py`**: Three CF classes (`MemoryCollaborativeFilter`, `TWMemoryCollaborativeFilter`, `LFCollaborativeFilter`)
- **`content_based_filter.py`**: `ContentBasedFilter` class using LDA topic modeling
- **`hybrid_recommender.py`**: `HybridFilter` class combining CF and CB approaches with alpha weighting
- **`metrics_and_utils.py`**: Evaluation metrics (MAE, MSE, RMSE) and utility functions
- **`data_splitting_utils.py`**: Train/test splitting with temporal ordering and minimum overlap filtering

### Key Design Principles

- **No inheritance**: All classes are standalone implementations (inheritance is forbidden per project constraints)
- **Modular evaluation**: Each recommender implements `fit()`, `compute_prediction()`, and `recommend()` methods
- **Caching and optimization**: Prediction caching, precomputed similarities, and efficient matrix operations
- **Comprehensive metrics**: Both accuracy metrics (MAE, RMSE, NDCG) and coverage metrics (user/item coverage)

## Development Commands

### Model Evaluation
```bash
# Run comprehensive evaluation of all recommender algorithms
python evaluation_final_version.py

# Run alternative evaluation versions
python evaluate_recommenders_2025version.py
python evaluate_recommenders_2025(v2).py
python evaluate_recommenders_2025(v3).py
```

### Hyperparameter Optimization
```bash
# Run systematic hyperparameter search for LFCollaborativeFilter
python hiperparameter_search.py
```

### Exploratory Analysis
```bash
# Jupyter notebooks for data preprocessing and analysis
jupyter notebook prepro.ipynb
jupyter notebook item_based_CF(memory_based).ipynb
jupyter notebook content_nb.ipynb
```

## Data Requirements

The system expects MovieLens dataset structure:
- **`ratings.csv`**: userId, movieId, rating, timestamp columns
- **`movies.csv`**: movieId, title, genres columns  
- **`content_based_LDA_data_to_train.csv`**: Preprocessed tokenized movie features

Data should be located in `../data/movie_lens_small/` relative to the source directory.

## Evaluation Framework

### Metrics Computed
- **Accuracy**: MAE, MSE, RMSE, NDCG@10
- **Coverage**: User coverage (% users receiving recommendations), Item coverage (% items recommended)
- **Temporal splitting**: Test set uses most recent 20% of interactions per user

### Evaluation Process
1. Data splitting with `train_test_split()` (temporal ordering, min overlap filtering)
2. Model fitting on training data
3. Prediction computation for test interactions
4. Recommendation generation for coverage analysis
5. Comprehensive metrics calculation and comparison

## Hyperparameter Search

The `HyperparameterOptimizer` class in `hiperparameter_search.py` implements a three-phase optimization:

1. **Random search**: Broad exploration of parameter space
2. **Focused grid search**: Refined search around promising regions
3. **Fine-tuning**: Detailed optimization around best parameters

Results are saved to `hyperparameter_search_results/` with visualizations and model checkpoints.

## Model Implementation Details

### LFCollaborativeFilter
- **Matrix factorization**: User and item latent factor matrices
- **Optimization**: Stochastic gradient descent with L2 regularization
- **Key hyperparameters**: `F` (latent factors), `lr` (learning rate), `reg` (regularization), `steps` (iterations)

### TWMemoryCollaborativeFilter  
- **Temporal weighting**: Time-decay function based on user interaction patterns
- **User clustering**: K-means clustering for adaptive temporal parameters
- **Optimization**: Silhouette analysis for optimal cluster count

### ContentBasedFilter
- **LDA modeling**: 11-topic LDA on tokenized movie features
- **Similarity**: Cosine similarity between movie topic vectors
- **Features**: Combined taglines, genres, and user-generated tags

### HybridFilter
- **Linear combination**: `alpha * CF_prediction + (1-alpha) * CB_prediction`
- **Adaptive weighting**: Alpha can be tuned per evaluation
- **Prediction caching**: Avoids redundant computations

## Results and Outputs

### Generated Files
- **`recommender_systems_metrics.xlsx`**: Comprehensive evaluation results
- **`best_hyperparameters.json`**: Optimal parameters from hyperparameter search
- **`hyperparameter_search_results/`**: Detailed optimization logs and visualizations
- **Various plots**: Coverage analysis, precision metrics, hyperparameter effects

### Performance Tracking
- **Evaluation logs**: Saved to `recommender_evaluation.log`
- **Progress tracking**: tqdm progress bars for long-running operations
- **Error handling**: Graceful handling of prediction failures and missing data

## Development Notes

- **Python environment**: Requires pandas, numpy, scipy, scikit-learn, gensim, torch, tqdm
- **Computational requirements**: Memory-based CF is computationally expensive; consider data pruning for large datasets
- **Reproducibility**: Random seeds set in hyperparameter optimization for consistent results
- **Error handling**: Extensive try-catch blocks for robust evaluation across different algorithms