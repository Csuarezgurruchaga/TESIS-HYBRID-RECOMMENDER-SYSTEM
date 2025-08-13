#!/usr/bin/env python3
"""
Performance comparison script for collaborative filtering variants.
Measures training time, prediction time, and recommendation generation time.
"""

import pandas as pd
import numpy as np
import time
import warnings
from tqdm import tqdm
import psutil
import os
import gc
import sys
from utils.data_splitting_utils import train_test_split
from utils.collaborative_filter import MemoryCollaborativeFilter, TWMemoryCollaborativeFilter, LFCollaborativeFilter

warnings.filterwarnings('ignore')

class PerformanceProfiler:
    """Class to profile performance of collaborative filtering algorithms"""
    
    def __init__(self, data_path='../data/movie_lens_small/'):
        self.data_path = data_path
        self.results = {}
        
    def load_data(self):
        """Load and prepare data for testing"""
        print("Loading MovieLens data...")
        ratings_path = os.path.join(self.data_path, 'ratings.csv')
        ratings = pd.read_csv(ratings_path)
        
        print(f"Dataset size: {len(ratings)} interactions")
        print(f"Users: {ratings['userId'].nunique()}, Movies: {ratings['movieId'].nunique()}")
        
        # Split data for collaborative filtering (with minimum overlap)
        print("Splitting data for collaborative filtering...")
        train_df, test_dict = train_test_split(ratings, min_overlap=True)
        
        print(f"Training size: {len(train_df)} interactions")
        print(f"Test users: {len(test_dict)} users")
        
        return train_df, test_dict, ratings
    
    def get_memory_usage_mb(self):
        """Get current memory usage in MB with more precision"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024
    
    def force_cleanup(self):
        """Force garbage collection and cleanup"""
        gc.collect()
        time.sleep(0.5)  # Give time for cleanup
    
    def measure_model_memory(self, model_func, *args, **kwargs):
        """
        Measure memory usage of a model more accurately by running it in isolation
        """
        # Force cleanup before measurement
        self.force_cleanup()
        
        # Measure baseline memory multiple times and take average
        baseline_measurements = []
        for _ in range(3):
            baseline_measurements.append(self.get_memory_usage_mb())
            time.sleep(0.1)
        baseline_memory = np.mean(baseline_measurements)
        
        # Run the model function
        start_time = time.time()
        result = model_func(*args, **kwargs)
        end_time = time.time()
        
        # Measure peak memory multiple times and take average
        peak_measurements = []
        for _ in range(3):
            peak_measurements.append(self.get_memory_usage_mb())
            time.sleep(0.1)
        peak_memory = np.mean(peak_measurements)
        
        # Calculate memory used (ensure it's not negative)
        memory_used = max(0, peak_memory - baseline_memory)
        
        return result, end_time - start_time, memory_used
    
    def test_memory_collaborative_filter(self, train_df, test_dict, sample_users=50):
        """Test MemoryCollaborativeFilter performance"""
        print("\n" + "="*60)
        print("TESTING MEMORY-BASED COLLABORATIVE FILTER")
        print("="*60)
        
        def train_model():
            cf_model = MemoryCollaborativeFilter(min_overlap=5, n_neighbours=40)
            cf_model.fit(train_df)
            return cf_model
        
        # Measure training time and memory
        print("Training MemoryCollaborativeFilter...")
        cf_model, train_time, memory_used = self.measure_model_memory(train_model)
        
        print(f"Training completed in {train_time:.2f} seconds")
        print(f"Memory used for training: {memory_used:.2f} MB")
        
        # Test prediction time on a sample of test users
        test_users = list(test_dict.keys())[:sample_users]
        print(f"Testing predictions for {len(test_users)} users...")
        
        prediction_times = []
        successful_predictions = 0
        
        for user_id in tqdm(test_users, desc="Computing predictions"):
            user_items = list(test_dict[user_id].keys())
            user_pred_start = time.time()
            
            for item_id in user_items:
                try:
                    prediction = cf_model.compute_prediction(u=user_id, i=item_id)
                    if prediction is not None:
                        successful_predictions += 1
                except:
                    pass
            
            user_pred_time = time.time() - user_pred_start
            prediction_times.append(user_pred_time)
        
        avg_prediction_time = np.mean(prediction_times)
        total_prediction_time = sum(prediction_times)
        
        # Test recommendation generation time
        print("Testing recommendation generation...")
        rec_start = time.time()
        recommendations_generated = 0
        
        for user_id in tqdm(test_users[:10], desc="Generating recommendations"):  # Test fewer users for recommendations
            try:
                recs = cf_model.recommend(user_id, n_recommendations=10)
                if recs:
                    recommendations_generated += 1
            except:
                pass
        
        rec_time = time.time() - rec_start
        avg_rec_time = rec_time / min(10, len(test_users))
        
        return {
            'model_name': 'MemoryCollaborativeFilter',
            'train_time': train_time,
            'total_prediction_time': total_prediction_time,
            'avg_prediction_time_per_user': avg_prediction_time,
            'recommendation_time': rec_time,
            'avg_recommendation_time_per_user': avg_rec_time,
            'total_time': train_time + total_prediction_time + rec_time,
            'memory_used_mb': memory_used,
            'successful_predictions': successful_predictions,
            'recommendations_generated': recommendations_generated,
            'users_tested': len(test_users)
        }
    
    def test_tw_memory_collaborative_filter(self, train_df, test_dict, sample_users=50):
        """Test TWMemoryCollaborativeFilter performance"""
        print("\n" + "="*60)
        print("TESTING TIME-WEIGHTED MEMORY-BASED COLLABORATIVE FILTER")
        print("="*60)
        
        def train_model():
            tw_cf_model = TWMemoryCollaborativeFilter(min_overlap=5, n_neighbours=40)
            tw_cf_model.fit(train_df)
            return tw_cf_model
        
        # Measure training time and memory
        print("Training TWMemoryCollaborativeFilter...")
        tw_cf_model, train_time, memory_used = self.measure_model_memory(train_model)
        
        print(f"Training completed in {train_time:.2f} seconds")
        print(f"Memory used for training: {memory_used:.2f} MB")
        
        # Test prediction time on a sample of test users
        test_users = list(test_dict.keys())[:sample_users]
        print(f"Testing predictions for {len(test_users)} users...")
        
        prediction_times = []
        successful_predictions = 0
        
        for user_id in tqdm(test_users, desc="Computing predictions"):
            user_items = list(test_dict[user_id].keys())
            user_pred_start = time.time()
            
            for item_id in user_items:
                try:
                    prediction = tw_cf_model.compute_prediction(u=user_id, i=item_id)
                    if prediction is not None:
                        successful_predictions += 1
                except:
                    pass
            
            user_pred_time = time.time() - user_pred_start
            prediction_times.append(user_pred_time)
        
        avg_prediction_time = np.mean(prediction_times)
        total_prediction_time = sum(prediction_times)
        
        # Test recommendation generation time
        print("Testing recommendation generation...")
        rec_start = time.time()
        recommendations_generated = 0
        
        for user_id in tqdm(test_users[:10], desc="Generating recommendations"):  # Test fewer users for recommendations
            try:
                recs = tw_cf_model.recommend(user_id, n_recommendations=10)
                if recs:
                    recommendations_generated += 1
            except:
                pass
        
        rec_time = time.time() - rec_start
        avg_rec_time = rec_time / min(10, len(test_users))
        
        return {
            'model_name': 'TWMemoryCollaborativeFilter',
            'train_time': train_time,
            'total_prediction_time': total_prediction_time,
            'avg_prediction_time_per_user': avg_prediction_time,
            'recommendation_time': rec_time,
            'avg_recommendation_time_per_user': avg_rec_time,
            'total_time': train_time + total_prediction_time + rec_time,
            'memory_used_mb': memory_used,
            'successful_predictions': successful_predictions,
            'recommendations_generated': recommendations_generated,
            'users_tested': len(test_users)
        }
    
    def test_lf_collaborative_filter(self, train_df, test_dict, sample_users=50):
        """Test LFCollaborativeFilter performance"""
        print("\n" + "="*60)
        print("TESTING LATENT FACTOR COLLABORATIVE FILTER")
        print("="*60)
        
        def train_model():
            lf_cf_model = LFCollaborativeFilter(reg=0.83, steps=8000, lr=3e-4)
            lf_cf_model.fit(train_df, F=19)  # Using optimal F value
            return lf_cf_model
        
        # Measure training time and memory
        print("Training LFCollaborativeFilter...")
        lf_cf_model, train_time, memory_used = self.measure_model_memory(train_model)
        
        print(f"Training completed in {train_time:.2f} seconds")
        print(f"Memory used for training: {memory_used:.2f} MB")
        
        # Test prediction time on a sample of test users
        test_users = list(test_dict.keys())[:sample_users]
        print(f"Testing predictions for {len(test_users)} users...")
        
        prediction_times = []
        successful_predictions = 0
        
        for user_id in tqdm(test_users, desc="Computing predictions"):
            user_items = list(test_dict[user_id].keys())
            user_pred_start = time.time()
            
            for item_id in user_items:
                try:
                    prediction = lf_cf_model.compute_prediction(u=user_id, i=item_id)
                    if prediction is not None:
                        successful_predictions += 1
                except:
                    pass
            
            user_pred_time = time.time() - user_pred_start
            prediction_times.append(user_pred_time)
        
        avg_prediction_time = np.mean(prediction_times)
        total_prediction_time = sum(prediction_times)
        
        # Test recommendation generation time
        print("Testing recommendation generation...")
        rec_start = time.time()
        recommendations_generated = 0
        
        for user_id in tqdm(test_users[:10], desc="Generating recommendations"):  # Test fewer users for recommendations
            try:
                recs = lf_cf_model.recommend(user_id, n_recommendations=10)
                if recs:
                    recommendations_generated += 1
            except:
                pass
        
        rec_time = time.time() - rec_start
        avg_rec_time = rec_time / min(10, len(test_users))
        
        return {
            'model_name': 'LFCollaborativeFilter',
            'train_time': train_time,
            'total_prediction_time': total_prediction_time,
            'avg_prediction_time_per_user': avg_prediction_time,
            'recommendation_time': rec_time,
            'avg_recommendation_time_per_user': avg_rec_time,
            'total_time': train_time + total_prediction_time + rec_time,
            'memory_used_mb': memory_used,
            'successful_predictions': successful_predictions,
            'recommendations_generated': recommendations_generated,
            'users_tested': len(test_users)
        }
    
    def generate_comparison_report(self, results):
        """Generate a comprehensive comparison report"""
        print("\n" + "="*80)
        print("COLLABORATIVE FILTERING PERFORMANCE COMPARISON REPORT")
        print("="*80)
        
        # Create DataFrame for easy comparison
        df_results = pd.DataFrame(results)
        df_results = df_results.set_index('model_name')
        
        # Display detailed results
        print("\nDETAILED PERFORMANCE METRICS:")
        print("-" * 50)
        
        for model_name, result in zip(df_results.index, results):
            print(f"\n{model_name}:")
            print(f"  Training time: {result['train_time']:.2f} seconds")
            print(f"  Total prediction time: {result['total_prediction_time']:.2f} seconds")
            print(f"  Avg prediction time per user: {result['avg_prediction_time_per_user']:.4f} seconds")
            print(f"  Recommendation generation time: {result['recommendation_time']:.2f} seconds")
            print(f"  Avg recommendation time per user: {result['avg_recommendation_time_per_user']:.4f} seconds")
            print(f"  Total time: {result['total_time']:.2f} seconds")
            print(f"  Memory used: {result['memory_used_mb']:.2f} MB")
            print(f"  Successful predictions: {result['successful_predictions']}")
            print(f"  Recommendations generated: {result['recommendations_generated']}")
        
        # Summary comparison table
        print("\n\nSUMMARY COMPARISON TABLE:")
        print("-" * 50)
        
        comparison_df = pd.DataFrame({
            'Model': [r['model_name'] for r in results],
            'Training Time (s)': [f"{r['train_time']:.2f}" for r in results],
            'Total Time (s)': [f"{r['total_time']:.2f}" for r in results],
            'Memory (MB)': [f"{r['memory_used_mb']:.2f}" for r in results],
            'Pred/User (s)': [f"{r['avg_prediction_time_per_user']:.4f}" for r in results],
            'Rec/User (s)': [f"{r['avg_recommendation_time_per_user']:.4f}" for r in results]
        })
        
        print(comparison_df.to_string(index=False))
        
        # Performance rankings
        print("\n\nPERFORMANCE RANKINGS:")
        print("-" * 50)
        
        # Fastest training
        fastest_train = min(results, key=lambda x: x['train_time'])
        print(f"Fastest Training: {fastest_train['model_name']} ({fastest_train['train_time']:.2f}s)")
        
        # Fastest prediction
        fastest_pred = min(results, key=lambda x: x['avg_prediction_time_per_user'])
        print(f"Fastest Prediction: {fastest_pred['model_name']} ({fastest_pred['avg_prediction_time_per_user']:.4f}s per user)")
        
        # Fastest recommendation
        fastest_rec = min(results, key=lambda x: x['avg_recommendation_time_per_user'])
        print(f"Fastest Recommendation: {fastest_rec['model_name']} ({fastest_rec['avg_recommendation_time_per_user']:.4f}s per user)")
        
        # Lowest memory usage
        lowest_memory = min(results, key=lambda x: x['memory_used_mb'])
        print(f"Lowest Memory Usage: {lowest_memory['model_name']} ({lowest_memory['memory_used_mb']:.2f} MB)")
        
        # Fastest overall
        fastest_overall = min(results, key=lambda x: x['total_time'])
        print(f"Fastest Overall: {fastest_overall['model_name']} ({fastest_overall['total_time']:.2f}s total)")
        
        # Save results to CSV
        comparison_df.to_csv('cf_performance_comparison.csv', index=False)
        print(f"\nResults saved to: cf_performance_comparison.csv")
        
        return comparison_df

def main():
    """Main function to run performance comparison"""
    print("COLLABORATIVE FILTERING PERFORMANCE COMPARISON")
    print("=" * 60)
    
    # Initialize profiler
    profiler = PerformanceProfiler()
    
    # Load data
    train_df, test_dict, ratings = profiler.load_data()
    
    # Run performance tests
    results = []
    
    try:
        # Test Memory-based CF
        memory_cf_results = profiler.test_memory_collaborative_filter(train_df, test_dict)
        results.append(memory_cf_results)
    except Exception as e:
        print(f"Error testing MemoryCollaborativeFilter: {str(e)}")
    
    try:
        # Test Time-weighted Memory-based CF
        tw_memory_cf_results = profiler.test_tw_memory_collaborative_filter(train_df, test_dict)
        results.append(tw_memory_cf_results)
    except Exception as e:
        print(f"Error testing TWMemoryCollaborativeFilter: {str(e)}")
    
    try:
        # Test Latent Factor CF
        lf_cf_results = profiler.test_lf_collaborative_filter(train_df, test_dict)
        results.append(lf_cf_results)
    except Exception as e:
        print(f"Error testing LFCollaborativeFilter: {str(e)}")
    
    # Generate comparison report
    if results:
        profiler.generate_comparison_report(results)
    else:
        print("No results to compare. All tests failed.")

if __name__ == "__main__":
    main()