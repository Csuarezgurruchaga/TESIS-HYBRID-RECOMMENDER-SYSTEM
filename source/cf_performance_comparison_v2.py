#!/usr/bin/env python3
"""
Improved performance comparison script for collaborative filtering variants.
Measures training time, prediction time, recommendation generation time, and memory usage more accurately.
"""

import pandas as pd
import numpy as np
import time
import warnings
from tqdm import tqdm
import psutil
import os
import gc
import tracemalloc
from utils.data_splitting_utils import train_test_split
from utils.collaborative_filter import MemoryCollaborativeFilter, TWMemoryCollaborativeFilter, LFCollaborativeFilter

warnings.filterwarnings('ignore')

class ImprovedPerformanceProfiler:
    """Improved class to profile performance of collaborative filtering algorithms with better memory tracking"""
    
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
    
    def get_process_memory_mb(self):
        """Get current process memory in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def force_cleanup(self):
        """Force garbage collection and cleanup"""
        gc.collect()
        time.sleep(0.2)
    
    def measure_with_tracemalloc(self, func, *args, **kwargs):
        """
        Measure memory usage using tracemalloc for more accurate results
        """
        # Force cleanup and start memory tracing
        self.force_cleanup()
        tracemalloc.start()
        
        # Record start time and memory
        start_time = time.time()
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Record end time and get memory peak
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Convert to MB
        peak_memory_mb = peak / 1024 / 1024
        
        return result, end_time - start_time, peak_memory_mb
    
    def test_collaborative_filter_improved(self, model_class, model_params, train_df, test_dict, model_name, sample_users=30):
        """
        Improved testing function that measures performance more accurately
        """
        print(f"\n{'='*60}")
        print(f"TESTING {model_name}")
        print(f"{'='*60}")
        
        # Training phase with memory tracking
        def train_model():
            if model_class == LFCollaborativeFilter:
                model = model_class(**model_params)
                model.fit(train_df, F=19)  # Using optimal F value
            else:
                model = model_class(**model_params)
                model.fit(train_df)
            return model
        
        print(f"Training {model_name}...")
        model, train_time, train_memory = self.measure_with_tracemalloc(train_model)
        
        print(f"Training completed in {train_time:.2f} seconds")
        print(f"Peak memory during training: {train_memory:.2f} MB")
        
        # Prediction testing phase
        test_users = list(test_dict.keys())[:sample_users]
        print(f"Testing predictions for {len(test_users)} users...")
        
        def test_predictions():
            successful_predictions = 0
            for user_id in test_users:
                user_items = list(test_dict[user_id].keys())
                for item_id in user_items:
                    try:
                        prediction = model.compute_prediction(u=user_id, i=item_id)
                        if prediction is not None:
                            successful_predictions += 1
                    except:
                        pass
            return successful_predictions
        
        successful_predictions, prediction_time, prediction_memory = self.measure_with_tracemalloc(test_predictions)
        avg_prediction_time = prediction_time / len(test_users) if test_users else 0
        
        # Recommendation testing phase
        print("Testing recommendation generation...")
        rec_test_users = test_users[:10]  # Test fewer users for recommendations
        
        def test_recommendations():
            recommendations_generated = 0
            for user_id in rec_test_users:
                try:
                    recs = model.recommend(user_id, n_recommendations=10)
                    if recs:
                        recommendations_generated += 1
                except:
                    pass
            return recommendations_generated
        
        recommendations_generated, rec_time, rec_memory = self.measure_with_tracemalloc(test_recommendations)
        avg_rec_time = rec_time / len(rec_test_users) if rec_test_users else 0
        
        # Calculate total metrics
        total_time = train_time + prediction_time + rec_time
        peak_memory = max(train_memory, prediction_memory, rec_memory)
        
        # Force cleanup after testing
        del model
        self.force_cleanup()
        
        return {
            'model_name': model_name,
            'train_time': train_time,
            'train_memory_mb': train_memory,
            'prediction_time': prediction_time,
            'prediction_memory_mb': prediction_memory,
            'avg_prediction_time_per_user': avg_prediction_time,
            'recommendation_time': rec_time,
            'recommendation_memory_mb': rec_memory,
            'avg_recommendation_time_per_user': avg_rec_time,
            'total_time': total_time,
            'peak_memory_mb': peak_memory,
            'successful_predictions': successful_predictions,
            'recommendations_generated': recommendations_generated,
            'users_tested': len(test_users)
        }
    
    def generate_comparison_report(self, results):
        """Generate a comprehensive comparison report"""
        print("\n" + "="*80)
        print("IMPROVED COLLABORATIVE FILTERING PERFORMANCE COMPARISON REPORT")
        print("="*80)
        
        # Display detailed results
        print("\nDETAILED PERFORMANCE METRICS:")
        print("-" * 70)
        
        for result in results:
            print(f"\n{result['model_name']}:")
            print(f"  Training time: {result['train_time']:.2f} seconds")
            print(f"  Training memory: {result['train_memory_mb']:.2f} MB")
            print(f"  Prediction time: {result['prediction_time']:.2f} seconds")
            print(f"  Prediction memory: {result['prediction_memory_mb']:.2f} MB")
            print(f"  Avg prediction time per user: {result['avg_prediction_time_per_user']:.4f} seconds")
            print(f"  Recommendation time: {result['recommendation_time']:.2f} seconds")
            print(f"  Recommendation memory: {result['recommendation_memory_mb']:.2f} MB")
            print(f"  Avg recommendation time per user: {result['avg_recommendation_time_per_user']:.4f} seconds")
            print(f"  Total time: {result['total_time']:.2f} seconds")
            print(f"  Peak memory usage: {result['peak_memory_mb']:.2f} MB")
            print(f"  Successful predictions: {result['successful_predictions']}")
            print(f"  Recommendations generated: {result['recommendations_generated']}")
        
        # Summary comparison table
        print("\n\nSUMMARY COMPARISON TABLE:")
        print("-" * 80)
        
        comparison_df = pd.DataFrame({
            'Model': [r['model_name'] for r in results],
            'Train Time (s)': [f"{r['train_time']:.2f}" for r in results],
            'Total Time (s)': [f"{r['total_time']:.2f}" for r in results],
            'Peak Memory (MB)': [f"{r['peak_memory_mb']:.2f}" for r in results],
            'Train Memory (MB)': [f"{r['train_memory_mb']:.2f}" for r in results],
            'Pred/User (s)': [f"{r['avg_prediction_time_per_user']:.4f}" for r in results],
            'Rec/User (s)': [f"{r['avg_recommendation_time_per_user']:.4f}" for r in results]
        })
        
        print(comparison_df.to_string(index=False))
        
        # Performance rankings
        print("\n\nPERFORMANCE RANKINGS:")
        print("-" * 50)
        
        # Fastest training
        fastest_train = min(results, key=lambda x: x['train_time'])
        print(f"🏆 Fastest Training: {fastest_train['model_name']} ({fastest_train['train_time']:.2f}s)")
        
        # Fastest prediction
        fastest_pred = min(results, key=lambda x: x['avg_prediction_time_per_user'])
        print(f"🏆 Fastest Prediction: {fastest_pred['model_name']} ({fastest_pred['avg_prediction_time_per_user']:.4f}s per user)")
        
        # Fastest recommendation
        fastest_rec = min(results, key=lambda x: x['avg_recommendation_time_per_user'])
        print(f"🏆 Fastest Recommendation: {fastest_rec['model_name']} ({fastest_rec['avg_recommendation_time_per_user']:.4f}s per user)")
        
        # Lowest memory usage
        lowest_memory = min(results, key=lambda x: x['peak_memory_mb'])
        print(f"🏆 Lowest Memory Usage: {lowest_memory['model_name']} ({lowest_memory['peak_memory_mb']:.2f} MB)")
        
        # Fastest overall
        fastest_overall = min(results, key=lambda x: x['total_time'])
        print(f"🏆 Fastest Overall: {fastest_overall['model_name']} ({fastest_overall['total_time']:.2f}s total)")
        
        # Memory efficiency (speed per MB)
        print("\n\nMEMORY EFFICIENCY ANALYSIS:")
        print("-" * 50)
        for result in results:
            efficiency = result['total_time'] / result['peak_memory_mb'] if result['peak_memory_mb'] > 0 else float('inf')
            print(f"{result['model_name']}: {efficiency:.4f} seconds per MB")
        
        # Save results to CSV
        comparison_df.to_csv('cf_performance_comparison_improved.csv', index=False)
        
        # Save detailed results to JSON
        import json
        with open('cf_detailed_performance_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to:")
        print(f"  - cf_performance_comparison_improved.csv")
        print(f"  - cf_detailed_performance_results.json")
        
        return comparison_df

def main():
    """Main function to run improved performance comparison"""
    print("IMPROVED COLLABORATIVE FILTERING PERFORMANCE COMPARISON")
    print("=" * 70)
    
    # Initialize profiler
    profiler = ImprovedPerformanceProfiler()
    
    # Load data
    train_df, test_dict, ratings = profiler.load_data()
    
    # Define models to test with their parameters
    models_to_test = [
        (MemoryCollaborativeFilter, {'min_overlap': 5, 'n_neighbours': 40}, 'MemoryCollaborativeFilter'),
        (TWMemoryCollaborativeFilter, {'min_overlap': 5, 'n_neighbours': 40}, 'TWMemoryCollaborativeFilter'),
        (LFCollaborativeFilter, {'reg': 0.83, 'steps': 8000, 'lr': 3e-4}, 'LFCollaborativeFilter')
    ]
    
    # Run performance tests
    results = []
    
    for model_class, model_params, model_name in models_to_test:
        try:
            print(f"\n{'='*20} Starting {model_name} {'='*20}")
            result = profiler.test_collaborative_filter_improved(
                model_class, model_params, train_df, test_dict, model_name, sample_users=30
            )
            results.append(result)
            print(f"✅ {model_name} completed successfully")
        except Exception as e:
            print(f"❌ Error testing {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Generate comparison report
    if results:
        profiler.generate_comparison_report(results)
    else:
        print("❌ No results to compare. All tests failed.")

if __name__ == "__main__":
    main()