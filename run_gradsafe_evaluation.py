#!/usr/bin/env python3
"""
Main script to run GradSafe evaluation on LLaVA 1.6 with vision-language benchmark

This script:
1. Sets up the environment and activates the llava conda environment
2. Loads the benchmark datasets
3. Runs GradSafe evaluation
4. Reports results

Usage:
    # Quick test (200 samples from XSTest):
    python run_gradsafe_evaluation.py --quick_test

    # Full benchmark (1800 samples):
    python run_gradsafe_evaluation.py [--model_path MODEL_PATH] [--use_training_data] [--output_file OUTPUT_FILE]
"""

import os
import sys
import argparse
import subprocess
import time
from datetime import datetime

def check_conda_env():
    """Check if we're in the llava conda environment"""
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    if conda_env != 'llava':
        print("Warning: Not in 'llava' conda environment.")
        print("Please activate the llava environment first:")
        print("conda activate llava")
        return False
    return True

def check_dependencies():
    """Check if required dependencies are available"""
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
    except ImportError:
        print("Error: PyTorch not found. Please install PyTorch.")
        return False
    
    try:
        import llava
        print("LLaVA library found.")
    except ImportError:
        print("Error: LLaVA library not found. Please install LLaVA.")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Run GradSafe evaluation on LLaVA 1.6')
    parser.add_argument('--model_path', type=str, default='model/llava-v1.6-vicuna-7b',
                        help='Path to LLaVA model directory')
    parser.add_argument('--use_training_data', action='store_true',
                        help='Use training data to find critical parameters (default: use built-in examples)')
    parser.add_argument('--output_file', type=str, default='gradsafe_llava_results.json',
                        help='Output file for results')
    parser.add_argument('--skip_env_check', action='store_true',
                        help='Skip conda environment check')
    parser.add_argument('--quick_test', action='store_true',
                        help='Run quick test on 100+100 XSTest samples instead of full 1800 samples')
    # parser.add_argument('--optimize_threshold', action='store_true',
    #                     help='Find and use optimal threshold instead of default 0.25')
    parser.add_argument('--disable_cache', action='store_true',
                        help='Disable caching of gradients and scores')
    parser.add_argument('--cooling_interval', type=int, default=10,
                        help='Number of samples to process before cooling break (default: 10)')
    parser.add_argument('--cooling_time', type=int, default=30,
                        help='Cooling break duration in seconds (default: 30)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("GRADSAFE EVALUATION ON LLAVA 1.6")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model path: {args.model_path}")
    print(f"Use training data: {args.use_training_data}")
    print(f"Output file: {args.output_file}")
    print(f"Caching: {'disabled' if args.disable_cache else 'enabled'}")
    print(f"Cooling: {args.cooling_time}s break every {args.cooling_interval} samples")
    print("="*80)
    
    # Check environment
    if not args.skip_env_check:
        if not check_conda_env():
            print("Please activate the llava conda environment and try again.")
            return 1
    
    if not check_dependencies():
        print("Dependency check failed. Please install required packages.")
        return 1
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} does not exist.")
        return 1
    
    # Import evaluation modules (after environment checks)
    try:
        from gradsafe_benchmark_evaluation import BenchmarkEvaluator
    except ImportError as e:
        print(f"Error importing evaluation modules: {e}")
        print("Make sure all required files are in the current directory.")
        return 1
    
    # Initialize evaluator
    print("Initializing benchmark evaluator...")
    evaluator = BenchmarkEvaluator(args.model_path)
    
    try:
        # Load datasets
        print("\nStep 1: Loading datasets...")
        start_time = time.time()
        
        if args.use_training_data:
            training_data = evaluator.load_training_datasets()
        else:
            training_data = None
            print("Using default samples for critical parameter identification")
        
        test_data = evaluator.load_test_datasets(quick_test=args.quick_test)
        
        load_time = time.time() - start_time
        print(f"Dataset loading completed in {load_time:.2f} seconds")
        
        # Initialize GradSafe
        print("\nStep 2: Initializing GradSafe...")
        start_time = time.time()
        evaluator.initialize_gradsafe()
        init_time = time.time() - start_time
        print(f"Model initialization completed in {init_time:.2f} seconds")
        
        # Find critical parameters
        print("\nStep 3: Finding safety-critical parameters...")
        start_time = time.time()
        gradient_norms_compare, minus_row_cos, minus_col_cos = evaluator.find_critical_parameters(training_data)
        param_time = time.time() - start_time
        print(f"Critical parameter identification completed in {param_time:.2f} seconds")

        # Evaluate on test set
        print("\nStep 4: Evaluating on test set...")
        start_time = time.time()
        metrics, safety_scores, predictions, labels = evaluator.evaluate_on_test_set(
            test_data, gradient_norms_compare, minus_row_cos, minus_col_cos,
            use_cache=not args.disable_cache,
            cooling_interval=args.cooling_interval,
            cooling_time=args.cooling_time
        )
        eval_time = time.time() - start_time
        print(f"Test set evaluation completed in {eval_time:.2f} seconds")
        
        # Print and save results
        print("\nStep 5: Results...")
        evaluator.print_results(metrics)
        evaluator.save_results(metrics, args.output_file)
        
        # Print timing summary
        total_time = load_time + init_time + param_time + eval_time
        print(f"\nTiming Summary:")
        print(f"Dataset loading:     {load_time:.2f}s")
        print(f"Model initialization: {init_time:.2f}s")
        print(f"Parameter finding:   {param_time:.2f}s")
        print(f"Test evaluation:     {eval_time:.2f}s")
        print(f"Total time:          {total_time:.2f}s")
        
        print(f"\nEvaluation completed successfully!")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Clean up
        print("\nCleaning up...")
        evaluator.cleanup()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
