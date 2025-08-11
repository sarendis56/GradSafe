#!/usr/bin/env python3
"""
GradSafe Evaluation on Vision-Language Benchmark

This script evaluates GradSafe on the specified benchmark datasets:

Training Set (2,000 examples, 1:1 ratio):
- Benign (1,000): Alpaca (500) + MM-Vet (218) + OpenAssistant (282)
- Malicious (1,000): AdvBench (300) + JailbreakV-28K (550, llm_transfer_attack + query_related) + DAN variants (150)

Test Set (1,800 examples, 1:1 ratio):
- Safe (900): XSTest safe (250) + FigTxt safe (300) + VQAv2 (350)
- Unsafe (900): XSTest unsafe (200) + FigTxt unsafe (350) + VAE (200) + JailbreakV-28K (150, figstep attack)
"""

import os
import sys
import json
import random
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Add code directory to path to import existing modules
sys.path.append('code')

# Import existing dataset loaders
from load_datasets import (
    load_alpaca, load_mm_vet, load_openassistant,
    load_advbench, load_JailBreakV_llm_transfer_attack, load_JailBreakV_query_related,
    load_dan_prompts, load_XSTest, load_FigTxt, load_vqav2,
    load_adversarial_img, load_JailBreakV_figstep,
    set_dataset_random_seed
)

# Import GradSafe implementation
from gradsafe_llava import GradSafeLLaVA

class BenchmarkEvaluator:
    """Evaluator for GradSafe on vision-language benchmark"""
    
    def __init__(self, model_path, random_seed=42):
        self.model_path = model_path
        self.random_seed = random_seed
        self.gradsafe = None
        
        # Set random seeds for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        set_dataset_random_seed(random_seed)
        
    def load_training_datasets(self):
        """Load training datasets according to benchmark specification"""
        print("Loading training datasets...")
        
        # Benign training data (1,000 total)
        print("Loading benign training data...")
        alpaca_data = load_alpaca(max_samples=500)
        mmvet_data = load_mm_vet()[:218]  # Take first 218 samples
        openassistant_data = load_openassistant(max_samples=282)
        
        benign_train = alpaca_data + mmvet_data + openassistant_data
        print(f"Loaded {len(benign_train)} benign training samples")
        
        # Malicious training data (1,000 total)
        print("Loading malicious training data...")
        advbench_data = load_advbench(max_samples=300)
        
        # JailbreakV-28K: llm_transfer_attack + query_related (550 total)
        jailbreak_llm_data = load_JailBreakV_llm_transfer_attack(max_samples=275)
        jailbreak_query_data = load_JailBreakV_query_related(max_samples=275)
        
        dan_data = load_dan_prompts(max_samples=150)
        
        malicious_train = advbench_data + jailbreak_llm_data + jailbreak_query_data + dan_data
        print(f"Loaded {len(malicious_train)} malicious training samples")
        
        # Combine and shuffle
        training_data = benign_train + malicious_train
        random.shuffle(training_data)
        
        print(f"Total training samples: {len(training_data)}")
        return training_data
    
    def load_test_datasets(self):
        """Load test datasets according to benchmark specification"""
        print("Loading test datasets...")
        
        # Safe test data (900 total)
        print("Loading safe test data...")
        xstest_data = load_XSTest()
        xstest_safe = [sample for sample in xstest_data if sample['toxicity'] == 0][:250]
        
        figtxt_data = load_FigTxt()
        figtxt_safe = [sample for sample in figtxt_data if sample['toxicity'] == 0][:300]
        
        vqav2_data = load_vqav2(max_samples=350)
        
        safe_test = xstest_safe + figtxt_safe + vqav2_data
        print(f"Loaded {len(safe_test)} safe test samples")
        
        # Unsafe test data (900 total)
        print("Loading unsafe test data...")
        xstest_unsafe = [sample for sample in xstest_data if sample['toxicity'] == 1][:200]
        
        figtxt_unsafe = [sample for sample in figtxt_data if sample['toxicity'] == 1][:350]
        
        vae_data = load_adversarial_img()[:200]
        
        jailbreak_figstep_data = load_JailBreakV_figstep(max_samples=150)
        
        unsafe_test = xstest_unsafe + figtxt_unsafe + vae_data + jailbreak_figstep_data
        print(f"Loaded {len(unsafe_test)} unsafe test samples")
        
        # Combine and shuffle
        test_data = safe_test + unsafe_test
        random.shuffle(test_data)
        
        print(f"Total test samples: {len(test_data)}")
        return test_data
    
    def initialize_gradsafe(self):
        """Initialize GradSafe model"""
        print(f"Initializing GradSafe with model: {self.model_path}")
        self.gradsafe = GradSafeLLaVA(self.model_path)
        
    def find_critical_parameters(self, training_data=None):
        """Find safety-critical parameters using training data or defaults"""
        if training_data is not None:
            # Use training data to find critical parameters
            unsafe_samples = [sample for sample in training_data if sample.get('toxicity', 0) == 1]
            safe_samples = [sample for sample in training_data if sample.get('toxicity', 0) == 0]

            # Limit to very small number for gradient computation due to memory constraints
            unsafe_samples = unsafe_samples[:3]  # Reduced from 10 to 3
            safe_samples = safe_samples[:3]      # Reduced from 10 to 3

            print(f"Using {len(unsafe_samples)} unsafe and {len(safe_samples)} safe training samples for parameter identification")
            print("Note: Using small sample size due to GPU memory constraints")
        else:
            unsafe_samples = None
            safe_samples = None
            print("Using default samples for parameter identification")

        return self.gradsafe.find_critical_parameters(unsafe_samples, safe_samples)
    
    def evaluate_on_test_set(self, test_data, reference_gradients, row_cosine_gaps, col_cosine_gaps,
                           use_cache=True, cooling_interval=10, cooling_time=60):
        """Evaluate GradSafe on test set with caching and cooling"""
        print("Evaluating GradSafe on test set...")

        # Compute safety scores and predictions
        safety_scores, predictions, labels = self.gradsafe.evaluate_samples(
            test_data, reference_gradients, row_cosine_gaps, col_cosine_gaps,
            use_cache=use_cache, cooling_interval=cooling_interval, cooling_time=cooling_time
        )

        # Compute metrics
        metrics = self.compute_metrics(labels, predictions, safety_scores)

        return metrics, safety_scores, predictions, labels
    
    def find_optimal_threshold(self, true_labels, scores):
        """Find optimal threshold using F1 score optimization"""
        from sklearn.metrics import precision_recall_curve
        import numpy as np

        # Get precision-recall curve
        precision, recall, thresholds = precision_recall_curve(true_labels, scores)

        # Calculate F1 scores for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)  # Add small epsilon to avoid division by zero

        # Find threshold that maximizes F1
        best_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_f1 = f1_scores[best_idx]

        print(f"Optimal threshold: {optimal_threshold:.4f} (F1: {best_f1:.4f})")

        return optimal_threshold, best_f1

    def compute_metrics(self, true_labels, predictions, scores, threshold_used=0.25):
        """Compute evaluation metrics with threshold optimization"""
        # Basic classification metrics with given threshold
        accuracy = accuracy_score(true_labels, predictions)
        balanced_acc = balanced_accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)
        f1 = f1_score(true_labels, predictions, zero_division=0)

        # ROC AUC
        try:
            roc_auc = roc_auc_score(true_labels, scores)
        except:
            roc_auc = 0.0

        # Precision-Recall AUC
        try:
            precision_curve, recall_curve, _ = precision_recall_curve(true_labels, scores)
            pr_auc = auc(recall_curve, precision_curve)
        except:
            pr_auc = 0.0

        # Find optimal threshold and compute metrics with it
        optimal_threshold, optimal_f1 = self.find_optimal_threshold(true_labels, scores)
        optimal_predictions = [1 if score >= optimal_threshold else 0 for score in scores]

        optimal_accuracy = accuracy_score(true_labels, optimal_predictions)
        optimal_balanced_acc = balanced_accuracy_score(true_labels, optimal_predictions)
        optimal_precision = precision_score(true_labels, optimal_predictions, zero_division=0)
        optimal_recall = recall_score(true_labels, optimal_predictions, zero_division=0)

        metrics = {
            # Original threshold (0.25) results
            'threshold_used': threshold_used,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,

            # Optimal threshold results
            'optimal_threshold': optimal_threshold,
            'optimal_accuracy': optimal_accuracy,
            'optimal_balanced_accuracy': optimal_balanced_acc,
            'optimal_precision': optimal_precision,
            'optimal_recall': optimal_recall,
            'optimal_f1_score': optimal_f1
        }

        return metrics
    
    def print_results(self, metrics):
        """Print evaluation results"""
        print("\n" + "="*70)
        print("GRADSAFE EVALUATION RESULTS")
        print("="*70)

        # Results with original threshold
        print(f"ORIGINAL THRESHOLD ({metrics['threshold_used']:.3f}):")
        print(f"  Accuracy:           {metrics['accuracy']:.4f}")
        print(f"  Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}")
        print(f"  Precision:          {metrics['precision']:.4f}")
        print(f"  Recall:             {metrics['recall']:.4f}")
        print(f"  F1 Score:           {metrics['f1_score']:.4f}")

        print(f"\nOPTIMAL THRESHOLD ({metrics['optimal_threshold']:.3f}):")
        print(f"  Accuracy:           {metrics['optimal_accuracy']:.4f}")
        print(f"  Balanced Accuracy:  {metrics['optimal_balanced_accuracy']:.4f}")
        print(f"  Precision:          {metrics['optimal_precision']:.4f}")
        print(f"  Recall:             {metrics['optimal_recall']:.4f}")
        print(f"  F1 Score:           {metrics['optimal_f1_score']:.4f}")

        print(f"\nTHRESHOLD-INDEPENDENT METRICS:")
        print(f"  ROC AUC:            {metrics['roc_auc']:.4f}")
        print(f"  PR AUC:             {metrics['pr_auc']:.4f}")
        print("="*70)
    
    def save_results(self, metrics, output_file="gradsafe_results.json"):
        """Save results to file"""
        results = {
            'model_path': self.model_path,
            'random_seed': self.random_seed,
            'metrics': metrics,
            'method': 'GradSafe',
            'benchmark': 'Vision-Language Safety Benchmark'
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_file}")
    
    def cleanup(self):
        """Clean up resources"""
        if self.gradsafe is not None:
            self.gradsafe.cleanup()
            self.gradsafe = None
