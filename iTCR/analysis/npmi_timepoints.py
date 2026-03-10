import pandas as pd
import numpy as np
import os
import tidytcells as tt
from typing import Literal
from joblib import Parallel, delayed
import multiprocessing as mp
from collections import defaultdict
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize
import seaborn as sns
from scipy import stats
import pickle
from itertools import combinations
from matplotlib.colors import LinearSegmentedColormap
from statsmodels.stats.multitest import multipletests
from scipy.stats import mannwhitneyu, ranksums
import argparse
from .sample_parser import create_sample_mapping
  
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'

def parse_sample_id(sample_id):
    """
    Parse sample ID to extract patient ID and timepoint
    MODIFY THIS FUNCTION according to your sample naming convention
    
    Args:
        sample_id (str): Sample identifier
        
    Returns:
        tuple: (patient_id, timepoint)
        
    Example:
        "UPN24 pretreatment" -> ("UPN24", "pretreatment")
        "UPN24 posttreatment" -> ("UPN24", "posttreatment")
    """
    parts = sample_id.split(' ')
    if len(parts) >= 2:
        patient_id = parts[0]  # UPN24
        timepoint = parts[1]   # pretreatment/posttreatment
        return patient_id, timepoint
    else:
        return sample_id, 'unknown'

def get_patient_pairs(sample_ids):
    """
    Identify patients with both pre and post treatment samples
    MODIFY THIS FUNCTION according to your sample naming convention
    
    Args:
        sample_ids (list): List of sample identifiers
        
    Returns:
        dict: {patient_id: {'pretreatment': sample_id, 'posttreatment': sample_id}}
    """
    sample_mapping = create_sample_mapping()
    patient_timepoints = defaultdict(dict)
    
    for sample_id in sample_ids:
        patient_id, timepoint = parse_sample_id(sample_id)
        if timepoint in ['pretreatment', 'posttreatment']:
            patient_timepoints[patient_id][timepoint] = sample_id
    
    # Filter to only patients with both pre and post
    paired_patients = {}
    for patient_id, timepoints in patient_timepoints.items():
        if 'pretreatment' in timepoints and 'posttreatment' in timepoints:
            try:
                # Also check if patient has posttreatment data in sample_mapping
                if patient_id in sample_mapping and sample_mapping[patient_id]['posttreatment'] is None:
                    continue
            except:
                pass
            paired_patients[patient_id] = timepoints
    
    return paired_patients

def calculate_cliff_delta(group1, group2):
    n1, n2 = len(group1), len(group2)
    
    more = 0
    less = 0
    
    for x in group2:
        for y in group1:
            if x > y:
                more += 1
            elif x < y:
                less += 1
    
    delta = (more - less) / (n1 * n2)
    return abs(delta)

def load_data(npmi_path, samples_path):
    """Load NPMI data and sample data"""
    npmi_data = pd.read_pickle(npmi_path)
    data_dict = pd.read_pickle(samples_path)
    return npmi_data, data_dict

def permutation_test_matrices(pre_matrices, post_matrices, 
                                          n_permutations=10000, 
                                          random_state=42,
                                          correction_method='fdr_bh',
                                          min_effect_size=0.5):
    """
    Perform Mann-Whitney U test with rank-biserial correlation effect size
    
    Parameters:
    -----------
    pre_matrices : list of arrays
        Pre-treatment gene expression matrices
    post_matrices : list of arrays
        Post-treatment gene expression matrices
    n_permutations : int, default=10000
        Number of permutations (not used in current implementation)
    random_state : int, default=42
        Random seed for reproducibility
    correction_method : str, default='fdr_bh'
        Multiple testing correction method ('fdr_bh', 'bonferroni', etc.)
    min_effect_size : float, default=0.2
        Minimum threshold for Cliff's Delta or rank-biserial correlation
    
    Returns:
    --------
    significant_positions : ndarray
        Boolean matrix indicating significant gene pairs
    corrected_p_matrix : ndarray
        Matrix of corrected p-values
    observed_diff : ndarray
        Matrix of observed mean differences (post - pre)
    effect_sizes : ndarray
        Matrix of effect sizes (Cliff's Delta)
    """
    if len(pre_matrices) == 0 or len(post_matrices) == 0:
        return None, None, None
    
    np.random.seed(random_state)
    
    pre_stack = np.stack(pre_matrices)
    post_stack = np.stack(post_matrices)
    
    n_genes1, n_genes2 = pre_stack.shape[1], pre_stack.shape[2]
    min_observations = int(pre_stack.shape[0] * 0.5)
    
    pre_means = np.mean(pre_stack, axis=0)
    post_means = np.mean(post_stack, axis=0)
    observed_diff = post_means - pre_means
    
    p_values = np.full((n_genes1, n_genes2), np.nan)
    effect_sizes = np.full((n_genes1, n_genes2), np.nan)
    significant_positions = np.zeros((n_genes1, n_genes2), dtype=bool)
    
    for i in range(n_genes1):
        for j in range(n_genes2):
            pre_values = pre_stack[:, i, j]
            post_values = post_stack[:, i, j]

            valid1 = ~np.isnan(pre_values)
            valid2 = ~np.isnan(post_values)
            
            n_obs1 = np.sum(valid1)
            n_obs2 = np.sum(valid2)
            
            if n_obs1 >= min_observations and n_obs2 >= min_observations:
                valid_values1 = pre_values[valid1]
                valid_values2 = post_values[valid2]
                
                try:
                    statistic, p_val = mannwhitneyu(
                        valid_values1, 
                        valid_values2, 
                        alternative='two-sided'
                    )
                    p_values[i, j] = p_val
                    effect_size = calculate_cliff_delta(valid_values1, valid_values2)
                    effect_sizes[i, j] = effect_size
                    
                    if p_val < 0.05 and effect_size >= min_effect_size:
                        significant_positions[i, j] = True
                        
                except Exception as e:
                    print(f"Warning at position ({i},{j}): {e}")
                    continue
    
    # Multiple testing correction
    if np.any(~np.isnan(p_values)):
        original_shape = p_values.shape
        
        p_values_flat = p_values.flatten()
        valid_mask = ~np.isnan(p_values_flat)
        valid_p_values = p_values_flat[valid_mask]
        
        rejected, corrected_p_values, _, _ = multipletests(
            valid_p_values, 
            alpha=0.05, 
            method=correction_method
        )
        
        corrected_p_flat = np.full_like(p_values_flat, np.nan)
        corrected_p_flat[valid_mask] = corrected_p_values
        
        rejected_flat = np.zeros_like(p_values_flat, dtype=bool)
        rejected_flat[valid_mask] = rejected
        
        # Re-evaluate significance: require both corrected p-value and effect size thresholds
        effect_sizes_flat = effect_sizes.flatten()
        final_significant = (rejected_flat & 
                           (effect_sizes_flat >= min_effect_size))

        corrected_p_matrix = corrected_p_flat.reshape(original_shape)
        significant_positions = final_significant.reshape(original_shape)
        
        return significant_positions, corrected_p_matrix, observed_diff, effect_sizes
    
    return significant_positions, p_values, observed_diff, effect_sizes

def process_single_patient_feature_pair(patient_id, feature_pair, timepoints, sample_ids, 
                                       sample_data, n_resamples, n_permutations):
    """
    Process a single patient-feature pair combination
    """
    sample_mapping = create_sample_mapping()
    
    pre_sample_id = timepoints['pretreatment']
    post_sample_id = timepoints['posttreatment']
    
    # Find indices of pre and post samples
    try:
        pre_idx = sample_ids.index(pre_sample_id)
        post_idx = sample_ids.index(post_sample_id)
    except ValueError as e:
        return patient_id, feature_pair, None, f"Could not find sample indices: {e}"
    
    # Collect NPMI matrices for all resamples
    pre_matrices = []
    post_matrices = []
    
    for resample_idx in range(n_resamples):
        if resample_idx in sample_data:
            matrices = sample_data[resample_idx]
            if pre_idx < len(matrices) and post_idx < len(matrices):
                pre_matrices.append(matrices[pre_idx])
                post_matrices.append(matrices[post_idx])
    
    if len(pre_matrices) == 0 or len(post_matrices) == 0:
        return patient_id, feature_pair, None, "No matrices found"
    
    # Perform permutation test
    significant_positions, p_values, observed_diff, effect_sizes = permutation_test_matrices(
        pre_matrices, post_matrices, n_permutations, random_state=42
    )
    
    if significant_positions is not None:
        # Count significant changes
        significant_count = np.sum(significant_positions)
        total_tested = np.sum(~np.isnan(p_values))
        
        result = {
            'significant_count': significant_count,
            'total_tested': total_tested,
            'significant_positions': significant_positions,
            'p_values': p_values,
            'effect_sizes':effect_sizes,
            'observed_differences': observed_diff,
            'pre_sample_id': pre_sample_id,
            'post_sample_id': post_sample_id,
            'response_info': sample_mapping.get(patient_id, {})
        }
        return patient_id, feature_pair, result, None
    else:
        return patient_id, feature_pair, None, "Could not perform test"

def analyze_patient_npmi_changes(npmi_data, data_dict, n_permutations=10000, alpha=0.05, n_jobs=-1):
    """
    Analyze NPMI changes for each patient across all gene pairs using permutation tests
    Accelerated with joblib parallel processing
    
    Parameters:
    -----------
    n_jobs : int, default=-1
        Number of parallel jobs. -1 means use all available cores
    
    Returns:
    --------
    dict: {patient_id: {feature_pair: {'significant_count': int, 'total_tested': int, 
                                      'significant_positions': array, 'p_values': array, 
                                      'observed_differences': array}}}
    """
    # Get sample IDs from data_dict
    sample_ids = list(data_dict.keys())
    print(f"Total samples available: {len(sample_ids)}")
    
    # Get patients with both pre and post samples
    paired_patients = get_patient_pairs(sample_ids)
    print(f"Patients with both pre and post samples: {list(paired_patients.keys())}")
    
    # Prepare all tasks for parallel processing
    tasks = []
    
    for feature_pair in npmi_data.keys():
        feature1, feature2 = feature_pair
        print(f"Preparing tasks for feature pair: {feature1} vs {feature2}")
        
        # Get sample data for this feature pair
        sample_data = npmi_data[feature_pair]
        n_resamples = len(sample_data)
        
        for patient_id, timepoints in paired_patients.items():
            tasks.append((
                patient_id, feature_pair, timepoints, sample_ids, 
                sample_data, n_resamples, n_permutations
            ))
    
    print(f"Total tasks to process: {len(tasks)}")
    print(f"Running with {n_jobs} parallel jobs...")
    
    # Execute tasks in parallel
    parallel_results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(process_single_patient_feature_pair)(
            patient_id, feature_pair, timepoints, sample_ids,
            sample_data, n_resamples, n_permutations
        ) for patient_id, feature_pair, timepoints, sample_ids, 
             sample_data, n_resamples, n_permutations in tasks
    )
    
    # Organize results
    results = defaultdict(lambda: defaultdict(dict))
    
    for patient_id, feature_pair, result, error in parallel_results:
        if result is not None:
            results[patient_id][feature_pair] = result
        else:
            print(f"Warning: Failed to process {patient_id} - {feature_pair}: {error}")
    
    print(f"\nCompleted processing for {len(results)} patients")
    
    return results

def summarize_patient_changes(analysis_results):
    """
    Summarize significant changes across all feature pairs for each patient
    
    Returns:
    --------
    pandas.DataFrame: Summary of changes per patient
    """
    summary_data = []
    sample_mapping = create_sample_mapping()
    
    for patient_id, feature_results in analysis_results.items():
        total_significant = 0
        total_tested = 0
        feature_counts = {}
        
        # Get patient info from sample_mapping
        patient_info = sample_mapping.get(patient_id, {})
        
        for feature_pair, results in feature_results.items():
            feature_name = f"{feature_pair[0]}_{feature_pair[1]}"
            significant_count = results['significant_count']
            tested_count = results['total_tested']
            
            total_significant += significant_count
            total_tested += tested_count
            feature_counts[f"{feature_name}_significant"] = significant_count
            feature_counts[f"{feature_name}_tested"] = tested_count
            feature_counts[f"{feature_name}_percentage"] = (significant_count / tested_count * 100) if tested_count > 0 else 0
        
        summary_row = {
            'Patient_ID': patient_id,
            'Total_Significant_Changes': total_significant,
            'Total_Gene_Pairs_Tested': total_tested,
            'Overall_Percentage_Changed': (total_significant / total_tested * 100) if total_tested > 0 else 0,
            'response_info': patient_info,
            **feature_counts
        }
        
        summary_data.append(summary_row)
    
    return pd.DataFrame(summary_data)

def save_results(analysis_results, summary_df, output_dir):
    """Save analysis results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert defaultdict to regular dict for pickling
    regular_dict_results = {}
    for patient_id, feature_results in analysis_results.items():
        regular_dict_results[patient_id] = dict(feature_results)
    
    # Save detailed results
    with open(os.path.join(output_dir, 'patient_PLS_detailed.pickle'), 'wb') as f:
        pickle.dump(regular_dict_results, f)
    
    # Save summary
    summary_df.to_csv(os.path.join(output_dir, 'patient_PLS_summary.csv'), index=False)
def main():
    parser = argparse.ArgumentParser(description='Analyze patient NPMI changes across timepoints')
    
    parser.add_argument('--npmi-data', 
                        type=str, 
                        required=True,
                        help='Path to NPMI data pickle file')
    
    parser.add_argument('--data-dict', 
                        type=str, 
                        required=True,
                        help='Path to data dictionary pickle file')
    
    parser.add_argument('--output-dir', 
                        type=str, 
                        required=True,
                        help='Directory to save results')
    
    parser.add_argument('--n-permutations', 
                        type=int, 
                        default=10000,
                        help='Number of permutations for statistical testing (default: 10000)')
    
    parser.add_argument('--n-jobs', 
                        type=int, 
                        default=-1,
                        help='Number of parallel jobs (default: -1, use all cores)')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading NPMI data from: {args.npmi_data}")
    npmi_data = pd.read_pickle(args.npmi_data)
    
    print(f"Loading data dictionary from: {args.data_dict}")
    data_dict = pd.read_pickle(args.data_dict)
    
    # Analyze changes
    print(f"Analyzing NPMI changes with {args.n_permutations} permutations...")
    results = analyze_patient_npmi_changes(npmi_data=npmi_data, data_dict=data_dict, 
                                         n_permutations=args.n_permutations, n_jobs=args.n_jobs)
    
    # Summarize results
    print("Summarizing results...")
    summary_df = summarize_patient_changes(results)
    print(summary_df)
    
    # Save results
    print(f"Saving results to: {args.output_dir}")
    save_results(results, summary_df, args.output_dir)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()
