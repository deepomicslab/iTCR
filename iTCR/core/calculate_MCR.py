"""
Main script for TCR entropy and mutual information analysis
"""
import pandas as pd
import numpy as np
import os
import multiprocessing
from joblib import Parallel, delayed
import pickle
from .. import utils
from typing import Literal, Dict, Any, Tuple
import gc
from collections import defaultdict
import argparse
import time
import random
from ..config import (SINGLE_FEATURES, CONDITIONAL_FEATURES, CROSS_FEATURES)

random.seed(0)
np.random.seed(0)

def get_corrected_for_different_samples_accelerated(df_dict, col1, col2=None, P=1000, alpha=0.05, sample_times=100, nested_resamples=50, 
                                sample_weights=None, random_state=0, base=np.e, method: Literal["percentile", "basic"] = "basic",
                                use_gpu=False, statistic_type: Literal["mcr", "entropy", "mi"] = "mcr", inner_jobs=None, outer_jobs=None, n_jobs=-1):
    """
    Two-level parallelized comparison of statistics between different HLA types

    Parameters:
    df_dict: Dictionary of dataframes
    col1, col2: The names of the two columns to calculate statistics
    P: Number of bootstrap samples
    alpha: Significance level for confidence intervals
    sample_times: Number of times to repeat sampling
    nested_resamples: Number of nested resamples for bootstrap
    sample_weights: Column name for sampling weights
    random_state: Random seed for reproducibility
    base: Base for logarithm in mutual information calculation
    n_jobs: Number of parallel jobs for inner parallelization
    outer_jobs: Number of parallel jobs for outer sampling loop
    method: Method for bootstrap ("percentile" or "basic")
    use_gpu: Whether to use GPU acceleration
    statistic_type: Type of statistic to calculate ("mcr", "entropy", or "mi")

    Returns:
    Dictionary containing statistic values for each iteration and HLA type
    """

    individuals = list(df_dict.keys())
    min_shape = min(len(df_indi) for df_indi in df_dict.values())
    if (min_shape < 1000) & (min_shape > 200):
        min_shape -= 50
    elif (min_shape >= 1000) & (min_shape < 10000):
        min_shape -= 100
    elif min_shape > 10000:
        min_shape -= 1000
    else:
        pass

    # Determine optimal parallelization strategy
    total_cpus = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
    print(f"Total available CPU cores: {total_cpus}")
    
    # Calculate outer_jobs if not specified
    if outer_jobs is None:
        outer_jobs = min(4, total_cpus // 2)
    # Calculate inner_jobs if not specified
    if inner_jobs is None:
        # Use at least 2 cores per inner job, but don't exceed what's available divided by outer jobs
        inner_jobs = min(4, total_cpus // outer_jobs)
    
    print(f"Running with {outer_jobs} outer parallel jobs, each using {inner_jobs} cores")
    
    # Function to process one sample iteration
    def process_sample_iteration(iteration_idx):
        # Set seed for this iteration
        iter_seed = random_state + iteration_idx
        np.random.seed(iter_seed)
        
        # Sample from each dataframe
        sampled_dfs = {}
        if min_shape is not None:
            for indi in individuals:
                df_indi = df_dict[indi]
                if df_indi['sample_ID'].nunique() == 1:
                    if len(df_indi) > min_shape:
                        sampled_dfs[indi] = df_indi.sample(n=min_shape, weights=df_indi[sample_weights].values if sample_weights else None, random_state=iter_seed)
                    else:
                        sampled_dfs[indi] = df_dict[indi]
                else:
                    if len(df_indi) > min_shape:
                        sample_counts = df_indi['sample_ID'].value_counts()
                        total_size = len(df_indi)
                        stratified_samples = []
                        sampled_indices = set()
                        
                        for sample_id, count in sample_counts.items():
                            sample_data = df_indi[df_indi['sample_ID'] == sample_id]
                            sample_target = int(min_shape * count / total_size)
                            
                            if sample_target > 0:
                                if len(sample_data) > sample_target:
                                    sampled_sample = sample_data.sample(
                                        n=sample_target,
                                        weights=sample_data[sample_weights].values if sample_weights else None,
                                        random_state=iter_seed
                                    )
                                else:
                                    sampled_sample = sample_data
                                
                                stratified_samples.append(sampled_sample)
                                sampled_indices.update(sampled_sample.index)
                        
                        if stratified_samples:
                            current_sampled = pd.concat(stratified_samples, ignore_index=False)
                        
                        current_size = len(current_sampled)
                        if current_size < min_shape:
                            remaining_needed = min_shape - current_size
                            unsampled_data = df_indi[~df_indi.index.isin(sampled_indices)]

                            if len(unsampled_data) > 0:
                                additional_sample = unsampled_data.sample(
                                    n=remaining_needed,
                                    weights=unsampled_data[sample_weights].values if sample_weights else None,
                                    random_state=iter_seed
                                )
                                current_sampled = pd.concat([current_sampled, additional_sample], ignore_index=False)
                        
                        sampled_dfs[indi] = current_sampled.reset_index(drop=True)
                    else:
                        sampled_dfs[indi] = df_dict[indi]
                            
        else:
            for indi in individuals:
                sampled_dfs[indi] = df_dict[indi]

        # Process function for each dataframe
        def process_df(temp_df, seed):
            indi, temp_df = temp_df
            if method == "percentile":
                return indi, utils.percentile_t_bootstrap_parallel(
                    temp_df, col1, col2, P=P, alpha=alpha, 
                    nested_resamples=nested_resamples, random_state=seed, 
                    base=base, n_jobs=-1, use_gpu=use_gpu
                )
            else:
                if statistic_type == "mcr":
                    mi, val2, val3 = utils.basic_bootstrap_statistic(
                            temp_df, col1, col2, base=base, n_resamples=P, 
                            random_state=seed, use_gpu=use_gpu, statistic_type="mi"
                        )
                    return indi, np.exp(-mi), np.exp(-val2), np.exp(-val3)
                else:
                    mi, val2, val3 = utils.basic_bootstrap_statistic(
                            temp_df, col1, col2, base=base, n_resamples=P, 
                            random_state=seed, use_gpu=use_gpu, statistic_type=statistic_type
                        )
                    return indi, mi, val2, val3
        # Seeds for each dataframe
        hla_seeds = [iter_seed + j for j in range(len(sampled_dfs))]
        
        # Inner parallelization - process dataframes in parallel
        results = Parallel(n_jobs=inner_jobs)(
            delayed(process_df)(df_indi, seed) 
            for df_indi, seed in zip(sampled_dfs.items(), hla_seeds)
        )
        
        # Extract MI values
        stat_values = {indi: stat for indi, stat, _, _ in results}
        print(stat_values)
        #print(f"Completed iteration {iteration_idx} (PID: {os.getpid()})")
        
        del results, sampled_dfs
        gc.collect()
        
        return iteration_idx, stat_values
    
    # Outer parallelization - process sample iterations in parallel
    print(f"Running {sample_times} iterations in parallel...")
    results = Parallel(n_jobs=outer_jobs, verbose=10)(
        delayed(process_sample_iteration)(i) 
        for i in range(sample_times)
    )
    
    # mcr_all results
    stat_all = {}
    for iteration_idx, sample_result in results:
        stat_all[iteration_idx] = sample_result
    
    return stat_all 

def format_time(seconds):
    """Format seconds into human-readable time"""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {secs:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {secs:.1f}s"

def calculate_entropy_statistics(df_dict, args) -> Dict[str, Any]:
    """Calculate entropy statistics for single features and conditional entropies"""
    print("=== CALCULATING ENTROPY STATISTICS ===")
    entropy_all = {}
    
    # Calculate single feature entropies
    print(f"Processing {len(SINGLE_FEATURES)} single feature entropies...")
    single_start_time = time.time()
    
    for i, feature in enumerate(SINGLE_FEATURES, 1):
        feature_start_time = time.time()
        print(f"  [{i}/{len(SINGLE_FEATURES)}] Processing entropy for: {feature}")
        
        entropy_all[feature] = get_corrected_for_different_samples_accelerated(
            df_dict, col1=feature, sample_times=args.sample_times, 
            method="basic", statistic_type="entropy", sample_weights=args.sample_weights,
            inner_jobs=args.inner_jobs, outer_jobs=args.outer_jobs
        )
        
        feature_end_time = time.time()
        feature_duration = feature_end_time - feature_start_time
        print(f"  ✓ Completed {feature} in {format_time(feature_duration)}")
    
    single_end_time = time.time()
    single_total_time = single_end_time - single_start_time
    print(f"✓ All single feature entropies completed in {format_time(single_total_time)}")
    
    # Calculate conditional entropies
    print(f"\nProcessing {len(CONDITIONAL_FEATURES)} conditional entropies...")
    conditional_start_time = time.time()
    
    for i, (feature1, condition) in enumerate(CONDITIONAL_FEATURES, 1):
        feature_start_time = time.time()
        print(f"  [{i}/{len(CONDITIONAL_FEATURES)}] Processing conditional entropy: {feature1}|{condition}")
        
        entropy_all[f"{feature1}|{condition}"] = get_corrected_for_different_samples_accelerated(
            df_dict, col1=feature1, col2=condition, sample_times=args.sample_times, 
            method="basic", statistic_type="entropy", sample_weights=args.sample_weights,
            inner_jobs=args.inner_jobs, outer_jobs=args.outer_jobs
        )
        
        feature_end_time = time.time()
        feature_duration = feature_end_time - feature_start_time
        print(f"  ✓ Completed {feature1}|{condition} in {format_time(feature_duration)}")
    
    conditional_end_time = time.time()
    conditional_total_time = conditional_end_time - conditional_start_time
    print(f"✓ All conditional entropies completed in {format_time(conditional_total_time)}")
    print("=== ENTROPY CALCULATIONS COMPLETED ===")
    
    return entropy_all

def calculate_mcr_statistics(df_dict, args) -> Dict[Tuple[str, str], Any]:
    """Calculate mutual information statistics for feature pairs"""
    print("=== CALCULATING MUTUAL INFORMATION STATISTICS ===")
    mcr_all = {}
    
    # Calculate mutual information for all feature pairs
    all_mi_features = CROSS_FEATURES
    
    print(f"Total MI calculations to perform: {len(all_mi_features)}")
    mi_start_time = time.time()
    
    for i, (feature1, feature2) in enumerate(all_mi_features, 1):
        feature_start_time = time.time()
        print(f"  [{i}/{len(all_mi_features)}] Processing MCR for: {feature1} - {feature2}")
        
        mcr_all[(feature1, feature2)] = get_corrected_for_different_samples_accelerated(
            df_dict, feature1, feature2, sample_times=args.sample_times, 
            method="basic", statistic_type="mcr", sample_weights=args.sample_weights,
            inner_jobs=args.inner_jobs, outer_jobs=args.outer_jobs
        )
        
        feature_end_time = time.time()
        feature_duration = feature_end_time - feature_start_time
        print(f"  ✓ Completed {feature1} - {feature2} in {format_time(feature_duration)}")
    
    mi_end_time = time.time()
    mi_total_time = mi_end_time - mi_start_time
    print(f"✓ All MCR calculations completed in {format_time(mi_total_time)}")
    print("=== MCR CALCULATIONS COMPLETED ===")
    
    return mcr_all

def save_data(data, output_path):
    """Save data to a pickle file"""
    print(f"Saving data to {output_path}...")
    save_start = time.time()
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    save_end = time.time()
    print(f"✓ Data saved in {format_time(save_end - save_start)}")

def load_pickle(filepath: str):
    """Load an object from a pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def process_single_group(df_dict, args):
    """Process a single group of data"""
    print("Processing single group...")
    entropy_all = {}
    mcr_all = {}
    
    for indi, df in df_dict.items(): 
        df['Alpha'] = df['TRAV'].str.cat([df['cdr3A'], df['TRAJ']], sep='_')
        df['Beta'] = df['TRBV'].str.cat([df['cdr3B'], df['TRBJ']], sep='_')
    
    # Calculate entropy statistics if requested
    if args.analysis_type in ['entropy', 'both']:
        entropy_start = time.time()
        entropy_all = calculate_entropy_statistics(df_dict, args)
        entropy_end = time.time()
        print(f"Total entropy analysis time: {format_time(entropy_end - entropy_start)}")
    
    # Calculate MI statistics if requested
    if args.analysis_type in ['mcr', 'both']:
        mi_start = time.time()
        mcr_all = calculate_mcr_statistics(df_dict, args)
        mi_end = time.time()
        print(f"Total MI analysis time: {format_time(mi_end - mi_start)}")
    
    return entropy_all, mcr_all

def process_multiple_groups(df_dict, args):
    """Process multiple groups of data"""
    print("Processing multiple groups...")
    entropy_all = defaultdict(dict)
    mcr_all = defaultdict(dict)
    
    for group_idx, (group, sub_dict) in enumerate(df_dict.items(), 1):
        group_start_time = time.time()
        print(f"\n=== PROCESSING GROUP {group_idx}/{len(df_dict)}: {group} ===")
        group_entropy, group_mi = process_single_group(sub_dict, args)
        
        # Store results by feature then by group
        for feature, values in group_entropy.items():
            entropy_all[feature][group] = values
        
        for feature_pair, values in group_mi.items():
            mcr_all[feature_pair][group] = values
        
        group_end_time = time.time()
        group_duration = group_end_time - group_start_time
        print(f"✓ Group {group} completed in {format_time(group_duration)}")
    
    return entropy_all, mcr_all

def main():
    parser = argparse.ArgumentParser(description='TCR entropy and mutual information analysis')
    parser.add_argument('--sample_times', type=int, default=100,
                       help='Number of bootstrap samples')
    parser.add_argument('--sample_weights', type=str, default='clonotype.freq',
                       help='sample_weights')
    parser.add_argument('--outer_jobs', type=int, default=8, 
                       help='Number of outer permutation tasks to run in parallel')
    parser.add_argument('--inner_jobs', type=int, default=None, 
                       help='Number of cores per permutation task')
    parser.add_argument('--inputfile', type=str, required=True,
                       help='Path to input pickle file')
    parser.add_argument('--outputdir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--analysis_type', type=str, choices=['entropy', 'mcr', 'both'], 
                       default='both', help='Type of analysis to run')
    
    args = parser.parse_args()
    
    total_start_time = time.time()
    
    print(f"=== STARTING TCR ANALYSIS ===")
    print(f"Analysis type: {args.analysis_type}")
    print(f"Sample times: {args.sample_times}")
    print(f"Outer jobs: {args.outer_jobs}")
    print(f"Inner jobs: {args.inner_jobs}")
    print(f"Loading data from: {args.inputfile}")
    
    # Data loading
    print("\nLoading data...")
    data_load_start = time.time()
    df_dict = pd.read_pickle(args.inputfile)
    data_load_end = time.time()
    print(f"✓ Data loading completed in {format_time(data_load_end - data_load_start)}")
    
    # Determine if single group or multiple groups
    if isinstance(next(iter(df_dict.values())), pd.DataFrame):
        # Single group case
        entropy_all, mcr_all = process_single_group(df_dict, args)
    elif isinstance(next(iter(df_dict.values())), dict):
        # Multiple groups case
        entropy_all, mcr_all = process_multiple_groups(df_dict, args)
    else:
        raise ValueError("Unexpected data structure in input file")
    
    # Save results based on analysis type
    save_start_time = time.time()
    if args.analysis_type in ['entropy', 'both'] and entropy_all:
        save_data(entropy_all, os.path.join(args.outputdir, 'entropy.pickle'))
        print("✓ Entropy analysis completed and saved!")
    
    if args.analysis_type in ['mcr', 'both'] and mcr_all:
        save_data(mcr_all, os.path.join(args.outputdir, 'mcr.pickle'))
        print("✓ MCR analysis completed and saved!")
    
    save_end_time = time.time()
    print(f"✓ All data saving completed in {format_time(save_end_time - save_start_time)}")
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"\n=== ANALYSIS COMPLETED SUCCESSFULLY ===")
    print(f"Total execution time: {format_time(total_duration)}")

if __name__ == "__main__":
    main()