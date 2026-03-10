"""
Main script for TCR normalized pointwise mutual information analysis
"""
import pandas as pd
import numpy as np
import os
import multiprocessing
from joblib import Parallel, delayed
import pickle
from .. import utils
from typing import Literal, Dict, Any, Tuple, List
import gc
from collections import defaultdict
import argparse
import time
import tidytcells as tt

def get_functional_genes():
    """Get functional gene lists from topnotch"""
    F_TRBVs = tt.tr.query(contains_pattern="TRBV", functionality="F", precision="gene")
    F_TRAVs = tt.tr.query(contains_pattern="TRAV", functionality="F", precision="gene")
    F_TRBJs = tt.tr.query(contains_pattern="TRBJ", functionality="F", precision="gene")
    F_TRAJs = tt.tr.query(contains_pattern="TRAJ", functionality="F", precision="gene")
    
    return sorted(F_TRBVs), sorted(F_TRAVs), sorted(F_TRBJs), sorted(F_TRAJs)

# Define cross-chain feature pairs for NPMI analysis
NPMI_FEATURES = [
    ('TRAV', 'TRBV'),  # F_TRAVs vs F_TRBVs
    ('TRAJ', 'TRBJ')   # F_TRAJs vs F_TRBJs
]

def filter_functional_genes(df, col, functional_genes):
    """Filter dataframe to only include functional genes"""
    return df[df[col].isin(functional_genes)]

def calculate_npmi_matrix_ordered(df, col1, col2, functional_genes1, functional_genes2, base=np.e):
    """
    Calculate NPMI matrix for functional gene pairs in the order of functional gene lists
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe for one sample
    col1, col2 : str
        Column names for the two gene types
    functional_genes1, functional_genes2 : list
        Lists of functional genes for each column (in desired order)
    base : float
        Logarithm base for NPMI calculation
        
    Returns:
    --------
    numpy.ndarray
        2D numpy array with NPMI values (0 for missing gene pairs)
    """
    # Filter dataframe to only include functional genes
    df_filtered = df.copy()
    df_filtered = filter_functional_genes(df_filtered, col1, functional_genes1)
    df_filtered = filter_functional_genes(df_filtered, col2, functional_genes2)
    
    # Initialize NPMI matrix with 0 values (for missing gene pairs)
    npmi_matrix = np.full((len(functional_genes1), len(functional_genes2)), 0.0)
    
    if len(df_filtered) == 0:
        return npmi_matrix
    
    # Calculate NPMI using the provided function
    npmi_df = utils.return_normalized_pointwise_mutual_information(
        df_filtered, col1, col2, base=base, use_gpu=False
    )
    
    # Create a dictionary for quick lookup of NPMI values
    npmi_dict = {}
    for _, row in npmi_df.iterrows():
        gene1, gene2, npmi_val = row[col1], row[col2], row['normalized_pointwise']
        npmi_dict[(gene1, gene2)] = npmi_val
    
    # Fill matrix in the order of functional gene lists
    for i, gene1 in enumerate(functional_genes1):
        for j, gene2 in enumerate(functional_genes2):
            if (gene1, gene2) in npmi_dict:
                npmi_matrix[i, j] = npmi_dict[(gene1, gene2)]
            # If gene pair not found, it remains 0 (default value)
    
    return npmi_matrix

def get_corrected_npmi_for_different_samples_accelerated(df_dict, col1, col2, functional_genes1, functional_genes2,
                                                        sample_times=100, sample_weights=None, random_state=0, base=np.e,
                                                        inner_jobs=None, outer_jobs=None, n_jobs=-1):
    """
    Two-level parallelized NPMI calculation with downsampling and multiple resampling
    
    Parameters:
    -----------
    df_dict: Dictionary of dataframes
    col1, col2: The names of the two columns to calculate NPMI for
    functional_genes1, functional_genes2: Lists of functional genes
    sample_times: Number of times to repeat sampling
    sample_weights: Column name for sampling weights
    random_state: Random seed for reproducibility
    base: Base for logarithm in NPMI calculation
    inner_jobs: Number of cores for inner parallelization
    outer_jobs: Number of parallel jobs for outer sampling loop
    n_jobs: Total number of parallel jobs
    
    Returns:
    --------
    Dictionary: {iteration_idx: array of NPMI matrices for all samples}
    """
    individuals = list(df_dict.keys())
    min_shape = min(len(df_indi) for df_indi in df_dict.values())
    
    # Apply the same min_shape adjustment logic as the original code
    if (min_shape < 1000) & (min_shape > 200):
        min_shape -= 50
    elif (min_shape >= 1000) & (min_shape < 10000):
        min_shape -= 100
    elif min_shape > 10000:
        min_shape -= 1000
    else:
        pass
    print(f"Downsampling to size: {min_shape}")
    
    # Determine optimal parallelization strategy
    total_cpus = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
    print(f"Total available CPU cores: {total_cpus}")
    
    # Calculate outer_jobs if not specified
    if outer_jobs is None:
        outer_jobs = min(4, total_cpus // 2)
    # Calculate inner_jobs if not specified
    if inner_jobs is None:
        # Use at least 2 cores per inner job, but don't exceed what's available divided by outer jobs
        inner_jobs = min(2, total_cpus // outer_jobs)
    
    print(f"Running with {outer_jobs} outer parallel jobs, each using {inner_jobs} cores")
    
    # Function to process one sample iteration
    def process_sample_iteration(iteration_idx):
        # Set seed for this iteration
        iter_seed = random_state + iteration_idx
        np.random.seed(iter_seed)
        
        # Sample from each dataframe
        sampled_dfs = []
        if min_shape is not None:
            for indi in individuals:
                df_indi = df_dict[indi]
                if len(df_indi) > min_shape:
                    sampled_dfs.append(df_indi.sample(n=min_shape, weights=df_indi[sample_weights].values if sample_weights else None, random_state=iter_seed))
                else:
                    sampled_dfs.append(df_dict[indi])
        else:
            for indi in individuals:
                sampled_dfs.append(df_dict[indi])

        # Process function for each dataframe
        def process_df(hla_df, seed):
            return calculate_npmi_matrix_ordered(
                hla_df, col1, col2, functional_genes1, functional_genes2, base
            )
        
        # Seeds for each dataframe
        data_seeds = [iter_seed + j for j in range(len(sampled_dfs))]
        
        # Inner parallelization - process dataframes in parallel
        results = Parallel(n_jobs=inner_jobs)(
            delayed(process_df)(df_indi, seed) 
            for df_indi, seed in zip(sampled_dfs, data_seeds)
        )
        
        # Convert results to numpy array
        npmi_values = np.array(results)
        
        del results, sampled_dfs
        gc.collect()
        
        return iteration_idx, npmi_values
    
    # Outer parallelization - process sample iterations in parallel
    print(f"Running {sample_times} iterations in parallel...")
    results = Parallel(n_jobs=outer_jobs, verbose=10)(
        delayed(process_sample_iteration)(i) 
        for i in range(sample_times)
    )
    
    # npmi_all results
    npmi_all = {}
    for iteration_idx, sample_result in results:
        npmi_all[iteration_idx] = sample_result
    
    return npmi_all

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

def calculate_npmi_statistics(df_dict, args) -> Dict[Tuple[str, str], Any]:
    """Calculate NPMI statistics for functional gene pairs"""
    print("=== CALCULATING NPMI STATISTICS ===")
    
    # Get functional genes
    F_TRBVs, F_TRAVs, F_TRBJs, F_TRAJs = get_functional_genes()
    
    print(f"Functional genes loaded:")
    print(f"  F_TRAVs: {len(F_TRAVs)} genes")
    print(f"  F_TRBVs: {len(F_TRBVs)} genes") 
    print(f"  F_TRAJs: {len(F_TRAJs)} genes")
    print(f"  F_TRBJs: {len(F_TRBJs)} genes")
    
    npmi_all = {}
    
    print(f"Total NPMI calculations to perform: {len(NPMI_FEATURES)}")
    npmi_start_time = time.time()
    
    for i, (feature1, feature2) in enumerate(NPMI_FEATURES, 1):
        feature_start_time = time.time()
        print(f"  [{i}/{len(NPMI_FEATURES)}] Processing NPMI for: {feature1} - {feature2}")
        
        # Get corresponding functional genes
        if feature1 == 'TRAV' and feature2 == 'TRBV':
            functional_genes1, functional_genes2 = F_TRAVs, F_TRBVs
        elif feature1 == 'TRAJ' and feature2 == 'TRBJ':
            functional_genes1, functional_genes2 = F_TRAJs, F_TRBJs
        else:
            print(f"    Warning: Unknown gene pair combination {feature1}-{feature2}")
            continue
        
        print(f"    Using {len(functional_genes1)} {feature1} genes and {len(functional_genes2)} {feature2} genes")
        print(f"    Matrix dimensions: {len(functional_genes1)} x {len(functional_genes2)}")
        
        npmi_all[(feature1, feature2)] = get_corrected_npmi_for_different_samples_accelerated(
            df_dict, feature1, feature2, functional_genes1, functional_genes2,
            sample_times=args.sample_times, sample_weights=args.sample_weights,
            inner_jobs=args.inner_jobs, outer_jobs=args.outer_jobs, base=args.base
        )
        
        feature_end_time = time.time()
        feature_duration = feature_end_time - feature_start_time
        print(f"  ✓ Completed {feature1} - {feature2} in {format_time(feature_duration)}")
    
    npmi_end_time = time.time()
    npmi_total_time = npmi_end_time - npmi_start_time
    print(f"✓ All NPMI calculations completed in {format_time(npmi_total_time)}")
    print("=== NPMI CALCULATIONS COMPLETED ===")
    
    return npmi_all

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

def process_single_group(df_dict, args):
    """Process a single group of data"""
    print("Processing single group...")
    
    npmi_start = time.time()
    npmi_all = calculate_npmi_statistics(df_dict, args)
    npmi_end = time.time()
    print(f"Total NPMI analysis time: {format_time(npmi_end - npmi_start)}")
    
    return npmi_all

def process_multiple_groups(df_dict, args):
    """Process multiple groups of data"""
    print("Processing multiple groups...")
    npmi_all = defaultdict(dict)
    
    for group_idx, (group, sub_dict) in enumerate(df_dict.items(), 1):
        group_start_time = time.time()
        print(f"\n=== PROCESSING GROUP {group_idx}/{len(df_dict)}: {group} ===")
        group_npmi = process_single_group(sub_dict, args)
        
        # Store results by feature pair then by group
        for feature_pair, values in group_npmi.items():
            npmi_all[feature_pair][group] = values
        
        group_end_time = time.time()
        group_duration = group_end_time - group_start_time
        print(f"✓ Group {group} completed in {format_time(group_duration)}")
    
    return npmi_all

def main():
    parser = argparse.ArgumentParser(description='TCR normalized pointwise mutual information analysis')
    parser.add_argument('--sample_times', type=int, default=300,
                       help='Number of bootstrap samples')
    parser.add_argument('--sample_weights', type=str, default='clonotype.freq',
                       help='sample_weights')
    parser.add_argument('--outer_jobs', type=int, default=8, 
                       help='Number of outer permutation tasks to run in parallel')
    parser.add_argument('--inner_jobs', type=int, default=None, 
                       help='Number of cores per permutation task')
    parser.add_argument('--base', type=float, default=np.e,
                       help='Logarithm base for NPMI calculation')
    parser.add_argument('--inputfile', type=str, required=True,
                       help='Path to input pickle file')
    parser.add_argument('--outputdir', type=str, required=True,
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    total_start_time = time.time()
    
    print(f"=== STARTING TCR NPMI ANALYSIS ===")
    print(f"Sample times: {args.sample_times}")
    print(f"Outer jobs: {args.outer_jobs}")
    print(f"Inner jobs: {args.inner_jobs}")
    print(f"Logarithm base: {args.base}")
    print(f"Sample weights: {args.sample_weights}")
    print(f"Loading data from: {args.inputfile}")
    print(f"NPMI feature pairs: {NPMI_FEATURES}")
    
    # Data loading
    print("\nLoading data...")
    data_load_start = time.time()
    df_dict = pd.read_pickle(args.inputfile)
    data_load_end = time.time()
    print(f"✓ Data loading completed in {format_time(data_load_end - data_load_start)}")
    
    # Determine if single group or multiple groups
    if isinstance(next(iter(df_dict.values())), pd.DataFrame):
        # Single group case
        npmi_all = process_single_group(df_dict, args)
    elif isinstance(next(iter(df_dict.values())), dict):
        # Multiple groups case
        npmi_all = process_multiple_groups(df_dict, args)
    else:
        raise ValueError("Unexpected data structure in input file")
    
    # Save results
    save_start_time = time.time()
    if npmi_all:
        # Save main results
        save_data(npmi_all, os.path.join(args.outputdir, 'npmi.pickle'))
        print("✓ NPMI analysis completed and saved!")
    
    save_end_time = time.time()
    print(f"✓ All data saving completed in {format_time(save_end_time - save_start_time)}")
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"\n=== ANALYSIS COMPLETED SUCCESSFULLY ===")
    print(f"Total execution time: {format_time(total_duration)}")

if __name__ == "__main__":
    main()

