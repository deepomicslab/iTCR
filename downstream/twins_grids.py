import pandas as pd
import numpy as np
import pyrepseq as prs
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize
import seaborn as sns
import os
from scipy import stats
import concurrent.futures
from joblib import Parallel, delayed
import multiprocessing
import argparse
import pickle

def analysis_significant(twin_result, nontwin_result, features, method='auto', n_permutations=10000, plot=False, alternative='two-sided'):
    """
    Analyze the statistical significance between twin and non-twin results.
    
    Parameters:
    -----------
    twin_result : dict
        Dictionary containing results for twins
    nontwin_result : dict
        Dictionary containing results for non-twins
    features : list or tuple
        Features to analyze. If tuple, contains separate features for rows and columns
    method : str, optional
        Statistical method to use: 'auto' (automatically choose t-test or Mann-Whitney U)
        or 'permutation' (use permutation test)
    n_permutations : int, optional
        Number of permutations for permutation test (only used if method='permutation')
    """
    if isinstance(features, tuple):
        features1 = features[0]
        features2 = features[1]
        upper_triangle_only = False
    else:
        features1 = features
        features2 = features
        upper_triangle_only = True
    pvalues = {}
    
    if plot:
        figsize = (2, 2)
        if upper_triangle_only:
            n_rows = int(len(features1) * (len(features1) + 1) / 2)
            if n_rows %2 !=0: 
                n_cols = 1
                fig, ax = plt.subplots(int(n_rows), n_cols, figsize=(10,10))
            else:
                n_cols = 2
                fig, ax = plt.subplots(int(n_rows/2), n_cols, figsize=(10,10))
        else:
            n_rows = n_cols = len(features1)
            fig, ax = plt.subplots(len(features1), len(features1), figsize=(10,10))
        axes = ax.flatten()
        plot_pointer = 0
    
    for i, feature1 in enumerate(features1):
        for j, feature2 in enumerate(features2):
            if upper_triangle_only and j < i : continue
            
            twins_data = [grid[i,j] for pairs, grid in twin_result.items() if not np.isinf(grid[i,j])]
            nontwins_data = [grid[i,j] for pairs, grid in nontwin_result.items() if not np.isinf(grid[i,j])]
            
            # # Display basic statistics
            # print(f"\nResults for {feature1} and {feature2}:")
            # print(f"Twin mean = {np.mean(twins_data):.4f}, Non-twin mean = {np.mean(nontwins_data):.4f}")
            # print(f"Mean difference = {np.mean(twins_data) - np.mean(nontwins_data):.4f}")
            
            if method == 'auto':
                # Original analysis_significiant logic
                # Check if data follows normal distribution
                stat_twins, p_twins = stats.shapiro(twins_data)
                stat_nontwins, p_nontwins = stats.shapiro(nontwins_data)
                
                # Choose appropriate test based on normality test results
                alpha = 0.05
                if p_twins > alpha and p_nontwins > alpha:
                    # If both groups follow normal distribution, use t-test
                    stat, p_value = stats.ttest_ind(twins_data, nontwins_data)
                    test_name = "Independent t-test"
                else:
                    # If at least one group doesn't follow normal distribution, use Mann-Whitney U test
                    stat, p_value = stats.mannwhitneyu(twins_data, nontwins_data)
                    test_name = "Mann-Whitney U test"

                print(f"\n{test_name} results:")
                print(f"Statistic = {stat:.4f}, p-value = {p_value:.4f}")
                
            elif method == 'permutation':
                # Permutation test logic
                # Calculate original difference (using mean difference)
                original_diff = np.mean(twins_data) - np.mean(nontwins_data)
                
                # Combine data for permutation
                combined = twins_data + nontwins_data
                n_twins = len(twins_data)
                
                # Calculate p-value for permutation test
                count_more_extreme = 0
                
                for _ in range(n_permutations):
                    # Randomly shuffle data
                    np.random.shuffle(combined)
                    # Split into two groups
                    perm_twins = combined[:n_twins]
                    perm_nontwins = combined[n_twins:]
                    # Calculate difference after permutation
                    perm_diff = np.mean(perm_twins) - np.mean(perm_nontwins)
                    
                    # Count cases more extreme than original difference (two-tailed test)
                    if alternative == "two-sided":
                        if abs(perm_diff) >= abs(original_diff):
                            count_more_extreme += 1
                    elif alternative == 'greater':
                    # (twins > nontwins) 
                        if perm_diff >= original_diff:
                            count_more_extreme += 1
                    elif alternative == 'less':
                    # (twins < nontwins) 
                        if perm_diff <= original_diff:
                            count_more_extreme += 1
                
                # Calculate p-value for permutation test
                p_value = count_more_extreme / n_permutations
                
                # Output results
                print(f"\nPermutation test results:")
                print(f"Original mean difference = {original_diff:.4f}")
                print(f"Permutation test p-value = {p_value:.4f} (based on {n_permutations} permutations)")
            
            else:
                raise ValueError("Method must be either 'auto' or 'permutation'")
            
            pvalues[(feature1, feature2)] = p_value
                
    return pvalues

def confirm_min_shape(individuals, datadir):
    shapes = []
    for ins in individuals:
        in1rep = pd.read_csv(os.path.join(datadir, "{}.txt".format(ins)), sep="\t")
        shapes.append(in1rep.shape[0])    
    return min(shapes)
    
def get_alpha_beta_grids(pairs, datadir, min_shapes=None, random_state=0):
    alpha_beta_grids = {}
    for in1, in2 in pairs:
        in1rep = pd.read_csv(os.path.join(datadir, "{}.txt".format(in1)), sep="\t")
        in1rep['full.beta'] = in1rep['VH'].str.cat([in1rep['CDRH3_AA'], in1rep['JH']])
        in1rep['full.alpha'] = in1rep['VL'].str.cat([in1rep['CDRL3_AA'], in1rep['JL']])
        in1rep['full.tcr'] = in1rep['full.alpha'].str.cat(in1rep['full.beta'])
        in1rep['freqs'] = in1rep['full.tcr'].map(in1rep['full.tcr'].value_counts())
        in2rep = pd.read_csv(os.path.join(datadir, "{}.txt".format(in2)), sep="\t")
        in2rep['full.beta'] = in2rep['VH'].str.cat([in2rep['CDRH3_AA'], in2rep['JH']])
        in2rep['full.alpha'] = in2rep['VL'].str.cat([in2rep['CDRL3_AA'], in2rep['JL']])
        in2rep['full.tcr'] = in2rep['full.alpha'].str.cat(in2rep['full.beta'])
        in2rep['freqs'] = in2rep['full.tcr'].map(in2rep['full.tcr'].value_counts())
        
        if min_shapes is not None:
            in1rep = in1rep.sample(n=min_shapes, weights=in1rep['freqs'].values, random_state=random_state)
            in2rep = in2rep.sample(n=min_shapes, weights=in2rep['freqs'].values, random_state=random_state)
        
        twin = pd.concat([in1rep, in2rep])
        
        alpha_beta_grid = np.zeros((2,2))
        # (alpha, beta)
        alpha_beta_grid[0 ,1] = alpha_beta_grid[1 ,0] = cached_backgrounds["Alpha"]  + cached_backgrounds["Beta"] - prs.renyi2_entropy(twin, "full.tcr")
        # (alpha)
        alpha_beta_grid[0 ,0] = cached_backgrounds["Alpha"] - prs.renyi2_entropy(twin, "full.alpha")
        # (beta)
        alpha_beta_grid[1 ,1] = cached_backgrounds["Beta"] - prs.renyi2_entropy(twin, "full.beta")
        
        alpha_beta_grids[(in1, in2)] = alpha_beta_grid
    
    return alpha_beta_grids

def get_relevancy(in1, in2, datadir, features, min_shapes, cached_backgrounds, random_state):
    relevancy_grid = np.zeros((len(features), len(features)))
    in1rep = pd.read_csv(os.path.join(datadir, "{}.txt".format(in1)), sep="\t")
    in1rep['full.beta'] = in1rep['VH'].str.cat([in1rep['CDRH3_AA'], in1rep['JH']])
    in1rep['full.alpha'] = in1rep['VL'].str.cat([in1rep['CDRL3_AA'], in1rep['JL']])
    in1rep['full.tcr'] = in1rep['full.alpha'].str.cat(in1rep['full.beta'])
    in1rep['freqs'] = in1rep['full.tcr'].map(in1rep['full.tcr'].value_counts())
    in2rep = pd.read_csv(os.path.join(datadir, "{}.txt".format(in2)), sep="\t")
    in2rep['full.beta'] = in2rep['VH'].str.cat([in2rep['CDRH3_AA'], in2rep['JH']])
    in2rep['full.alpha'] = in2rep['VL'].str.cat([in2rep['CDRL3_AA'], in2rep['JL']])
    in2rep['full.tcr'] = in2rep['full.alpha'].str.cat(in2rep['full.beta'])
    in2rep['freqs'] = in2rep['full.tcr'].map(in2rep['full.tcr'].value_counts())
    if min_shapes is not None:
        in1rep = in1rep.sample(n=min_shapes, weights=in1rep['freqs'].values, random_state=random_state)
        in2rep = in2rep.sample(n=min_shapes, weights=in2rep['freqs'].values, random_state=random_state)
    specific_dataframe = pd.concat([in1rep, in2rep])
    specific_dataframe = specific_dataframe.rename(columns={"CDRL3_AA": "CDR3A", "CDRH3_AA": "CDR3B", "VL": "TRAV", "VH": "TRBV",
                                "JL": "TRAJ", "JH": "TRBJ"})
    for i, feature_1 in enumerate(features):
        for j, feature_2 in enumerate(features[i:], start=i): 
            if feature_1 == feature_2:
                relevancy_grid[i, j] = cached_backgrounds[feature_1] - prs.renyi2_entropy(specific_dataframe, feature_1)
            else:
                relevancy_grid[i, j] = cached_backgrounds[(feature_1, feature_2)] - prs.renyi2_entropy(specific_dataframe, [feature_1, feature_2])
            if i != j:
                relevancy_grid[j, i] = relevancy_grid[i, j]  
    return (in1, in2), relevancy_grid

def get_part_grids(pairs, datadir, features, min_shapes=None, n_jobs=-1, random_state=0):
    
    # Determine the number of cores to use
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    
    # Execute all pair calculations in parallel
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(get_relevancy)(
            in1, in2, datadir, features, min_shapes, cached_backgrounds, random_state
        ) for in1, in2 in pairs
    )
    
    # Convert results to dictionary
    part_rlevancy_grids = dict(results)
    
    return part_rlevancy_grids

def get_cross_relevancy(in1, in2, datadir, features_1, features_2, min_shapes, cached_backgrounds, random_state):
    """Process calculations for a single pair as the basic unit for parallel processing"""
    relevancy_grid = np.zeros((len(features_1), len(features_2)))
    
    # Read data
    in1rep = pd.read_csv(os.path.join(datadir, "{}.txt".format(in1)), sep="\t")
    in1rep['full.beta'] = in1rep['VH'].str.cat([in1rep['CDRH3_AA'], in1rep['JH']])
    in1rep['full.alpha'] = in1rep['VL'].str.cat([in1rep['CDRL3_AA'], in1rep['JL']])
    in1rep['full.tcr'] = in1rep['full.alpha'].str.cat(in1rep['full.beta'])
    in1rep['freqs'] = in1rep['full.tcr'].map(in1rep['full.tcr'].value_counts())
    in2rep = pd.read_csv(os.path.join(datadir, "{}.txt".format(in2)), sep="\t")
    in2rep['full.beta'] = in2rep['VH'].str.cat([in2rep['CDRH3_AA'], in2rep['JH']])
    in2rep['full.alpha'] = in2rep['VL'].str.cat([in2rep['CDRL3_AA'], in2rep['JL']])
    in2rep['full.tcr'] = in2rep['full.alpha'].str.cat(in2rep['full.beta'])
    in2rep['freqs'] = in2rep['full.tcr'].map(in2rep['full.tcr'].value_counts())
    
    # Sampling (if needed)
    if min_shapes is not None:
        in1rep = in1rep.sample(n=min_shapes, weights=in1rep['freqs'].values, random_state=random_state)
        in2rep = in2rep.sample(n=min_shapes, weights=in2rep['freqs'].values, random_state=random_state)
    
    # Merge datasets
    specific_dataframe = pd.concat([in1rep, in2rep])
    specific_dataframe = specific_dataframe.rename(columns={"CDRL3_AA": "CDR3A", "CDRH3_AA": "CDR3B", "VL": "TRAV", "VH": "TRBV",
                                "JL": "TRAJ", "JH": "TRBJ"})
    
    # Calculate grid
    for i, feature1 in enumerate(features_1):
        for j, feature2 in enumerate(features_2):
            relevancy_grid[i, j] = cached_backgrounds[feature1] + cached_backgrounds[feature2] - prs.renyi2_entropy(specific_dataframe, [feature1, feature2])
    
    return (in1, in2), relevancy_grid

def get_cross_relevancy_grid_parallel(pairs, datadir, features_1, features_2, min_shapes=None, n_jobs=-1, random_state=0):
    # Determine the number of cores to use
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    
    # Execute all pair calculations in parallel
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(get_cross_relevancy)(
            in1, in2, datadir, features_1, features_2, min_shapes, cached_backgrounds, random_state
        ) for in1, in2 in pairs
    )
    
    # Convert results to dictionary
    alpha_beta_grids = dict(results)
    
    return alpha_beta_grids

def save_pickle(obj, filepath):
    """
    Save an object to a pickle file.
    
    Args:
        obj: The Python object to serialize
        filepath: Path where the pickle file will be saved
        protocol: Pickle protocol version (None uses highest available)
        
    Returns:
        bool: True if successful, False otherwise
        
    Example:
        >>> data = {"key": "value", "numbers": [1, 2, 3]}
        >>> save_pickle(data, "data.pkl")
    """
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: str):
    """
    Load an object from a pickle file.
    
    Args:
        filepath: Path to the pickle file
        default: Value to return if loading fails
        
    Returns:
        The unpickled object or default value if loading fails
        
    Example:
        >>> data = load_pickle("data.pkl")
        >>> if data is not None:
        >>>     print(data)
    """
    
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def process_single_permutation(i, twin_pairs, non_related_pairs, twinsdir, minshape, cached_backgrounds, inner_jobs=4, celltype="memory"):
    """Process a single permutation with all analysis types"""
    
    # Set inner parallelism level
    n_jobs = inner_jobs
    
    # Create a results dictionary for this permutation
    perm_result = {}
    
    # Run all analyses with the same random state
    perm_result[f'twin_alphabeta_grids_{celltype}'] = get_alpha_beta_grids(twin_pairs, twinsdir, min_shapes=minshape, random_state=i)
    perm_result[f'nontwin_alphabeta_grids_{celltype}'] = get_alpha_beta_grids(non_related_pairs, twinsdir, min_shapes=minshape, random_state=i)
    
    perm_result[f'twin_alpha_part_grids_{celltype}'] = get_part_grids(twin_pairs, twinsdir, ["CDR3A", "TRAV", "TRAJ"], 
                                                min_shapes=minshape, n_jobs=n_jobs, random_state=i)
    perm_result[f'nontwin_alpha_part_grids_{celltype}'] = get_part_grids(non_related_pairs, twinsdir, ["CDR3A", "TRAV", "TRAJ"], 
                                                     min_shapes=minshape, n_jobs=n_jobs, random_state=i)
    
    perm_result[f'twin_beta_part_grids_{celltype}'] = get_part_grids(twin_pairs, twinsdir, ["CDR3B", "TRBV", "TRBJ"], 
                                               min_shapes=minshape, n_jobs=n_jobs, random_state=i)
    perm_result[f'nontwin_beta_part_grids_{celltype}'] = get_part_grids(non_related_pairs, twinsdir, ["CDR3B", "TRBV", "TRBJ"], 
                                                    min_shapes=minshape, n_jobs=n_jobs, random_state=i)
    
    perm_result[f'twin_cross_grid_{celltype}'] = get_cross_relevancy_grid_parallel(twin_pairs, twinsdir, 
                                                                            ["CDR3A", "TRAV", "TRAJ"], 
                                                                            ["CDR3B", "TRBV", "TRBJ"], 
                                                                            min_shapes=minshape, 
                                                                            n_jobs=n_jobs, 
                                                                            random_state=i)
    perm_result[f'nontwin_cross_grid_{celltype}'] = get_cross_relevancy_grid_parallel(non_related_pairs, twinsdir, 
                                                                               ["CDR3A", "TRAV", "TRAJ"], 
                                                                               ["CDR3B", "TRBV", "TRBJ"], 
                                                                               min_shapes=minshape, 
                                                                               n_jobs=n_jobs, 
                                                                               random_state=i)
    
    print(f"Completed permutation {i}")
    return i, perm_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_times', type=int, default=100)
    parser.add_argument('--outputdir', type=str, default='/home/yipingzou2/TCRMI')
    parser.add_argument('--outer_jobs', type=int, default=4, help='Number of outer permutation tasks to run in parallel')
    parser.add_argument('--inner_jobs', type=int, default=None, help='Number of cores per permutation task')
    args = parser.parse_args()
    
    total_cores = multiprocessing.cpu_count()
    print(f"Total available CPU cores: {total_cores}")
    
    # Calculate inner_jobs if not specified
    if args.inner_jobs is None:
        # Use at least 2 cores per inner job, but don't exceed what's available divided by outer jobs
        args.inner_jobs = min(2, total_cores // args.outer_jobs)
    
    print(f"Running with {args.outer_jobs} outer parallel jobs, each using {args.inner_jobs} cores")
    
    cached_backgrounds = np.load("/scratch/yipingzou2/TCRMI/twins/cached_backgrouds.npy", allow_pickle=True).item()
    twinsdir = "/scratch/yipingzou2/TCRdataset/pair"
    
    twin1 = [individual + " naive" for individual in ['A1', 'B1', 'C1']]
    twin2 = [individual + " naive" for individual in ['A2', 'B2', 'C2']]
    # 1. Generate twin pairs
    twin_pairs = []
    for i in range(len(twin1)):
        twin_pairs.append((twin1[i], twin2[i]))

    # 2. Generate non-related individual pairs
    non_related_pairs = []

    # Intra-group non-related pairs (within twin1)
    for i in range(len(twin1)):
        for j in range(i+1, len(twin1)):
            non_related_pairs.append((twin1[i], twin1[j]))

    # Intra-group non-related pairs (within twin2)
    for i in range(len(twin2)):
        for j in range(i+1, len(twin2)):
            non_related_pairs.append((twin2[i], twin2[j]))

    # Cross-group non-twin pairs
    for i in range(len(twin1)):
        for j in range(len(twin2)):
            if i != j:  # Exclude twin pairs
                non_related_pairs.append((twin1[i], twin2[j]))
    minshape = confirm_min_shape(twin1 + twin2, twinsdir)
    # Run permutations in parallel
    print(f"Starting {args.sample_times} permutations with {args.outer_jobs} parallel workers")
    results = Parallel(n_jobs=args.outer_jobs, verbose=10)(
        delayed(process_single_permutation)(
            i, twin_pairs, non_related_pairs, twinsdir, minshape, cached_backgrounds, args.inner_jobs, celltype="naive"
        ) for i in range(args.sample_times)
    )
    
    # Organize results
    naive_grids = {}
    for i, perm_result in results:
        naive_grids[i] = perm_result
    save_pickle(naive_grids, os.path.join(args.outputdir, "naive_grids.pickle"))
    print("Successfully save naive_grids.")
    
    twin1 = [individual + " memory" for individual in ['A1', 'B1', 'C1']]
    twin2 = [individual + " memory" for individual in ['A2', 'B2', 'C2']]
    # 1. Generate twin pairs
    twin_pairs = []
    for i in range(len(twin1)):
        twin_pairs.append((twin1[i], twin2[i]))

    # 2. Generate non-related individual pairs
    non_related_pairs = []

    # Intra-group non-related pairs (within twin1)
    for i in range(len(twin1)):
        for j in range(i+1, len(twin1)):
            non_related_pairs.append((twin1[i], twin1[j]))

    # Intra-group non-related pairs (within twin2)
    for i in range(len(twin2)):
        for j in range(i+1, len(twin2)):
            non_related_pairs.append((twin2[i], twin2[j]))

    # Cross-group non-twin pairs
    for i in range(len(twin1)):
        for j in range(len(twin2)):
            if i != j:  # Exclude twin pairs
                non_related_pairs.append((twin1[i], twin2[j]))
    minshape = confirm_min_shape(twin1 + twin2, twinsdir)
    # Run permutations in parallel
    print(f"Starting {args.sample_times} permutations with {args.outer_jobs} parallel workers")
    results = Parallel(n_jobs=args.outer_jobs, verbose=10)(
        delayed(process_single_permutation)(
            i, twin_pairs, non_related_pairs, twinsdir, minshape, cached_backgrounds, args.inner_jobs, celltype="memory"
        ) for i in range(args.sample_times)
    )
    
    # Organize results
    memory_grids = {}
    for i, perm_result in results:
        memory_grids[i] = perm_result
    save_pickle(memory_grids, os.path.join(args.outputdir, "memory_grids.pickle"))
    print("Successfully save memory_grids.")
    
    twin1 = ['F1', 'E1', 'D1'] # 'A1', 'B1', 'C1', 
    twin2 = ['F2', 'E2', 'D2', 'X', 'Y', 'Z'] # 'A2', 'B2', 'C2', 
    # 1. Generate twin pairs
    twin_pairs = []
    for i in range(len(twin1)):
        twin_pairs.append((twin1[i], twin2[i]))

    # 2. Generate non-related individual pairs
    non_related_pairs = []

    # Intra-group non-related pairs (within twin1)
    for i in range(len(twin1)):
        for j in range(i+1, len(twin1)):
            non_related_pairs.append((twin1[i], twin1[j]))

    # Intra-group non-related pairs (within twin2)
    for i in range(len(twin2)):
        for j in range(i+1, len(twin2)):
            non_related_pairs.append((twin2[i], twin2[j]))

    # Cross-group non-twin pairs
    for i in range(len(twin1)):
        for j in range(len(twin2)):
            if i != j:  # Exclude twin pairs
                non_related_pairs.append((twin1[i], twin2[j]))
    
    minshape = confirm_min_shape(twin1 + twin2, twinsdir)
    # Run permutations in parallel
    print(f"Starting {args.sample_times} permutations with {args.outer_jobs} parallel workers")
    results = Parallel(n_jobs=args.outer_jobs, verbose=10)(
        delayed(process_single_permutation)(
            i, twin_pairs, non_related_pairs, twinsdir, minshape, cached_backgrounds, args.inner_jobs, celltype="totalT"
        ) for i in range(args.sample_times)
    )
     # Organize results
    totalT_grids = {}
    for i, perm_result in results:
        totalT_grids[i] = perm_result
    save_pickle(totalT_grids, os.path.join(args.outputdir, "totalT_grids.pickle"))
    print("Successfully save totalT_grids.")
