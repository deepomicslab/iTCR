import numpy as np
import pandas as pd
import math
from sklearn.metrics import mutual_info_score
import os
from joblib import Parallel, delayed
from typing import Literal
from scipy.stats import permutation_test
import time
from scipy.stats import norm
from scipy.stats import bootstrap
from sklearn.preprocessing import LabelEncoder
import tidytcells as tt
from collections import Counter
import math
import ndd
from functools import partial
from scipy.stats import bootstrap
import cupy as cp
from collections import defaultdict

def return_normalized_pointwise_mutual_information(df, col1, col2, base=np.e, use_gpu=False):
    """
    Calculate Normalized Pointwise Mutual Information (NPMI)
    NPMI(x,y) = PMI(x,y) / (-log(P(x,y)))
    Range: [-1, 1], where 1 indicates perfect association, 0 indicates independence, 
    and -1 indicates perfect negative association
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    col1, col2 : str
        Column names to calculate NPMI for
    base : float
        Logarithm base (2, np.e, or 10)
    use_gpu : bool
        Whether to use CuPy for GPU acceleration
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns [col1, col2, 'normalized_pointwise']
    """
    # Choose CuPy or NumPy
    xp = cp if use_gpu else np
    
    # Total sample count
    n = len(df)
    
    # Use pandas value_counts for faster counting
    p_x = df[col1].value_counts(normalize=True).to_dict()
    p_y = df[col2].value_counts(normalize=True).to_dict()
    
    # Create joint distribution using groupby (faster)
    joint_counts = df.groupby([col1, col2]).size() / n
    
    # Choose appropriate log function based on base
    if base == 2:
        log_func = xp.log2  # Base-2 logarithm (bits)
    elif base == np.e:
        log_func = xp.log   # Natural logarithm (nats)
    elif base == 10:
        log_func = xp.log10 # Base-10 logarithm
    
    # Convert joint probabilities to arrays for batch computation
    joint_keys = list(joint_counts.keys())
    joint_probs = xp.array(joint_counts.values)
    
    # Get corresponding marginal probabilities
    px_values = xp.array([p_x[xi] for xi, _ in joint_keys])
    py_values = xp.array([p_y[yi] for _, yi in joint_keys])
    
    # Calculate PMI values
    pmi_values = log_func(joint_probs / (px_values * py_values))
    
    # Calculate NPMI: PMI / (-log(P(x,y)))
    # Handle cases where joint_probs might be 0 (though shouldn't happen with our data)
    npmi_values = xp.where(joint_probs > 0, 
                          pmi_values / (-log_func(joint_probs)), 
                          0)
    
    # Create dictionary containing NPMI values
    npmi_dict = {key: value for key, value in zip(joint_keys, npmi_values)}
    
    # Add normalized_pointwise column to original DataFrame
    df_result = df.copy()
    df_result['normalized_pointwise'] = df.apply(
        lambda row: npmi_dict.get((row[col1], row[col2]), 0), axis=1
    )
    
    return df_result[[col1, col2, 'normalized_pointwise']]

def compute_entropy_amino_level(df, col, base=np.e, use_gpu=False, seq_length=19):
    """
    Compute entropy with amino acid level awareness:
    - For sequences (CDR): H(S) = ∑(p=1 to L) H(Sp) 
    - For non-sequences: Standard entropy H(X)
    """
    xp = cp if use_gpu else np
    
    # Detect if column contains sequences
    col_is_seq = True if ("CDR" in col) or ("cdr" in col) else False
    
    # Choose log function
    log_func = xp.log2 if base == 2 else (xp.log10 if base == 10 else xp.log)
    
    if col_is_seq:
        # Position-wise entropy calculation for sequences
        return compute_sequence_entropy(df, col, log_func, xp, seq_length=seq_length)
    else:
        # Standard entropy calculation
        return calculate_entropy(df, col, base=base, use_gpu=use_gpu)


def compute_sequence_entropy(df, seq_col, log_func, xp, seq_length):
    """
    Compute position-wise entropy for sequences: H(S) = ∑(p=1 to L) H(Sp)
    """
    # Get sequence length
    n = len(df)
    
    total_entropy = 0.0
    # Sum entropy over all positions
    for pos in range(seq_length):
        # Extract residue at position pos
        residues_at_pos = df[seq_col].str[pos]
        
        # Calculate entropy for this position
        residue_counts = residues_at_pos.value_counts(sort=False)
        residue_probs = xp.array(residue_counts.values, dtype=xp.float32) / n
        
        # H(Sp) = -∑ p(r) * log(p(r))
        pos_entropy = -xp.sum(residue_probs * log_func(residue_probs))
        total_entropy += pos_entropy
    
    return float(total_entropy.get() if hasattr(total_entropy, 'get') else total_entropy)


def compute_conditional_entropy_amino_level(df, col1, col2, base=np.e, use_gpu=False, seq_length=19):
    """
    Compute conditional entropy with amino acid level awareness:
    - H(X|S) where S is sequence: ∑(p=1 to L) H(X|Sp)
    - H(S1|S2) where both are sequences: ∑(p=1 to L) H(S1p|S2p)  
    - Standard H(X|Y) for non-sequences
    """
    xp = cp if use_gpu else np
    
    # Detect sequence columns
    col1_is_seq = True if ("CDR" in col1) or ("cdr" in col1) else False
    col2_is_seq = True if ("CDR" in col2) or ("cdr" in col2) else False
    
    # Choose log function
    log_func = xp.log2 if base == 2 else (xp.log10 if base == 10 else xp.log)
    
    if col1_is_seq or col2_is_seq:
        # Position-wise conditional entropy for sequences
        return compute_sequence_conditional_entropy(df, col1, col2, col1_is_seq, col2_is_seq, log_func, xp, seq_length=seq_length)
    else:
        # Standard conditional entropy
        return calculate_conditional_entropy(df, col1, col2, base=base, use_gpu=use_gpu)


def compute_sequence_conditional_entropy(df, col1, col2, col1_is_seq, col2_is_seq, log_func, xp, seq_length):
    """
    Compute position-wise conditional entropy for sequences
    """
    n = len(df)
    
    if col1_is_seq and col2_is_seq:
        # Pre-compute marginal frequencies for S: f_p(r)
        seq2_marginals = {}
        for p in range(seq_length):
            seq2_pos_counts = df[col2].str[p].value_counts(sort=False)
            seq2_marginals[p] = (seq2_pos_counts / n).to_dict()
        
        total_cond_entropy = 0.0
        
        # ∑∑ (q=1 to L, p=1 to L) - ALL position pairs
        for q in range(seq_length):  # Positions in T (first sequence)
            for p in range(seq_length):  # Positions in S (second sequence)
                
                # Extract residues at positions q and p
                df_pair = df.copy()
                df_pair['seq1_pos_q'] = df_pair[col1].str[q]  # w at position q
                df_pair['seq2_pos_p'] = df_pair[col2].str[p]  # r at position p
                
                # Calculate joint frequencies f_{q,p}(w,r)
                joint_counts = df_pair.groupby(['seq1_pos_q', 'seq2_pos_p'], sort=False).size()
                
                # ∑∑ (w∈R, r∈R) - All residue pairs
                for (w, r), count in joint_counts.items():
                    # f_{q,p}(w,r): joint frequency
                    f_qp_wr = count / n
                    # f_p(r): marginal frequency of r at position p
                    f_p_r = seq2_marginals[p].get(r, 0)
                    
                    if f_qp_wr > 0 and f_p_r > 0:
                        # H(T|S) contribution: -f_{q,p}(w,r) * log(f_{q,p}(w,r) / f_p(r))
                        conditional_prob = f_qp_wr / f_p_r
                        contribution = f_qp_wr * log_func(conditional_prob)
                        total_cond_entropy -= contribution
        
        return float(total_cond_entropy.get() if hasattr(total_cond_entropy, 'get') else total_cond_entropy)
    
    elif col2_is_seq:
        # H(X|S) = ∑(p=1 to L) H(X|Sp)
        total_cond_entropy = 0.0
        
        for pos in range(seq_length):
            df_pos = df.copy()
            df_pos['seq_pos'] = df_pos[col2].str[pos]
            
            # Calculate H(X|Sp)
            joint_counts = df_pos.groupby([col1, 'seq_pos'], sort=False).size()
            seq_counts = df_pos['seq_pos'].value_counts(sort=False)
            
            joint_vals = joint_counts.values
            seq_marginals = joint_counts.index.get_level_values(1).map(seq_counts).values
            
            joint_probs = xp.array(joint_vals, dtype=xp.float32) / n
            seq_marg = xp.array(seq_marginals, dtype=xp.float32) / n
            
            conditional_probs = joint_probs / seq_marg
            pos_cond_entropy = -xp.sum(joint_probs * log_func(conditional_probs))
            total_cond_entropy += pos_cond_entropy
            
        return float(total_cond_entropy.get() if hasattr(total_cond_entropy, 'get') else total_cond_entropy)
    
    else:
        # col1 is sequence, col2 is not: H(S|X)
        total_cond_entropy = 0.0
        
        for pos in range(seq_length):
            df_pos = df.copy()
            df_pos['seq_pos'] = df_pos[col1].str[pos]
            
            # Calculate H(Sp|X)
            joint_counts = df_pos.groupby(['seq_pos', col2], sort=False).size()
            col2_counts = df_pos[col2].value_counts(sort=False)
            
            joint_vals = joint_counts.values
            col2_marginals = joint_counts.index.get_level_values(1).map(col2_counts).values
            
            joint_probs = xp.array(joint_vals, dtype=xp.float32) / n
            col2_marg = xp.array(col2_marginals, dtype=xp.float32) / n
            
            conditional_probs = joint_probs / col2_marg
            pos_cond_entropy = -xp.sum(joint_probs * log_func(conditional_probs))
            total_cond_entropy += pos_cond_entropy
            
        return float(total_cond_entropy.get() if hasattr(total_cond_entropy, 'get') else total_cond_entropy)

def calculate_entropy(df, col, base=np.e, use_gpu=False):
    xp = cp if use_gpu else np
    # Handle list inputs by combining columns
    if isinstance(col, list):
        col_data = df[col].astype(str).agg('_'.join, axis=1)
    else:
        col_data = df[col]
    
    # Get value counts (same pattern as other functions)
    counts = col_data.value_counts(sort=False)
    n = len(df)
    
    # Convert to target array type
    probs = xp.array(counts.values, dtype=xp.float32) / n
    
    # Choose log function (same pattern)
    log_func = xp.log2 if base == 2 else (xp.log10 if base == 10 else xp.log)
    
    # Vectorized calculation
    entropy = -xp.sum(probs * log_func(probs))
    
    return float(entropy.get() if use_gpu else entropy)


def calculate_conditional_entropy(df, col1, col2, base=np.e, use_gpu=False):
    n = len(df)
    
    xp = cp if use_gpu else np
    
    # Choose appropriate logarithm function based on base
    if base == 2:
        log_func = xp.log2  # Use base-2 logarithm (bits)
    elif base == np.e:
        log_func = xp.log   # Use natural logarithm (nats)
    elif base == 10:
        log_func = xp.log10 # Use base-10 logarithm
    
    # Handle cases where col1 and col2 might be lists
    if isinstance(col1, list):
        df['col1_combined'] = df[col1].astype(str).agg('_'.join, axis=1)
        col1_processed = 'col1_combined'
    else:
        col1_processed = col1
    
    if isinstance(col2, list):
        df['col2_combined'] = df[col2].astype(str).agg('_'.join, axis=1)
        col2_processed = 'col2_combined'
    else:
        col2_processed = col2
    
    # Calculate joint probability p(col1, col2)
    joint_counts = df.groupby([col1_processed, col2_processed], sort=False).size()
    n = len(df)
    # Get marginal counts for col2 only (we need P(col2) for conditional entropy)
    col2_counts = df[col2_processed].value_counts(sort=False)
    
    # Convert to numpy arrays
    joint_vals = joint_counts.values

    # Calculate conditional entropy H(col1|col2) = -∑ p(col1,col2) * log(p(col1,col2)/p(col2))
    conditional_entropy = 0
    
    # Vectorized marginal lookup using pandas map (same pattern as MI)
    col2_marginals = joint_counts.index.get_level_values(1).map(col2_counts).values
    
    # Convert to target array type
    joint_probs = xp.array(joint_vals, dtype=xp.float32) / n
    col2_marg = xp.array(col2_marginals, dtype=xp.float32) / n
    
    # Choose log function (same as MI function)
    log_func = xp.log2 if base == 2 else (xp.log10 if base == 10 else xp.log)
    
    # Vectorized calculation: H(col1|col2) = -∑ p(col1,col2) * log(p(col1,col2)/p(col2))
    conditional_probs = joint_probs / col2_marg
    conditional_entropy = -xp.sum(joint_probs * log_func(conditional_probs))
    
    # Clean up temporary columns
    if isinstance(col1, list):
        df.drop('col1_combined', axis=1, inplace=True)
    if isinstance(col2, list):
        df.drop('col2_combined', axis=1, inplace=True)
    
    return conditional_entropy


def compute_amino_level_mutual_information(df, col1, col2, base=np.e, use_gpu=False, normalization='arithmetic', seq_length=19, positions=None):
    
    xp = cp if use_gpu else np
    
    col1_is_seq = True if ("CDR" in col1)|("cdr" in col1) else False
    col2_is_seq = True if ("CDR" in col1)|("cdr" in col2) else False
    
    # Choose log function
    log_func = xp.log2 if base == 2 else (xp.log10 if base == 10 else xp.log)
    
    if col1_is_seq or col2_is_seq:
        # Use position-wise MI calculation for sequences
        return compute_amino_level_mi(df, col1, col2, col1_is_seq, col2_is_seq, 
                                 log_func, xp, seq_length=seq_length, positions=positions)
    else:
        # Standard MI calculation for non-sequence data
        return compute_normalized_mutual_information(df, col1, col2, base=log_func, normalization=normalization, use_gpu=use_gpu)
    
def compute_amino_level_mi(df, col1, col2, col1_is_seq, col2_is_seq, log_func, xp, seq_length=19, positions=None):
    """
    Compute position-wise MI according to the formula:
    I(X; S) = ∑∑∑ fx,p(x,r) log (fx,p(x,r) / (fx(x)fp(r)))
    """
    
    if col1_is_seq and col2_is_seq:
        # Both are sequences - need position-wise calculation for both
        return compute_sequence_to_sequence_mi(df, col1, col2, log_func, seq_length=seq_length, positions=positions)
    elif col1_is_seq:
        # col1 is sequence, col2 is categorical
        return compute_categorical_to_sequence_mi(df, col2, col1, log_func, seq_length=seq_length, positions=positions)
    else:
        # col2 is sequence, col1 is categorical  
        return compute_categorical_to_sequence_mi(df, col1, col2, log_func, seq_length=seq_length, positions=positions)
    
def compute_categorical_to_sequence_mi(df, cat_col, seq_col, log_func, seq_length=19, positions=None):
    """
    Compute I(X; S) where X is categorical and S is sequence
    """
    # Get sequence length
    
    n = len(df)
    
    # Pre-compute categorical probabilities once
    cat_counts = df[cat_col].value_counts(sort=False)
    cat_probs = (cat_counts / n).to_dict()
    
    # Convert to lists for faster access
    cat_list = df[cat_col].tolist()
    seq_list = df[seq_col].tolist()
    
    total_mi = 0.0
    position_entropies = []
    
    # Determine positions to process
    if positions is not None:
        positions_to_process = positions
    else:
        positions_to_process = range(seq_length)
    
    # Process each position
    for pos in positions_to_process:
        # Use defaultdict for automatic counting
        joint_counts = defaultdict(int)
        residue_counts = defaultdict(int)
        
        # Single pass counting for this position
        valid_count = 0
        for i in range(n):
            seq = seq_list[i]
            
            # Skip if position is out of bounds
            if pos >= len(seq):
                continue
                
            cat_val = cat_list[i]
            residue = seq[pos]
            
            joint_counts[(cat_val, residue)] += 1
            residue_counts[residue] += 1
            valid_count += 1
        
        # Skip if no valid residues at this position
        if valid_count == 0:
            position_entropies.append(0.0)
            continue
        
        # Convert residue counts to probabilities
        residue_probs = {res: count / valid_count for res, count in residue_counts.items()}
        
        # Calculate position entropy
        pos_entropy = 0.0
        for prob in residue_probs.values():
            if prob > 0:
                pos_entropy -= prob * log_func(prob)
        position_entropies.append(float(pos_entropy))
        
        # Calculate MI for this position
        for (cat_val, residue), count in joint_counts.items():
            joint_prob = count / valid_count
            marginal_cat = cat_probs.get(cat_val, 0)
            marginal_res = residue_probs.get(residue, 0)
            
            if joint_prob > 0 and marginal_cat > 0 and marginal_res > 0:
                total_mi += joint_prob * log_func(joint_prob / (marginal_cat * marginal_res))
    
    return float(total_mi)

def compute_sequence_to_sequence_mi(df, col1, col2, log_func, seq_length=19, positions=None):
    """
    Compute MI between two sequences using all position pairs
    I(T;S) = ∑∑∑∑ f_{q,p}(w,r) log(f_{q,p}(w,r) / (f_q(w)f_p(r)))
    """
    n = len(df)
    seq1_list = df[col1].tolist()
    seq2_list = df[col2].tolist()
    
    total_mi = 0.0
    
    if positions is None:
        positions = [(q, p) for q in range(seq_length) for p in range(seq_length)]
        
    for q, p in positions:
        if q >= seq_length or p >= seq_length:
            continue
        # Use defaultdict for automatic counting
        joint_counts = defaultdict(int)
        marginal_q_counts = defaultdict(int)
        marginal_p_counts = defaultdict(int)
        
        # Single pass counting
        for i in range(n):
            res_q = seq1_list[i][q]
            res_p = seq2_list[i][p]
            
            joint_counts[(res_q, res_p)] += 1
            marginal_q_counts[res_q] += 1
            marginal_p_counts[res_p] += 1
        
        # Convert to probabilities and compute MI
        for (res_q, res_p), count in joint_counts.items():
            joint_prob = count / n
            marginal_q_prob = marginal_q_counts[res_q] / n
            marginal_p_prob = marginal_p_counts[res_p] / n
            
            if joint_prob > 0 and marginal_q_prob > 0 and marginal_p_prob > 0:
                total_mi += joint_prob * log_func(joint_prob / (marginal_q_prob * marginal_p_prob))
    
    return float(total_mi)


def compute_mutual_information(df, col1, col2, base=np.e, use_gpu=False):
    xp = cp if use_gpu else np
    
    # Single groupby operation for joint counts
    joint_counts = df.groupby([col1, col2], sort=False).size()
    n = len(df)
    
    # Get marginal counts
    x_counts = df[col1].value_counts(sort=False)
    y_counts = df[col2].value_counts(sort=False)
    
    # Convert to numpy arrays
    joint_vals = joint_counts.values
    
    # Vectorized marginal lookup using pandas map (faster than list comprehension)
    x_marginals = joint_counts.index.get_level_values(0).map(x_counts).values
    y_marginals = joint_counts.index.get_level_values(1).map(y_counts).values
    
    # Convert to target array type
    joint_probs = xp.array(joint_vals, dtype=xp.float32) / n
    x_marg = xp.array(x_marginals, dtype=xp.float32) / n
    y_marg = xp.array(y_marginals, dtype=xp.float32) / n
    
    # Choose log function
    log_func = xp.log2 if base == 2 else (xp.log10 if base == 10 else xp.log)
    
    # Vectorized calculation
    mi_value = xp.sum(joint_probs * log_func(joint_probs / (x_marg * y_marg)))
    
    return float(mi_value.get() if use_gpu else mi_value)

def compute_normalized_mutual_information(df, col1, col2, base=np.e, use_gpu=False, normalization='arithmetic'):

    xp = cp if use_gpu else np
    
    # Single groupby operation for joint counts
    joint_counts = df.groupby([col1, col2], sort=False).size()
    n = len(df)
    
    # Get marginal counts
    x_counts = df[col1].value_counts(sort=False)
    y_counts = df[col2].value_counts(sort=False)
    
    # Convert to numpy arrays
    joint_vals = joint_counts.values
    
    # Vectorized marginal lookup using pandas map (faster than list comprehension)
    x_marginals = joint_counts.index.get_level_values(0).map(x_counts).values
    y_marginals = joint_counts.index.get_level_values(1).map(y_counts).values
    
    # Convert to target array type
    joint_probs = xp.array(joint_vals, dtype=xp.float32) / n
    x_marg = xp.array(x_marginals, dtype=xp.float32) / n
    y_marg = xp.array(y_marginals, dtype=xp.float32) / n
    
    # Choose log function
    log_func = xp.log2 if base == 2 else (xp.log10 if base == 10 else xp.log)
    
    # Calculate Mutual Information
    mi_value = xp.sum(joint_probs * log_func(joint_probs / (x_marg * y_marg)))
    
    # Calculate marginal entropies for normalization
    x_probs = xp.array(x_counts.values, dtype=xp.float32) / n
    y_probs = xp.array(y_counts.values, dtype=xp.float32) / n
    
    # Entropy calculations
    h_x = -xp.sum(x_probs * log_func(x_probs))
    h_y = -xp.sum(y_probs * log_func(y_probs))
    
    # Normalize MI based on chosen method
    if normalization == 'arithmetic':
        nmi_value = 2 * mi_value / (h_x + h_y)
    elif normalization == 'geometric':
        nmi_value = mi_value / xp.sqrt(h_x * h_y)
    elif normalization == 'max':
        nmi_value = mi_value / xp.maximum(h_x, h_y)
    elif normalization == 'min':
        nmi_value = mi_value / xp.minimum(h_x, h_y)
    else:
        raise ValueError("normalization must be one of: 'arithmetic', 'geometric', 'max', 'min'")
    
    # Handle edge cases (when one or both entropies are 0)
    if xp.isnan(nmi_value) or xp.isinf(nmi_value):
        nmi_value = 0.0
    
    return float(nmi_value.get() if use_gpu else nmi_value)


def bootstrap_corrected_mi_parallel(sample_df, col1, col2=None, P=500, random_state=0, base=2, n_jobs=-1, use_gpu=False,
                                    func=compute_mutual_information):
    """Percentile-t Bootstrap mutual information estimation using parallel processing"""
    # Calculate mutual information of the original sample
    I_bar = func(sample_df, col1, col2, base=base, use_gpu=use_gpu)
    
    # Define function for single bootstrap calculation
    def single_bootstrap(i):
        bootstrap_seed = random_state * 10000 + i
        alpha_star = sample_df.sample(n=len(sample_df), replace=True, random_state=bootstrap_seed)
        I_n = func(alpha_star, col1, col2, base=base, use_gpu=use_gpu)
        return I_n
    
    # Execute all bootstrap resampling in parallel
    results = Parallel(n_jobs=n_jobs)(delayed(single_bootstrap)(i) for i in range(P))
    I_boots = np.array([res for res in results])

    mi_corrected = 2*I_bar - np.mean(I_boots)
    
    return mi_corrected


def basic_bootstrap_statistic(sample_df, col1, col2=None, n_resamples=1000, alpha=0.95, random_state=0, base=np.e, 
                             use_gpu=False, statistic_type='nmi', seq_length=19, positions=None):
    # Select function based on statistic type
    if statistic_type == 'entropy':
        func = calculate_entropy if col2 is None else calculate_conditional_entropy
    elif statistic_type == 'mi':
        func = compute_mutual_information
    elif statistic_type == 'nmi':
        func = compute_normalized_mutual_information
    elif statistic_type == 'mi_amino':
        func = compute_amino_level_mutual_information
    elif statistic_type == 'entropy_amino':
        func = compute_entropy_amino_level if col2 is None else compute_conditional_entropy_amino_level
    else:
        raise ValueError("Invalid statistic_type")
    
    if statistic_type in ['mi', 'nmi', 'mi_amino'] and col2 is None:
        raise ValueError("MI requires col2")
    
    # Calculate original statistic
    kwargs = {'base': base, 'use_gpu': use_gpu}
    if 'amino' in statistic_type:
        kwargs['seq_length'] = seq_length
        kwargs['positions'] = positions
    
    if col2 is not None:
        I_bar = func(sample_df, col1, col2, **kwargs)
        
        def statistic_wrapper(*data):
            df = pd.DataFrame({col1: data[0], col2: data[1]})
            return func(df, col1, col2, **kwargs)
        
        bootstrap_result = bootstrap(
                            (sample_df[col1].values,
                            sample_df[col2].values), 
                            statistic_wrapper,
                            n_resamples=n_resamples,
                            confidence_level=alpha,
                            random_state=random_state,
                            method='basic',
                            paired=True)
    else:
        I_bar = func(sample_df, col1, **kwargs)
        
        def statistic_wrapper(*data):
            df = pd.DataFrame({col1: data[0]})
            return func(df, col1, **kwargs)
        
        bootstrap_result = bootstrap(
                            (sample_df[col1].values,),  
                            statistic_wrapper,
                            n_resamples=n_resamples,
                            confidence_level=alpha,
                            random_state=random_state,
                            method='basic',
                            paired=True)
    # Bias-corrected statistic
    I_bar_corrected = 2*I_bar - np.mean(bootstrap_result.bootstrap_distribution)
    
    return I_bar_corrected, bootstrap_result.confidence_interval[0], bootstrap_result.confidence_interval[1]

def basic_bootstrap_mi(sample_df, col1, col2=None, n_resamples=1000, alpha=0.95, random_state=0, base=np.e, 
                       use_gpu=False):
    
    I_bar = compute_mutual_information(sample_df, col1, col2, base=base, use_gpu=use_gpu)
    def mi_statistic_wrapper(*data, feature1='VL', feature2='VH', base=np.e, use_gpu=False):
        # Convert array to DataFrame with original column names
        df = pd.DataFrame({
            feature1: data[0],
            feature2: data[1]
        })
        return compute_mutual_information(df, feature1, feature2, base=base, use_gpu=use_gpu)
    statistic_func = partial(mi_statistic_wrapper, feature1=col1, feature2=col2, use_gpu=use_gpu)
    bootstrap_result = bootstrap(
                        (sample_df[col1].values,
                        sample_df[col2].values),  # Data needs to be passed as a sequence
                        statistic_func,
                        n_resamples=n_resamples,
                        confidence_level = alpha,
                        random_state=random_state,
                        method='basic',
                        paired=True)
    
    I_bar_corrected = 2*I_bar - np.mean(bootstrap_result.bootstrap_distribution)
    
    return I_bar_corrected, bootstrap_result.confidence_interval[0], bootstrap_result.confidence_interval[1]

# https://sci-hub.se/10.1089/106652704773416939
def percentile_t_bootstrap_parallel(sample_df, col1, col2=None, P=500, alpha=0.05, nested_resamples=50, 
                                   use_gpu=False, random_state=0, base=np.e, n_jobs=-1, statistic_type='entropy'):
    # Select appropriate function based on statistic type
    if statistic_type == 'entropy':
        if col2 is None:
            func = calculate_entropy
        else:
            func = calculate_conditional_entropy
    elif statistic_type == 'mi':
        func = compute_mutual_information
    else:
        raise ValueError("statistic_type must be 'entropy' or 'mi'")
    
    # Parameter validation
    if statistic_type == 'mi' and col2 is None:
        raise ValueError("Mutual information calculation requires col2")
    
    # Calculate statistic on original sample
    if col2 is not None:
        I_bar = func(sample_df, col1, col2, base=base, use_gpu=use_gpu)
        
        # Define function for single bootstrap calculation
        def single_bootstrap(i):
            bootstrap_seed = random_state * 10000 + i
            alpha_star = sample_df.sample(n=len(sample_df), replace=True, random_state=bootstrap_seed)
            I_n = func(alpha_star, col1, col2, base=base, use_gpu=use_gpu)
            
            # Nested Bootstrap
            nested_I = np.zeros(nested_resamples)
            for j in range(nested_resamples):
                nested_seed = bootstrap_seed * 100 + j
                nested_sample = alpha_star.sample(n=len(alpha_star), replace=True, random_state=nested_seed)
                nested_I[j] = func(nested_sample, col1, col2, base=base, use_gpu=use_gpu)
            
            sigma_n = np.std(nested_I, ddof=1)
            T_stat = (I_n - I_bar) / sigma_n if sigma_n > 0 else 0
            
            return I_n, T_stat
    else:
        I_bar = func(sample_df, col1, base=base, use_gpu=use_gpu)
        
        # Define function for single bootstrap calculation
        def single_bootstrap(i):
            bootstrap_seed = random_state * 10000 + i
            alpha_star = sample_df.sample(n=len(sample_df), replace=True, random_state=bootstrap_seed)
            I_n = func(alpha_star, col1, base=base, use_gpu=use_gpu)
            
            # Nested Bootstrap
            nested_I = np.zeros(nested_resamples)
            for j in range(nested_resamples):
                nested_seed = bootstrap_seed * 100 + j
                nested_sample = alpha_star.sample(n=len(alpha_star), replace=True, random_state=nested_seed)
                nested_I[j] = func(nested_sample, col1, base=base, use_gpu=use_gpu)
            
            sigma_n = np.std(nested_I, ddof=1)
            T_stat = (I_n - I_bar) / sigma_n if sigma_n > 0 else 0
            
            return I_n, T_stat
    
    results = Parallel(n_jobs=n_jobs)(delayed(single_bootstrap)(i) for i in range(P))
    I_boots = np.array([res[0] for res in results])
    T_stats = np.array([res[1] for res in results])
    
    T_sorted = np.sort(T_stats)
    sigma = np.std(I_boots, ddof=1)
    
    T_p = np.percentile(T_sorted, alpha/2 * 100)
    T_q = np.percentile(T_sorted, (1-alpha/2) * 100)
    
    ci_low = I_bar - sigma * T_q
    ci_high = I_bar - sigma * T_p
    
    mi_corrected = 2*I_bar - np.mean(I_boots)
    
    return mi_corrected, ci_low, ci_high

