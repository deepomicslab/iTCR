import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize
import seaborn as sns
import os
import gc
from scipy import stats
from joblib import Parallel, delayed, cpu_count
from scipy.stats import bootstrap
from typing import Literal
from Bio import pairwise2 as pw2
from itertools import combinations
import argparse
import pickle

class TCRAnalyzer:
    def __init__(self, df=None, focus="VL", n_jobs=-1, chunk_size=1000, group_name=None, cellstatus=None):
        """
        Initialize TCR analyzer
        
        Parameters:
        - df: Loaded DataFrame
        - focus: Chain type to focus on (VL or VH)
        - n_jobs: Number of parallel jobs
        - chunk_size: Size for chunked processing
        """
        self.df = df
        self.focus = focus
        self.n_jobs = n_jobs
        self.chunk_size = chunk_size
        self._identity_cache = {}  # Cache for sequence similarity calculations
        self.group_name = group_name
        self.cellstatus = cellstatus
        self.count = 1
        self.save = False

    def calculate_aa_identity(self, seq1, seq2):
        """Calculate similarity percentage between two amino acid sequences"""
        if len(seq1) != len(seq2):
            return 0.0
        
        matches = sum(a == b for a, b in zip(seq1, seq2))
        return (matches / len(seq1)) * 100.0

    def cached_calculate_aa_identity(self, seq1, seq2):
        """Amino acid sequence similarity calculation with caching"""
        # Create an order-independent key
        key = (seq1, seq2) if seq1 < seq2 else (seq2, seq1)
        
        if key not in self._identity_cache:
            self._identity_cache[key] = self.calculate_aa_identity(seq1, seq2)
        
        return self._identity_cache[key]
    
    def preprocess_data(self):
        """Preprocess data"""
        
        # Remove duplicate sequences
        self.df = self.df.drop_duplicates(subset=['VH', 'JH', 'CDRH3_AA', 'VL', 'JL', 'CDRL3_AA', 'donor'], keep='first')
        self.df['full.seq'] = self.df['VH'].str.cat([self.df['JH'], self.df['CDRH3_AA'], self.df['VL'], self.df['JL'], self.df['CDRL3_AA']])
        
        # Check number of donors
        donors = self.df['donor'].unique()
            
        return self.df
    
    def analyze_by_donor_groups(self):
        """Analyze by donor groups"""
        
        # Group by VH and CDRH3 length (in this case, data is already grouped by these criteria)
        donor_groups = {donor: self.df[self.df['donor'] == donor] for donor in self.df['donor'].unique()}
        
        # Create donor pairs
        donor_pairs = []
        donors = list(donor_groups.keys())
        for i in range(len(donors)):
            for j in range(i+1, len(donors)):
                donor_pairs.append((donors[i], donors[j]))
                
        
        # Process donor pairs in parallel
        if self.n_jobs != 1:
            results = Parallel(n_jobs=self.n_jobs, verbose=1)(
                delayed(self._process_donor_pair)(donor_pairs[i], donor_groups) 
                for i in range(len(donor_pairs))
            )
        else:
            results = [self._process_donor_pair(pair, donor_groups) for pair in donor_pairs]
            
        # Merge results
        all_pairs = []
        for pairs in results:
            all_pairs.extend(pairs)
            
        pairs_df = pd.DataFrame(all_pairs)
        
        
        return pairs_df
    
    def _process_donor_pair(self, donor_pair, donor_groups):
        """Process all possible pairs between a pair of donors"""
        donor1, donor2 = donor_pair
        df1 = donor_groups[donor1]
        df2 = donor_groups[donor2]

        pairs = []
        
        process_id = os.getpid()
        # Use chunking strategy for large groups
        if len(df1) * len(df2) > 100000:  # Adjustable threshold
            # Process data from both donors in chunks
            for chunk1 in np.array_split(df1, max(1, len(df1) // self.chunk_size)):
                for chunk2 in np.array_split(df2, max(1, len(df2) // self.chunk_size)):
                    print(chunk1.shape[0], chunk2.shape[0])
                    pairs.extend(self._generate_pairs(chunk1, chunk2, donor1, donor2))
                    if len(pairs) > 100000:
                        temp_df = pd.DataFrame(pairs)
                        temp_df.to_csv(os.path.join(args.outputdir, "{}".format(self.group_name) + self.cellstatus + str(process_id) +'_' + str(self.count) + ".csv"))
                        print(f"Saving {self.count} phased results to {args.outputdir}...")
                        self.count += 1
                        del temp_df
                        del pairs
                        pairs = []
                        gc.collect()
        else:
            # Process small groups directly
            pairs.extend(self._generate_pairs(df1, df2, donor1, donor2))
            
        return pairs
    
    def _generate_pairs(self, df1, df2, donor1, donor2):
        """Generate all pairs between two dataframes"""
        pairs = []
        
        # For efficiency, convert data to dictionaries first
        data1 = df1.to_dict('records')
        data2 = df2.to_dict('records')
        
        if donor1 is None or donor2 is None:
            print("ERROR: donor1 or donor2 is None!")
            return pairs
        
        for row1 in data1:
            for row2 in data2:
                clonotype_same = row1['full.seq'] == row2['full.seq']
                # Calculate CDRH3 similarity
                identity = self.cached_calculate_aa_identity(row1['CDRH3_AA'], row2['CDRH3_AA'])
                
                # Check if focus fields are the same
                focus_same = row1[self.focus] == row2[self.focus]
               
                pairs.append({
                    'donor1': str(donor1),
                    'donor2': str(donor2),
                    'celltype_1':row1['celltype'],
                    'celltype_2':row2['celltype'],
                    'CDRH3_AA_1': row1['CDRH3_AA'],
                    'CDRH3_AA_2': row2['CDRH3_AA'],
                    'CDRH3_identity': identity,
                    f'{self.focus}_1': row1[self.focus],
                    f'{self.focus}_2': row2[self.focus],
                    f'{self.focus}_same': focus_same,
                    'clonotype': clonotype_same,
                })
                    
        return pairs

def load_data(data_path):
    """
    Load data from a pickle file
    
    Args:
        data_path (str): Path to the pickle file containing df_dict_naive
        
    Returns:
        dict: Dictionary of DataFrames
    """
    print(f"Loading data from {data_path}...")
    data = pd.read_pickle(data_path)
    return data
    
def generate_different_groups(df_dict, col1='VH', col2='CDRH3.len'):
    """
    Generate pairs of TCRs from different donors with the same VH gene and CDRH3 length.
    Calculate amino acid identity for CDRH3 and check if VL genes are shared.
    
    Args:
        df_dict (dict): Dictionary of DataFrames containing TCR data, where keys are sample IDs
        col1 (str): First column to group by (default: 'VH')
        col2 (str): Second column to group by (default: 'CDRH3.len')
        n_jobs (int): Number of jobs for parallel processing (default: -1, uses all available cores)
        verbose (int): Verbosity level (default: 10, shows progress)
        
    Returns:
        pd.DataFrame: DataFrame containing paired TCR information from different donors
    """
    # Concatenate all DataFrames
    print("Concatenating dataframes...")
    dfs_combined = pd.concat([tcrrep for sampleid, tcrrep in df_dict.items()])
    
    # Pre-filter groups without multiple donors
    print("Pre-filtering data...")
    donor_counts = dfs_combined.groupby([col1, col2])['donor'].nunique()
    valid_groups = donor_counts[donor_counts > 1].index
    
    if len(valid_groups) == 0:
        print("No valid groups found with multiple donors.")
        return pd.DataFrame()
    
    # Keep only valid groups
    dfs_filtered = dfs_combined[dfs_combined.set_index([col1, col2]).index.isin(valid_groups)]
    
    # Group by specified columns
    print("Grouping data...")
    grouped = dfs_filtered.groupby([col1, col2])
    
    return grouped


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate TCR pairs from different donors')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the pickle file containing df_dict_naive')
    parser.add_argument('--col1', type=str, default='VH', help='First column to group by (default: VH)')
    parser.add_argument('--col2', type=str, default='CDRH3.len', help='Second column to group by (default: CDRH3.len)')
    parser.add_argument('--focus', type=str, default='VL', help='The gene focus to compare (default: VL)')
    parser.add_argument('--n_jobs', type=int, default=-1, help='Number of jobs for parallel processing (default: -1)')
    parser.add_argument('--outputdir', type=str, default='tcr_pairs.csv', help='Output file path (default: tcr_pairs.csv)')
    
    args = parser.parse_args()
    if not os.path.exists(args.outputdir):
        os.mkdir(args.outputdir)
    # Load data
    df_dict = load_data(args.data_path)
    #df_dict = {key:value for key,value in df_dict.items() if key in ['D1', 'D2', 'E1','E2','F1','F2','X','Y','Z']}
    cellstatus = os.path.basename(args.outputdir).split('_')[0]
    print("We have {} cpu cores".format(cpu_count()))
    
    group_dfs = generate_different_groups(df_dict, col1='VH', col2='CDRH3.len')
    group_dfs = {'_'.join([group_name[0], str(group_name[1])]): group_df for group_name, group_df in group_dfs}
    print("We have {} groups to address...".format(len(group_dfs)))
    for group_name, group_df in group_dfs.items():
        # initialize
        if os.path.exists(os.path.join(args.outputdir, "{}".format(group_name) + cellstatus + ".csv")):
            continue
        print(group_name, len(group_df))
        analyzer = TCRAnalyzer(df=group_df, focus="VL", n_jobs=args.n_jobs, chunk_size=300, group_name=group_name, cellstatus=cellstatus)
    
        # preprocessing...
        analyzer.preprocess_data()
    
        # analysis
        pairs_df = analyzer.analyze_by_donor_groups()
        print(f"Saving {group_name} results to {args.outputdir}...")
        pairs_df.to_csv(os.path.join(args.outputdir, "{}".format(group_name) + cellstatus + ".csv"))
        del pairs_df
        gc.collect()
