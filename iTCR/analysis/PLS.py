#!/usr/bin/env python3
"""
Pipeline for Longitudinal Study (PLS) - TCR NPMI Analysis
Step 1: Calculate NPMI matrices using calculate_NPMI.py
Step 2: Analyze timepoint changes using npmi_timepoints.py
"""

import os
import sys
import subprocess
import argparse
import time
import numpy as np

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

def run_step1_npmi_calculation(args):
    """Step 1: Run NPMI calculation"""
    print("=" * 60)
    print("STEP 1: CALCULATING NPMI MATRICES")
    print("=" * 60)
    
    step1_start = time.time()
    
    try:
        # Import and call function directly, similar to MCR approach
        from ..core.calculate_NPMI import main as calculate_npmi_main
        
        # Save original sys.argv
        original_argv = sys.argv.copy()
        
        # Set new sys.argv for calculate_NPMI to use
        sys.argv = [
            'calculate_NPMI.py',
            '--inputfile', args.inputfile,
            '--outputdir', args.outputdir,
            '--sample_times', str(args.sample_times),
            '--sample_weights', args.sample_weights,
            '--outer_jobs', str(args.outer_jobs),
            '--base', str(args.base)
        ]
        
        if args.inner_jobs is not None:
            sys.argv.extend(['--inner_jobs', str(args.inner_jobs)])
        
        print(f"Running NPMI calculation with args: {' '.join(sys.argv[1:])}")
        
        # Call function directly
        calculate_npmi_main()
        
        # Restore original sys.argv
        sys.argv = original_argv
        
        step1_end = time.time()
        print(f"✓ Step 1 completed in {format_time(step1_end - step1_start)}")
        return True
        
    except Exception as e:
        # Restore original sys.argv
        sys.argv = original_argv
        print(f"ERROR in Step 1: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_step2_timepoint_analysis(args):
    """Step 2: Run timepoint analysis"""
    print("=" * 60)
    print("STEP 2: ANALYZING TIMEPOINT CHANGES")
    print("=" * 60)
    
    step2_start = time.time()
    
    # Path to NPMI results from step 1
    npmi_file = os.path.join(args.outputdir, 'npmi.pickle')
    
    if not os.path.exists(npmi_file):
        print(f"ERROR: NPMI results file not found: {npmi_file}")
        return False
    
    try:
        # Import and call function directly
        from .npmi_timepoints import main as npmi_timepoints_main
        
        # Save original sys.argv
        original_argv = sys.argv.copy()
        
        # Set new sys.argv
        sys.argv = [
            'npmi_timepoints.py',
            '--npmi-data', npmi_file,
            '--data-dict', args.inputfile,
            '--output-dir', args.outputdir,
            '--n-permutations', str(args.n_permutations),
            '--n-jobs', str(args.n_jobs)
        ]
        
        print(f"Running timepoint analysis with args: {' '.join(sys.argv[1:])}")
        
        # Call function directly
        npmi_timepoints_main()
        
        # Restore original sys.argv
        sys.argv = original_argv
        
        step2_end = time.time()
        print(f"✓ Step 2 completed in {format_time(step2_end - step2_start)}")
        return True
        
    except Exception as e:
        # Restore original sys.argv
        sys.argv = original_argv
        print(f"ERROR in Step 2: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='Pipeline for Longitudinal Study (PLS) - TCR NPMI Analysis')
    
    # Input/Output
    parser.add_argument('--inputfile', type=str, required=True,
                       help='Path to input pickle file')
    parser.add_argument('--outputdir', type=str, required=True,
                       help='Output directory for results')
    
    # Step 1: NPMI calculation parameters
    parser.add_argument('--sample_times', type=int, default=300,
                       help='Number of bootstrap samples (default: 100)')
    parser.add_argument('--sample_weights', type=str, default='clonotype.freq',
                       help='Column name for sampling weights (default: clonotype.freq)')
    parser.add_argument('--outer_jobs', type=int, default=4, 
                       help='Number of outer permutation tasks to run in parallel (default: 4)')
    parser.add_argument('--inner_jobs', type=int, default=None, 
                       help='Number of cores per permutation task (default: auto)')
    parser.add_argument('--base', type=float, default=np.e,
                       help='Logarithm base for NPMI calculation (default: e)')
    
    # Step 2: Analysis parameters
    parser.add_argument('--n_permutations', type=int, default=10000,
                       help='Number of permutations for statistical testing (default: 10000)')
    parser.add_argument('--n_jobs', type=int, default=-1,
                       help='Number of parallel jobs for analysis (default: -1, use all cores)')
    
    # Pipeline control
    parser.add_argument('--skip_step1', action='store_true',
                       help='Skip Step 1 (NPMI calculation) and proceed to Step 2')
    parser.add_argument('--only_step1', action='store_true',
                       help='Only run Step 1 (NPMI calculation)')
    
    args = parser.parse_args()
    
    total_start_time = time.time()
    
    print("=" * 60)
    print("PIPELINE FOR LONGITUDINAL STUDY (PLS)")
    print("TCR NPMI Analysis")
    print("=" * 60)
    print(f"Input file: {args.inputfile}")
    print(f"Output directory: {args.outputdir}")
    print(f"Sample times: {args.sample_times}")
    print(f"Permutations: {args.n_permutations}")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(args.outputdir, exist_ok=True)
    
    success = True
    
    # Step 1: NPMI Calculation
    if not args.skip_step1:
        success = run_step1_npmi_calculation(args)
        if not success:
            print("Pipeline failed at Step 1")
            return
    else:
        print("Skipping Step 1 (NPMI calculation)")
    
    # Step 2: Timepoint Analysis
    if not args.only_step1 and success:
        success = run_step2_timepoint_analysis(args)
        if not success:
            print("Pipeline failed at Step 2")
            return
    elif args.only_step1:
        print("Stopping after Step 1 as requested")
    
    if success:
        total_end_time = time.time()
        total_duration = total_end_time - total_start_time
        
        print("=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Total execution time: {format_time(total_duration)}")
        print(f"Results saved in: {args.outputdir}")
        print("\nOutput files:")
        print(f"  - NPMI matrices: {args.outputdir}/npmi.pickle")
        if not args.only_step1:
            print(f"  - Detailed results: {args.outputdir}/patient_PLS_detailed.pickle")
            print(f"  - Summary: {args.outputdir}/patient_PLS_summary.csv")
        print("=" * 60)

if __name__ == "__main__":
    main()
