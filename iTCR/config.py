"""
Configuration file for TCR feature analysis
User can modify these feature sets according to their needs
"""

# Define feature sets for TCR analysis
SINGLE_FEATURES = ['cdr3A', 'cdr3B', 'TRAV', 'TRBV', 'TRAJ', 'TRBJ']

CONDITIONAL_FEATURES = [
    ('cdr3A', 'cdr3B'), ('cdr3B', 'cdr3A'), 
    ('TRAV', 'TRBV'), ('TRBV', 'TRAV'), 
    ('TRAJ', 'TRBJ'), ('TRBJ', 'TRAJ')
]

CROSS_FEATURES = [
    ('TRAV', 'TRBV'), ('TRAV', 'cdr3B'),
     ('TRAJ', 'TRBJ'), ('TRAJ', 'cdr3B'),
    ('cdr3A', 'TRBV'), ('cdr3A', 'cdr3B'), 
    ('cdr3A', 'TRBJ')]