import numpy as np
from scipy.stats import entropy

def calculate_normalized_entropy(relevance):
    """
    Calculates Shannon Entropy and Normalized Entropy of the relevance distribution.
    Use absolute values of relevance to treat as a probability distribution.
    
    Returns:
        tuple: (entropy, normalized_entropy)
    """
    rel = np.array(relevance)
    abs_rel = np.abs(rel)
    n = len(abs_rel)
    
    if n == 0:
        return 0.0, 0.0
    
    total_mass = abs_rel.sum()
    if total_mass > 0:
        probs = abs_rel / total_mass
    else:
        # Uniform if sum is 0
        probs = np.ones_like(abs_rel) / n

    # Shannon Entropy
    ent = entropy(probs)
    
    # Normalized Entropy (0 to 1)
    # max_ent = ln(N)
    max_ent = np.log(n)
    norm_ent = ent / max_ent if max_ent > 0 else 0
    
    return ent, norm_ent

def calculate_gini_coefficient(relevance):
    """
    Calculates the Gini Coefficient of the relevance absolute values.
    Values near 1 indicate very unequal distribution (high concentration/sparsity).
    Values near 0 indicate uniform distribution.
    """
    rel = np.array(relevance)
    abs_rel = np.abs(rel)
    n = len(abs_rel)
    
    if n == 0 or abs_rel.mean() == 0:
        return 0.0
        
    # Sort ascending for Gini calculation formula
    sorted_asc = np.sort(abs_rel)
    index = np.arange(1, n + 1)
    
    # Gini = (2 * sum(i * xi) - (n + 1) * sum(xi)) / (n * sum(xi))
    gini_coeff = (2 * np.sum(index * sorted_asc) - (n + 1) * np.sum(sorted_asc)) / (n * np.sum(sorted_asc))
    
    return gini_coeff

def calculate_top_mass_fraction(relevance, fraction=0.9):
    """
    Calculates how many tokens (and percentage) account for a specific fraction of the total attribution mass.
    
    Returns:
        tuple: (count, percentage, cdf_array, sorted_indices)
    """
    rel = np.array(relevance)
    abs_rel = np.abs(rel)
    n = len(abs_rel)
    
    if n == 0:
        return 0, 0.0, np.array([]), np.array([])
        
    total_mass = abs_rel.sum()
    if total_mass == 0:
        # Avoid division by zero
        # Return as if uniform or none
        return n, 100.0, np.linspace(0, 1, n), np.arange(n)

    # Sort descending
    sorted_indices = np.argsort(abs_rel)[::-1]
    sorted_mass = abs_rel[sorted_indices]
    cumulative_mass = np.cumsum(sorted_mass)
    cdf = cumulative_mass / total_mass
    
    # Find how many tokens account for 'fraction' mass
    count = np.searchsorted(cdf, fraction) + 1
    percentage = (count / n) * 100
    
    return count, percentage, cdf, sorted_indices, sorted_mass

def calculate_center_of_mass(relevance):
    """
    Calculates the center of mass (expected position) of the attribution.
    
    Returns:
        tuple: (absolute_center_index, relative_center_0_to_1)
    """
    rel = np.array(relevance)
    abs_rel = np.abs(rel)
    total_mass = abs_rel.sum()
    n = len(abs_rel)
    
    if n == 0 or total_mass == 0:
        return 0.0, 0.0
        
    indices = np.arange(n)
    # E[index] = sum(p_i * i)
    center_of_mass = np.sum(indices * abs_rel) / total_mass
    
    # Relative center: 0 = start, 1 = end
    # Use (n-1) as denominator because indices are 0 to n-1
    relative_center = center_of_mass / (n - 1) if n > 1 else 0.5
    
    return center_of_mass, relative_center

def calculate_early_late_ratio(relevance, split_ratio=0.5):
    """
    Calculates the ratio of mass in the first part vs the second part of the sequence.
    
    Args:
        relevance: The attribution array.
        split_ratio: The point to split early vs late (0.5 = middle).
        
    Returns:
        float: Ratio (Early Mass / Late Mass). 
               Returns infinity if Late Mass is 0.
               Returns 0 if Early Mass is 0.
    """
    rel = np.array(relevance)
    abs_rel = np.abs(rel)
    n = len(abs_rel)
    
    if n == 0:
        return 0.0
        
    split_index = int(n * split_ratio)
    
    early_mass = np.sum(abs_rel[:split_index])
    late_mass = np.sum(abs_rel[split_index:])
    
    if late_mass == 0:
        return float('inf') if early_mass > 0 else 0.0
        
    return early_mass / late_mass
