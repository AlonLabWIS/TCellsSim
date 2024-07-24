from typing import Union, Sequence, Callable, List
import numpy as np
import scipy.stats as st

from .t_cells import TCellProb, TCellsFactory, TCellType


def complement_dist(prob_arr: np.array, *prob_arrs: np.array, return_stacked=False) -> np.array:
    """
    Calculate the 1's complement of a sum of vector of probabilities with equal size.
    :param prob_arr: Array of densities from a pdf of some RV, could also be mass from a discrete RV. Shape is (n,)
    :param prob_arrs: Additional arrays of densities from pdfs of the same RV. Shape of each is (n,).
    :param return_stacked: If True, return the stacked array of pdfs with shape (l+1,n) where l is the number of arrays
        passed as prob_arrs. Else return the complement of the sum of the pdfs with shape (n,).
    :raises ValueError: If the sum of the pdfs at any point is greater than 1 or if any value is less than 0.
    :return: The 1's complement of the sum of the probabilities. Size of (n,) or (l+1,n) depending on return_stacked.
    """
    dists_stacked = np.stack((prob_arr, *prob_arrs), axis=0)
    _check_no_zero_prob(dists_stacked)
    stacked_sum = dists_stacked.sum(axis=0)
    if (stacked_sum > 1).any():
        raise ValueError("Sum of probabilities > 1")
    if return_stacked:
        return np.vstack((dists_stacked, 1 - stacked_sum))
    else:
        return 1 - stacked_sum


def sample_types_from_dist(probs: np.array, t_cell_types: Union[None, Sequence[TCellType]] = None,
                           repeats: int = 1) -> np.array:
    """
    Use the inverse transform sampling method to sample from a vector of cumulative probabilities.
    :param probs: An array of cumulative probabilities, shape is (L,N) where L is the number of categories.
    Each of the N columns MUST some up to 1. Each column is a categorical distribution.
    :param t_cell_types: Optional. Labels for each categorical distribution. Of size (L,). If `None`, labels
    correspond to the rows in `dist_arr`.
    :param repeats: Number of samples to generate (R), defaults to 1.
    :return: An array of the sampled types, shape is (N,R). If `t_cell_types` is `None`, the array is integer valued.
    Otherwise, it is of type `TCellType`.
    :raises ValueError: If the columns of `dist_arr` do not sum up to 1 with some tolerance or if any probability
    is less than 0.

    Notes
    -----
    The last label (last row in `dist_arr`) could be sampled slightly more frequently than the others stated due to
    numerical instability.
    To avoid this issue ensure that the columns sum up to 1 exactly.
    """
    num_cat = probs.shape[0]
    if t_cell_types is not None and len(tuple(t_cell_types)) != num_cat:
        raise ValueError("Number of labels must match number of categories")
    _check_legal_dist(probs, axis=0)
    generated_types = np.random.uniform(size=(probs.shape[1], repeats))

    # Assign row number to generated type using the inverse transform sampling method.
    numerical_types: int = num_cat - (probs - generated_types > 0).sum(axis=0)
    if t_cell_types is not None:
        t_cell_types = np.array(t_cell_types)
        return t_cell_types[numerical_types]
    else:
        return numerical_types


def generate_fate_prob_from_affinity_bins(affinity_bins: np.array, affinity_density: np.array,
                                          f_conv: Callable[[np.array], np.array],
                                          f_reg: Callable[[np.array], np.array]) -> Sequence[TCellProb]:
    conv_density = f_conv(affinity_bins)
    reg_density = f_reg(affinity_bins)
    full_density_per_bin = complement_dist(conv_density, reg_density, return_stacked=True)
    t_cell_types = [TCellType.CONV, TCellType.REG, TCellType.DEAD]
    t_cells: List[TCellProb] = [None] * full_density_per_bin.size
    for i in range(affinity_density.size):
        affinity_density_at_i = affinity_density.item(i)
        affinity = affinity_bins.item(i)
        for j, t_cell_type in enumerate(t_cell_types):
            prob_at_affinity = full_density_per_bin.item(j, i)
            t_cells[i * len(t_cell_types) + j] = TCellsFactory.create_t_cell(t_cell_type, affinity,
                                                                             prob_at_affinity,
                                                                             affinity_density_at_i)
    return t_cells


def generate_affinities_gamma_distrib(num_t_cells: int = int(10e6), alpha_hyper: float = 2.,
                                      beta_hyper: Union[None, float] = None):
    g = _generate_gamma_dist(alpha_hyper, beta_hyper)
    return np.sort(g.rvs(num_t_cells))


def generate_binned_gamma(num_bins: int, top_cutoff_bin: float = 0.999, alpha_hyper: float = 2.,
                          beta_hyper: Union[None, float] = None) -> np.array:
    g = _generate_gamma_dist(alpha_hyper, beta_hyper)
    top_bin_val = g.ppf(top_cutoff_bin)
    bins = np.linspace(0, top_bin_val, num_bins)
    return bins, g.pdf(bins)


def _generate_gamma_dist(alpha_hyper: float, beta_hyper: Union[None, float]) -> np.array:
    if beta_hyper is None:
        g = st.gamma(alpha_hyper)
    else:
        g = st.gamma(alpha_hyper, beta_hyper)
    return g


def _check_no_zero_prob(dist_arr: np.array) -> None:
    if (dist_arr == 0).any():
        raise ValueError("Zero probabilities found")


def _check_legal_dist(dist_arr: np.array, axis=None) -> None:
    _check_no_zero_prob(dist_arr)
    if np.isclose(dist_arr.sum(axis=axis), 1.0):
        raise ValueError("Sum of probabilities along axis is not 1")
