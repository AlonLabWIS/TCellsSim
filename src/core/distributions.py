import numpy as np
import scipy.stats as st
from scipy.integrate import cumulative_trapezoid


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
    _check_no_neg_prob(dists_stacked)
    stacked_sum = dists_stacked.sum(axis=0)
    if (stacked_sum > 1).any():
        raise ValueError("Sum of probabilities > 1")
    if return_stacked:
        return np.vstack((dists_stacked, 1 - stacked_sum))
    else:
        return 1 - stacked_sum


def generate_gamma_distrib(n_samples: int = int(10e6), alpha_hyper: float = 2.,
                           beta_hyper: float = 0.) -> np.array:
    """
    Sample from the gamma distribution with hyperparameters `alpha_hyper` and `beta_hyper` for `n_samples` times.
    :return: An array of samples from the gamma distribution with size `(n_samples,)`.
    """
    g = _generate_gamma_dist(alpha_hyper, beta_hyper)
    return np.sort(g.rvs(n_samples))


def generate_binned_gamma(num_bins: int, top_cutoff_bin: float = 0.999, alpha_hyper: float = 2.,
                          beta_hyper: float = 0.) -> (np.array, np.array):
    """
    Generate the pdf of a gamma distribution with hyperparameters `alpha_hyper` and `beta_hyper`.

    The number of samplings of the PDF is `num_bins`. The PDF is sampled up to the `top_cutoff_bin` quantile, starting
    from zero.
    :returns:
        - bins - The values for which the density function is evaluated. A float array of shape (`num_bins`).
        - density - The respecting values for each bin. A float array of shape (`num_bins`).
    """
    g = _generate_gamma_dist(alpha_hyper, beta_hyper)
    top_bin_val = g.ppf(top_cutoff_bin)
    bins = np.linspace(0, top_bin_val, num_bins)
    return bins, g.pdf(bins)


def ppf_from_distribution(x_values: np.array, pdf_values: np.array, quantile: float = 0.85) -> float:
    """
    Generate the inverse of the cumulative distribution function for a set of samples from some distribution.
    :param x_values: The values for which the density function is evaluated. A float array of shape `(n)`.
    :param pdf_values: The respecting values for each bin. A float array of shape `(n)`.
    If the integral does not sum up to 1, then each point is divided by the sum. All values must be non
    :param quantile: The quantile to calculate the inverse for. Defaults to 0.85. MUST be in the range [0, 1].
    :return: The value of the inverse of the cumulative distribution function at the given quantile.
    """
    cdf = _cdf_from_distribution(pdf_values, x_values)
    return np.interp(quantile, cdf, x_values, left=0.).item()


def cdf_from_distribution(x_values: np.array, pdf_values: np.array, value: float, norm_to_cdf: bool = True) -> float:
    """
    Generate the cumulative distribution function for a set of samples from some distribution.
    :param x_values: The values for which the density function is evaluated. A float array of shape `(n)`.
    :param pdf_values: The respecting values for each bin. A float array of shape `(n)`.
    If the integral does not sum up to 1, then each point is divided by the sum. All values must be non
    :param value: The value to calculate the CDF for.
    :param norm_to_cdf: If True, normalize the CDF to 1. Defaults to True. If false, the CDF can be for joint probability function at a point.
    :return: The value of the CDF at the given value.
    """
    cdf = _cdf_from_distribution(pdf_values, x_values, norm_to_cdf)
    return np.interp(value, x_values, cdf, left=0.).item()


def _cdf_from_distribution(pdf_values, x_values, norm_to_cdf: bool = True) -> np.ndarray[float]:
    _check_no_neg_prob(pdf_values)
    cdf = cumulative_trapezoid(pdf_values, x_values, initial=0)
    if norm_to_cdf:
        cdf /= cdf[-1]
    return cdf


def _generate_gamma_dist(alpha_hyper: float, beta_hyper) -> np.array:
    return st.gamma(alpha_hyper, beta_hyper)


def _check_no_neg_prob(dist_arr: np.array) -> None:
    if (dist_arr < 0).any():
        raise ValueError("Zero probabilities found")


def _check_legal_dist(dist_arr: np.array, axis=None) -> None:
    _check_no_neg_prob(dist_arr)
    if np.isclose(dist_arr.sum(axis=axis), 1.0):
        raise ValueError("Sum of probabilities along axis is not 1")
