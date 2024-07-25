from typing import Callable, Sequence, List

import numpy as np

from src.core.distributions import complement_dist
from src.core.t_cells import TCellProb, TCellType, TCellsProbFactory


def generate_fate_prob_from_affinity_bins(affinity_bins: np.array, affinity_density: np.array,
                                          f_conv: Callable[[np.array], np.array],
                                          f_reg: Callable[[np.array], np.array]) -> Sequence[TCellProb]:
    """
    Generate a sequence of TCellProb objects from the density of affinities and their discretized values
     (`affinity_bins`).
    :param affinity_bins: Values of affinities in arbitrary units. Shape is (n,). The larger `n` is, the smoother the
    resulting simulation. Values *should* be linearly spaced, but this is not enforced.
    :param affinity_density: Density of affinities at each value in `affinity_bins`. Shape is (n,). *Must* be positive,
    and integrate to 1 up to error due to discretization.
    :param f_conv: An arbitrary function
    :math: `f:\\mathbb{R}^{+}\\to\\left[0, 1\\right]` that generates the probability of a conv T-cell given affinity.
    :param f_reg: Same as above for regulatory T-cells.
    :raises ValueError: If the sum of `f_conv` and `f_reg` at each affinity value is greater than 1, or if any of which
    is negative anywhere.
    :return: A sequence of TCellProb objects of length `3n` where `n` is the number of affinities.
    """
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
            t_cells[i * len(t_cell_types) + j] = TCellsProbFactory.create_t_cell(t_cell_type, affinity,
                                                                                 prob_at_affinity,
                                                                                 affinity_density_at_i)
    return t_cells
