from src.core.distributions import generate_binned_gamma
from src.core.t_cells import TCellType
from src.plotting.model_utils import generate_fate_prob_from_affinity_bins
from seq_plotter import SeqPlotter
from src.utils import f_norm
"""
For debugging purposes only. This script is not used in the Streamlit app.
"""
if __name__ == "__main__":

    num_bins = 1000
    alpha = 2.0
    mu_conv = 5.
    sigma_conv = 1.
    f0_conv = 1.5
    mu_reg = 6.5
    sigma_reg = 1.
    f0_reg = 1.5

    f_conv = f_norm(mu_conv, sigma_conv, f0_conv)
    f_reg = f_norm(mu_reg, sigma_reg, f0_reg)

    bins, affinities = generate_binned_gamma(num_bins, alpha_hyper=alpha)
    t_cells = generate_fate_prob_from_affinity_bins(bins, affinities, f_conv, f_reg)
    seq_plotter = SeqPlotter(t_cells)
    min_dead = seq_plotter.find_minimal_affinity_for_type(TCellType.DEAD)
    for val in min_dead:
        print(f"Positive selection ≤{val:.2f} < Negative selection ≥{val:.2f}")
    seq_plotter.plotly_stacked_bar_plot()
    seq_plotter.plotly_density_plot()
    types_to_cum_density = seq_plotter.calculate_cumulative_densities()
    for k, v in types_to_cum_density.items():
        print(f"Joint cumulative density of T cell type *{k}* is {v:.23}")