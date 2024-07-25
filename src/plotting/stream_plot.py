import os
import sys
os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))  # Change working directory to root of the project

print(os.getcwd())
print(sys.path)

from src.core.distributions import generate_binned_gamma
from src.plotting.model_utils import generate_fate_prob_from_affinity_bins
from seq_plotter import SeqPlotter
from src.utils import f_norm

import streamlit as st

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    col_params, col_graphs = st.columns([0.5, 0.5])  # Two columns equal width
    with col_params:  # Column for parameter tuning
        # Distribution parameters
        with st.container(border=True):
            st.subheader("Distribution Parameters:", divider="grey")
            num_bins = st.slider("Choose number of bins", 100, 10000, 1000, 100)
            alpha = st.slider(r"Choose $\alpha$ value for $\Gamma$ distribution of self-affinities:", 1., 8., 2., 0.1)
        # Gaussian parameters for conv, reg
        with st.container(border=True):
            st.subheader(r"Parameters for Gaussians $f_{reg}$ and $f_{conv}$:", divider="grey")
            st.text("Factors stretch the function, there is no constraint for the Gaussians to be a distribution." +
                    "\n\nFor `factor=1` the Gaussian is a normal distribution.")
            st.text("It is enforced that")
            col_conv, col_reg = st.columns(2)
            # conv parameters
            with col_conv:
                mu_conv = st.slider(r"$\mu_{conv}$:", 0., 10., 5., 0.1)
                sigma_conv = st.slider(r"$\sigma_{conv}$:", 0.1, 4., 1., 0.1)
                f0_conv = st.slider(r"${factor}_{conv}$:", 0.5, 5., 1.5, 0.1)
            # reg parameters
            with col_reg:
                mu_reg = st.slider(r"$\mu_{reg}$:", mu_conv + 0.5, 10.5, 6.5, 0.5)
                sigma_reg = st.slider(r"$\sigma_{reg}$:", 0.1, 4., 1., 0.1)
                f0_reg = st.slider(r"${factor}_{reg}$:", 0.5, 5., 1.5, 0.1)

    # Initialize functions to derive conv, reg distribution per affinity. Sum should not exceed 1 anywhere.
    f_conv = f_norm(mu_conv, sigma_conv, f0_conv)
    f_reg = f_norm(mu_reg, sigma_reg, f0_reg)

    bins, affinities = generate_binned_gamma(num_bins, alpha_hyper=alpha)
    # Graphs column
    with col_graphs:
        # If `f_conv` and `f_reg` are negative or sum up to more than one anywhere, the simulation is invalid.
        try:
            t_cells = generate_fate_prob_from_affinity_bins(bins, affinities, f_conv, f_reg)
        except ValueError:
            st.error("Illegal probability values found, please adjust the parameters", icon="💀")
            st.stop()
        else:
            seq_plotter = SeqPlotter(t_cells)
            with st.container(border=True):
                st.subheader("Distribution at each affinity:", divider="grey")
                st.plotly_chart(seq_plotter.plotly_stacked_bar_plot())
            with st.container(border=True):
                st.subheader("Density plot of the distribution:", divider="grey")
                st.plotly_chart(seq_plotter.plotly_density_plot())
                # Print joint density for each T-cell type
                types_to_cum_density = seq_plotter.calculate_cumulative_densities()
                for k, v in types_to_cum_density.items():
                    st.write(f"Joint cumulative density *{k}* is {v:.3}")
