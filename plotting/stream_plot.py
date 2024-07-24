from t_cell_sim.sample.distributions import generate_binned_gamma, generate_fate_prob_from_affinity_bins
from seq_plotter import SeqPlotter
from t_cell_sim.utils import f_norm

import streamlit as st

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    col_params, col_graphs = st.columns([0.5, 0.5])
    with col_params:
        with st.container(border=True):
            st.subheader("Distribution Parameters:", divider="grey")
            num_bins = st.slider("Choose number of bins", 100, 10000, 1000, 100)
            alpha = st.slider(r"Choose $\alpha$ value for $\Gamma$ distribution of self-affinities:", 1., 3., 2., 0.1)
        with st.container(border=True):
            st.subheader(r"Parameters for Gaussians $f_{reg}$ and $f_{conv}$:", divider="grey")
            st.text("Factors stretch the function, there is no constraint for the Gaussians to be a distribution." +
                    "\n\nFor `factor=1` the Gaussian is a normal distribution.")
            st.text("It is enforced that")
            col_conv, col_reg = st.columns(2)
            with col_conv:
                mu_conv = st.slider(r"$\mu_{conv}$:", 0., 10., 5., 0.5)
                sigma_conv = st.slider(r"$\sigma_{conv}$:", 0.1, 2., 1., 0.1)
                f0_conv = st.slider(r"${factor}_{conv}$:", 0.5, 3., 1.5, 0.1)
            with col_reg:
                mu_reg = st.slider(r"$\mu_{reg}$:", mu_conv + 0.5, 10.5, 6.5, 0.5)
                sigma_reg = st.slider(r"$\sigma_{reg}$:", 0.1, 2., 1., 0.1)
                f0_reg = st.slider(r"${factor}_{reg}$:", 0.5, 3., 1.5, 0.1)

    f_conv = f_norm(mu_conv, sigma_conv, f0_conv)
    f_reg = f_norm(mu_reg, sigma_reg, f0_reg)

    bins, affinities = generate_binned_gamma(num_bins, alpha_hyper=alpha)
    t_cells = generate_fate_prob_from_affinity_bins(bins, affinities, f_conv, f_reg)
    seq_plotter = SeqPlotter(t_cells)
    with col_graphs:
        with st.container(border=True):
            st.subheader("Distribution at each affinity:", divider="grey")
            st.plotly_chart(seq_plotter.plotly_stacked_bar_plot())
        with st.container(border=True):
            st.subheader("Density plot of the distribution:", divider="grey")
            st.plotly_chart(seq_plotter.plotly_density_plot())
