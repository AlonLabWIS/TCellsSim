from typing import Sequence, Dict, List

from src.core.t_cells import TCellProb

import pandas as pd
import numpy as np
from scipy.integrate import trapezoid
import plotly.express as px
import plotly.graph_objects as go


class SeqPlotter:
    """
    Class to plot the distribution of T-cells. Receives the simulated probabilities and transforms to an
    easy-to-plot DataFrame.
    """

    _df_columns = ["Affinity", "t_cell_type_prob", "joint_prob", "T-cell Type"]

    def __init__(self, t_cells: Sequence[TCellProb]):
        """
        :param t_cells: Sequence of TCellProb objects.
        Every object in the sequence corresponds to a line in the DataFrame.
        """
        self._df = SeqPlotter.__transform_t_cells_seq_to_df(t_cells)

    def plotly_stacked_bar_plot(self) -> go.Figure:
        """
        X-axis is affinity, Y-axis is probability for each type at each affinity.
        """
        return px.bar(self._df, x=SeqPlotter._df_columns[0], y=SeqPlotter._df_columns[1],
                      color=SeqPlotter._df_columns[3],
                      labels={SeqPlotter._df_columns[1]: "Probability"})

    def plotly_density_plot(self) -> go.Figure:
        """
        Joint distribution of T-cell type and affinity
        """
        df = self.__create_density_plot()
        return px.line(df, x=SeqPlotter._df_columns[0], y=SeqPlotter._df_columns[2], color=SeqPlotter._df_columns[3])

    def calculate_cumulative_densities(self) -> Dict[str, float]:
        """
        For all types in the df, calculate the cumulative joint density across affinities.
        Add an "ALL" type which sums over all other types. It should have a value of 1 up to a small error, depending
        on the number of bins and smoothness of the density function.
        :return:
        """
        density_plot = self.__create_density_plot()
        density_plot = density_plot.groupby(SeqPlotter._df_columns[3])
        type_to_density: Dict[str, float] = {}
        for t_cell_type, density_of_type in density_plot:
            joint_cumulative_density = trapezoid(density_of_type[SeqPlotter._df_columns[2]],
                                                 density_of_type[SeqPlotter._df_columns[0]])
            type_to_density[str(t_cell_type)] = joint_cumulative_density
        return type_to_density

    @property
    def df(self) -> pd.DataFrame:
        """
        The dataframe to which the sequence was transformed.
        """
        return self._df

    def __create_density_plot(self):
        df = self._df.copy()[[SeqPlotter._df_columns[0]] + SeqPlotter._df_columns[2:]]
        #  This should be one as the sum of the probabilities should be 1
        all_densities = df.groupby(SeqPlotter._df_columns[0])[SeqPlotter._df_columns[2]].sum()
        all_densities_df = pd.DataFrame(all_densities).reset_index()
        all_densities_df[SeqPlotter._df_columns[3]] = "All"
        df = pd.concat([df, all_densities_df], ignore_index=True)
        return df

    @staticmethod
    def __transform_t_cells_seq_to_df(t_cells: Sequence[TCellProb]) -> pd.DataFrame:
        df_dict: Dict[str, List[...]] = {col: [None] * len(t_cells) for col in SeqPlotter._df_columns[:4]}
        for i, t_cell in enumerate(t_cells):
            df_dict[SeqPlotter._df_columns[0]][i] = t_cell.self_affinity
            df_dict[SeqPlotter._df_columns[1]][i] = t_cell.prob_at_affinity
            df_dict[SeqPlotter._df_columns[2]][i] = t_cell.joint_prob_at_affinity()
            df_dict[SeqPlotter._df_columns[3]][i] = t_cell.get_t_cell_type()
        return pd.DataFrame.from_dict(df_dict)
