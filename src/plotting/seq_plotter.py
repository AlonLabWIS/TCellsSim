from typing import Sequence, Dict, List

from src.core.t_cells import TCellProb, TCellType

import pandas as pd
import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import argrelmin
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
        self._full_df = self.__create_density_plot()

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
        return px.line(self._full_df, x=SeqPlotter._df_columns[0], y=SeqPlotter._df_columns[2],
                       color=SeqPlotter._df_columns[3])

    def calculate_cumulative_densities(self) -> Dict[str, float]:
        """
        For all types in the df, calculate the cumulative joint density across affinities.
        Add a "THYMOCYTE" type which sums over all other types. It should have a value of 1 up to a small error,
        depending on the number of bins and smoothness of the density function.
        :return:
        """
        density_plot = self._full_df.groupby(SeqPlotter._df_columns[3])
        type_to_density: Dict[str, float] = {}
        for t_cell_type, density_of_type in density_plot:
            joint_cumulative_density = trapezoid(density_of_type[SeqPlotter._df_columns[2]],
                                                 density_of_type[SeqPlotter._df_columns[0]])
            type_to_density[str(t_cell_type)] = joint_cumulative_density
        return type_to_density

    def calculate_reg_ratio_from_living(self) -> float:
        """
        Calculate the ratio of regulatory T-cells to living T-cells
        :return:
        """
        density_plot = self._full_df.groupby(SeqPlotter._df_columns[3])
        reg_density = density_plot.get_group(str(TCellType.REG))
        conv_density = density_plot.get_group(str(TCellType.CONV))
        reg_cumulative_density = trapezoid(reg_density[SeqPlotter._df_columns[2]],
                                           reg_density[SeqPlotter._df_columns[0]])
        conv_cumulative_density = trapezoid(conv_density[SeqPlotter._df_columns[2]],
                                            conv_density[SeqPlotter._df_columns[0]])
        return reg_cumulative_density / (reg_cumulative_density + conv_cumulative_density)

    def find_minimal_affinity_for_type(self, t_cell_type: TCellType, order: int = 3) -> np.ndarray[float]:
        """
        Find the affinity with minimal probability for a given T-cell type, excluding edges
        :param t_cell_type: The type of T-cell to find the minimal affinity for.
        :param order: The radius of the minimum (indices around). Defaults to 3.
        :return: The minimal affinity for the given T-cell type.
        """
        sorted_type_affinity, sorted_type_density = self.get_affinity_joint_density_per_type(t_cell_type)
        min_indices = argrelmin(sorted_type_density, order=order)[0]
        min_inds_no_edges = min_indices[~np.isin(min_indices, [0, len(sorted_type_density) - 1])]
        return sorted_type_affinity[min_inds_no_edges]

    def get_affinity_joint_density_per_type(self, t_cell_type: TCellType) -> (np.ndarray[float], np.ndarray[float]):
        """
        Get the joint density of a T-cell type at each affinity.
        :param t_cell_type: The type of T-cell to get the joint density for.
        :return: Ordered affinity and corresponding joint density of the T-cell type.
        """
        affinity_density_df = self._full_df.groupby(SeqPlotter._df_columns[3])[[
            SeqPlotter._df_columns[0], SeqPlotter._df_columns[2]]].get_group(
            str(t_cell_type))
        affinity_density_df.sort_values(by=SeqPlotter._df_columns[0], inplace=True)
        affinity, density = affinity_density_df.to_numpy().T
        return affinity, density

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
        all_densities_df[SeqPlotter._df_columns[3]] = str(TCellType.THYMOCYTE)
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
