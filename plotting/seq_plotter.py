from typing import Sequence, Dict, List

from t_cell_sim.sample.t_cells import TCellProb

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class SeqPlotter:
    df_columns = ["Affinity", "t_cell_type_prob", "joint_prob", "T-cell Type"]

    def __init__(self, t_cells: Sequence[TCellProb]):
        self.df = SeqPlotter.__transform_t_cells_seq_to_df(t_cells)

    def plotly_stacked_bar_plot(self) -> go.Figure:
        return px.bar(self.df, x=SeqPlotter.df_columns[0], y=SeqPlotter.df_columns[1], color=SeqPlotter.df_columns[3],
                      labels={SeqPlotter.df_columns[1]: "Probability"})

    def plotly_density_plot(self) -> go.Figure:
        df = self.df.copy()[[SeqPlotter.df_columns[0]] + SeqPlotter.df_columns[2:]]
        #  This should be one as the sum of the probabilities should be 1
        all_densities = df.groupby(SeqPlotter.df_columns[0])[SeqPlotter.df_columns[2]].sum()
        all_densities_df = pd.DataFrame(all_densities).reset_index()
        all_densities_df[SeqPlotter.df_columns[3]] = "All"
        df = pd.concat([df, all_densities_df], ignore_index=True)
        return px.line(df, x=SeqPlotter.df_columns[0], y=SeqPlotter.df_columns[2], color=SeqPlotter.df_columns[3])

    @staticmethod
    def __transform_t_cells_seq_to_df(t_cells: Sequence[TCellProb]) -> pd.DataFrame:
        df_dict: Dict[str, List[...]] = {col: [None] * len(t_cells) for col in SeqPlotter.df_columns[:4]}
        for i, t_cell in enumerate(t_cells):
            df_dict[SeqPlotter.df_columns[0]][i] = t_cell.self_affinity
            df_dict[SeqPlotter.df_columns[1]][i] = t_cell.prob_at_affinity
            df_dict[SeqPlotter.df_columns[2]][i] = t_cell.joint_prob_at_affinity()
            df_dict[SeqPlotter.df_columns[3]][i] = t_cell.get_t_cell_type()
        return pd.DataFrame.from_dict(df_dict)
