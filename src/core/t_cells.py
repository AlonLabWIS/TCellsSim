from abc import ABCMeta
from enum import Enum


class TCellType(Enum):
    CONV = 1
    REG = 2
    DEAD = 3
    THYMOCYTE = 4

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    @staticmethod
    def get_type_from_str(type_str: str):
        if type_str == "CONV":
            return TCellType.CONV
        elif type_str == "REG":
            return TCellType.REG
        elif type_str == "DEAD":
            return TCellType.DEAD
        else:
            return TCellType.THYMOCYTE


class TCellProb(metaclass=ABCMeta):
    """
    A probabilistic description of a T-cell fate at a given affinity for self.
    """

    def __init__(self, self_affinity: float, prob_at_affinity: float, total_density_at_affinity: float):
        """
        :param self_affinity: The self affinity of a T-cell. Positive value, unbounded but usually in the range [0, 10].
        Arbitrary units.
        :param prob_at_affinity: The probability for the t cell of this type given its self affinity.
        :param total_density_at_affinity: The value of the PDF of Thymocytes population to self antigens at the given
        self affinity.
        """
        TCellProb.__check_valid_prob(prob_at_affinity)
        self.self_affinity = self_affinity
        self.prob_at_affinity = prob_at_affinity
        self.total_density_at_affinity = total_density_at_affinity

    def get_t_cell_type(self) -> str:
        pass

    def joint_prob_at_affinity(self) -> float:
        return self.prob_at_affinity * self.total_density_at_affinity

    @staticmethod
    def __check_valid_prob(prob: float):
        if prob < 0 or prob > 1:
            raise ValueError("Probability must be between 0 and 1")


class ConvTCellProb(TCellProb):
    """
    A probabilistic description of a conventional T-cell  at a given affinity for self.
    """

    def get_t_cell_type(self) -> str:
        return str(TCellType.CONV)


class RegTCellProb(TCellProb):
    """
    A probabilistic description of a regulatory T-cell  at a given affinity for self.
    """

    def get_t_cell_type(self) -> str:
        return str(TCellType.REG)


class DeadTCellProb(TCellProb):
    """
    A probabilistic description of a dead T-cell at a given affinity for self.
    """

    def get_t_cell_type(self) -> str:
        return str(TCellType.DEAD)


class Thymocyte(TCellProb):
    """
    A probabilistic description of a thymocyte (before entering the thymus) at a given affinity for self.
    """

    def __init__(self, self_affinity: float, prob_at_affinity: float, total_density_at_affinity: float):
        if prob_at_affinity != total_density_at_affinity:
            raise ValueError("Thymocytes must have the same probability and total density at a given affinity.")
        super().__init__(self_affinity, prob_at_affinity, total_density_at_affinity)

    def get_t_cell_type(self) -> str:
        return str(TCellType.THYMOCYTE)


class TCellsProbFactory:

    @staticmethod
    def create_t_cell(t_cell_type: TCellType, self_affinity: float, prob_at_affinity: float,
                      total_density_at_affinity: float) -> TCellProb:
        """
        Create a TCell object of the specified type.
        :param t_cell_type: The type of TCell to create.
        :param self_affinity: The affinity of the TCell.
        :param prob_at_affinity: The probability of the TCell at the specified affinity.
        :param total_density_at_affinity: The total density at the specified affinity.
        :return: A TCell object of the specified type.
        """
        if t_cell_type == TCellType.CONV:
            return ConvTCellProb(self_affinity, prob_at_affinity, total_density_at_affinity)
        elif t_cell_type == TCellType.REG:
            return RegTCellProb(self_affinity, prob_at_affinity, total_density_at_affinity)
        elif t_cell_type == TCellType.DEAD:
            return DeadTCellProb(self_affinity, prob_at_affinity, total_density_at_affinity)
        else:
            return Thymocyte(self_affinity, prob_at_affinity, total_density_at_affinity)
