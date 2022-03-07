"""
This implements the Average Oracle (AO) 
which takes the game matrix as input

The meta solver is borrowed from https://arxiv.org/abs/2106.02745
More solver is needed, as we include CFR into consideration
"""

import numpy as np

# import numpy as np

from open_spiel.python.algorithms import lp_solver
from meta_solvers.prd_solver import projected_replicator_dynamics

# from open_spiel.python.egt import alpharank
import open_spiel.python.egt.utils as utils
from meta_solvers import alpha_rank
import pyspiel

EPSILON_MIN_POSITIVE_PROBA = 1e-8


def uniform_average(meta_games):
    supports = meta_games[0].shape
    # print(supports)
    average_distribution = []
    for i in range(len(supports)):
        average_distribution.append(np.ones(supports[i]) / supports[i])
    return average_distribution


def linear_average(meta_games):
    supports = meta_games[0].shape
    # print(supports)
    average_distribution = []
    for i in range(len(supports)):
        average_distribution.append(np.array([j + 1 for j in range(supports[i])]))
    return average_distribution


def last_one_average(meta_games):
    supports = meta_games[0].shape
    # print(supports)
    average_distribution = []
    for i in range(len(supports)):
        last_one_distribution = np.zeros(supports[i])
        last_one_distribution[-1] = 1
        average_distribution.append(last_one_distribution)
    return average_distribution


def nash_average(meta_games):
    return nash_strategy(meta_games)


def renormalize(probabilities):
    """Replaces all negative entries with zeroes and normalizes the result.

    Args:
      probabilities: probability vector to renormalize. Has to be one-dimensional.

    Returns:
      Renormalized probabilities.
    """
    probabilities[probabilities < 0] = 0
    probabilities = probabilities / np.sum(probabilities)
    return probabilities


def get_joint_strategy_from_marginals(probabilities):
    """Returns a joint strategy matrix from a list of marginals.

    Args:
      probabilities: list of probabilities.

    Returns:
      A joint strategy from a list of marginals.
    """
    probas = []
    for i in range(len(probabilities)):
        probas_shapes = [1] * len(probabilities)
        probas_shapes[i] = -1
        probas.append(probabilities[i].reshape(*probas_shapes))
    result = np.product(probas)
    return result.reshape(-1)


def nash_strategy(meta_games, return_joint=False):
    """Returns nash distribution on meta game matrix.

    This method only works for two player zero-sum games.

    Args:
      meta_games
      return_joint: If true, only returns marginals. Otherwise marginals as well
        as joint probabilities.

    Returns:
      Nash distribution on strategies.
    """
    # meta_games = solver.get_meta_game()
    if not isinstance(meta_games, list):
        meta_games = [meta_games, -meta_games]
    meta_games = [x.tolist() for x in meta_games]
    if len(meta_games) != 2:
        raise NotImplementedError(
            "nash_strategy solver works only for 2p zero-sum"
            "games, but was invoked for a {} player game".format(len(meta_games))
        )
    nash_prob_1, nash_prob_2, _, _ = lp_solver.solve_zero_sum_matrix_game(
        pyspiel.create_matrix_game(*meta_games)
    )
    result = [
        renormalize(np.array(nash_prob_1).reshape(-1)),
        renormalize(np.array(nash_prob_2).reshape(-1)),
    ]

    if not return_joint:
        return result
    else:
        joint_strategies = get_joint_strategy_from_marginals(result)
        return result, joint_strategies


def prd_strategy(meta_games, return_joint=False):
    """Computes Projected Replicator Dynamics strategies.

    Args:
      meta_games
      return_joint: If true, only returns marginals. Otherwise marginals as well
        as joint probabilities.

    Returns:
      PRD-computed strategies.
    """
    # meta_games = solver.get_meta_game()
    if not isinstance(meta_games, list):
        meta_games = [meta_games, -meta_games]
    # kwargs = solver.get_kwargs()
    result = projected_replicator_dynamics.projected_replicator_dynamics(meta_games)
    if not return_joint:
        return result
    else:
        joint_strategies = get_joint_strategy_from_marginals(result)
        return result, joint_strategies


def alpha_rank_average(meta_games):
    # print()
    supports = meta_games[0].shape
    meta_strategies = []
    for i in range(len(supports)):
        meta_strategies.append(np.zeros(supports[i]))
    payoffs_are_hpt_format = utils.check_payoffs_are_hpt(meta_games)
    try:
        suggested_alpha = alpha_rank.suggest_alpha(meta_games)
        alpha_rank_values = alpha_rank.compute_and_report_alpharank(
            meta_games, alpha=suggested_alpha
        )
    except ValueError:
        alpha_rank_values = alpha_rank.compute_and_report_alpharank(
            meta_games, alpha=1e2
        )
    # print(alpha_rank_values)
    # print(strat_labels)
    num_strats_per_population = utils.get_num_strats_per_population(
        meta_games, payoffs_are_hpt_format
    )
    for i in range(len(alpha_rank_values)):
        strat_label = utils.get_strat_profile_from_id(num_strats_per_population, i)
        # print("{}-{}".format(strat_label, alpha_rank_values[i]))
        for j in range(len(strat_label)):
            meta_strategies[j][strat_label[j]] += alpha_rank_values[i]
    # for values in meta_strategies:
    #     renormalize(values)
    return meta_strategies
