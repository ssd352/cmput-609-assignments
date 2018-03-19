#!/usr/bin/env python

"""
  Author: Adam White, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Code for the Gambler's problem environment from the Sutton and Barto
  Reinforcement Learning: An Introduction Chapter 4.
  For use in the Reinforcement Learning course, Fall 2017, University of Alberta
"""

from __future__ import division, print_function
from utils import rand_norm, rand_in_range, rand_un
import numpy as np

current_state = None
grid_size = (9, 6)


def is_obstacle(x, y):
    if x == 7 and (3 <= y <= 5):
        return True
    if x == 2 and (2 <= y <= 4):
        return True
    if x == 5 and y == 1:
        return True
    return False


def env_init():
    global current_state
    current_state = np.zeros((1, 2))


def env_start():
    """ returns numpy array """
    global current_state  # current_state : tuple (col(y), row(x))

    state = (0, 3)  # The S letter in the grid
    current_state = np.asarray(state)
    return current_state


def env_step(action):
    """
    Arguments
    ---------
    action : tuple (vy, vx)
        the action taken by the agent in the current state

    Returns
    -------
    result : dict
        dictionary with keys {reward, state, isTerminal} containing the results
        of the action taken
    """
    global current_state

    # current_state[0] += action[0]
    # current_state[1] += action[1]
    if not is_obstacle(current_state[0] + action[0], current_state[1] + action[1]):
        current_state[0] += action[0]
        current_state[1] += action[1]

    # checking the boundaries
    if current_state[0] < 0:
        current_state[0] = 0
    if current_state[0] >= grid_size[0]:
        current_state[0] = grid_size[0] - 1

    if current_state[1] < 0:
        current_state[1] = 0
    if current_state[1] >= grid_size[1]:
        current_state[1] = grid_size[1] - 1

    reward = 0.0
    is_terminal = False
    if np.array_equal(current_state, np.array((8, 5))):
        is_terminal = True
        reward = 1.0
        current_state = None
    # print(current_state)
    result = {"reward": reward, "state": np.copy(current_state), "isTerminal": is_terminal}

    return result


def env_cleanup():
    #
    return


def env_message(in_message):  # returns string, in_message: string
    """
    Arguments
    ---------
    in_message : string
        the message being passed

    Returns
    -------
    string : the response to the message
    """
    return ""


if __name__ == "__main__":
    env_init()
    env_start()
    env_step((0, 1))
    env_step((0, 1))
    env_step((0, 1))
    env_step((0, 1))
    env_step((0, 1))
    env_step((0, 1))
    env_step((0, 1))
    env_step((0, 1))
    env_step((0, 1))
    env_step((1, 0))
    env_step((1, 0))
    env_step((1, 0))
    env_step((1, 0))
    env_step((0, -1))
    env_step((0, -1))
    env_cleanup()
