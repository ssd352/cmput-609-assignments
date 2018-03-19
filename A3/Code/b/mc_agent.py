#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Skeleton code for Monte Carlo Exploring Starts Control Agent
           for use on A3 of Reinforcement learning course University of Alberta Fall 2017
 
"""

from __future__ import division
from utils import rand_in_range, rand_un
import numpy as np
import pickle

# states = actions =
state_action_pairs = policy = Q = Returns = None


def record_state_action(state, action):
    global Returns, state_action_pairs
    # if (state, action) not in Returns:
    #     Returns[(state, action)] = []
    state_action_pairs.add((state, action))


def agent_init():
    """
    Hint: Initialize the variables that need to be reset before each run begins
    Returns: nothing
    """
    global policy, Q, Returns  # states, actions, state_action_pairs
    Returns = np.zeros((101, 51))  # {}
    Q = np.zeros((101, 51))
    # initialize the policy array in a smart way
    # policy = np.array([np.min([s, 100 - s]) for s in range(100 + 1)])
    policy = np.ones(100 + 1)


def agent_start(state):
    """
    Hint: Initialize the variables that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    global policy, Returns, state_action_pairs  # actions, states, state_action_pairs

    state_action_pairs = set()
    s = int(state[0])
    # pick the first action, don't forget about exploring starts
    action = rand_in_range(np.min([s, 100 - s])) + 1
    record_state_action(s, action)
    return action


def agent_step(reward, state):  # returns NumPy array, reward: floating point, this_observation: NumPy array
    """
    Arguments: reward: floating point, state: integer
    Returns: action: integer
    """
    global Q, Returns  # actions, states, state_action_pairs
    # select an action, based on Q
    s = int(state[0])
    action = policy[s]
    record_state_action(s, action)
    return action


def agent_end(reward):
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    global Q, policy, Returns, state_action_pairs  # states, actions, state_action_pairs

    # do learning and update pi

    for state, action in state_action_pairs:
        Returns[state][action] += 1
        Q[state][action] += (reward - Q[state][action]) / Returns[state][action]

    for state, action in state_action_pairs:
        policy[state] = np.argmax(Q[state][1:]) + 1
    return


def agent_cleanup():
    """
    This function is not used
    """
    # clean up
    return


def agent_message(in_message):  # returns string, in_message: string
    global Q
    """
    Arguments: in_message: string
    returns: The value function as a string.
    This function is complete. You do not need to add code here.
    """
    # should not need to modify this function. Modify at your own risk
    if in_message == 'ValueFunction':
        return pickle.dumps(np.max(Q[1:99][1:], axis=1), protocol=0)
    else:
        return "I don't know what to return!!"

