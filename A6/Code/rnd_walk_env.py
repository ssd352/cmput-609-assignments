#!/usr/bin/env python

"""
  Author: Adam White, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Code for the Gambler's problem environment from the Sutton and Barto
  Reinforcement Learning: An Introduction Chapter 4.
  For use in the Reinforcement Learning course, Fall 2017, University of Alberta 
"""

# from utils import rand_norm, rand_in_range, rand_un
import numpy as np

# size = 1000
k = 100
num_total_states = 1000  # num_total_states: integer
current_state = None


def env_init():
    global current_state
    current_state = np.zeros(1)


def env_start():
    """ returns numpy array """
    global current_state

    # state = rand_in_range(num_total_states) + 1 # This is required for exploring starts
    state = num_total_states // 2
    current_state = np.asarray([state])
    return current_state


def env_step(action):
    """
    Arguments
    ---------
    action : int
        the action taken by the agent in the current state

    Returns
    -------
    result : dict
        dictionary with keys {reward, state, isTerminal} containing the results
        of the action taken
    """
    global current_state
    if action < -k or action > k or action == 0:
        print "Invalid action taken!!"
        print "action : ", action
        print "current_state : ", current_state
        exit(1)

    current_state += action
    # print(current_state)
    
    reward = 0.0
    is_terminal = False
    if current_state[0] > num_total_states:
        is_terminal = True
        current_state = None
        reward = 1.0
    elif current_state[0] < 1:
        is_terminal = True
        reward = -1.0
        current_state = None

    result = {"reward": reward, "state": current_state, "isTerminal": is_terminal}

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
