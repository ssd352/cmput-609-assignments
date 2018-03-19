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

epsilon = 0.1
alpha = 0.5
gamma = 1.0
NUMBER_OF_ACTIONS = 4
Q = previous_state = previous_action = total_number_steps = None



def choose_action(state):
    global Q
    mx = np.amax(Q[state[0], state[1], :])
    arg = np.argwhere(Q[state[0], state[1], :] == mx)
    inds = np.reshape(arg, np.size(arg))
    greedy_action = np.random.choice(inds)
    # greedy_action = np.argmax(Q[state[0], state[1], :])

    toss = rand_un()
    if toss > epsilon:
        action = greedy_action
    else:
        action = rand_in_range(NUMBER_OF_ACTIONS)

    return action


def convert_action(action):
    if action == 0:
        return 0, 1
    elif action == 1:
        return 0, -1
    elif action == 2:
        return 1, 0
    elif action == 3:
        return -1, 0
    # a = action
    # if a == 4:
    #     a = 8
    # conv_action = ((a % 3) - 1, (a // 3) - 1)
    # return conv_action


def agent_init():
    global Q, total_number_steps
    """
    Hint: Initialize the variables that need to be reset before each run begins
    Returns: nothing
    """
    Q = np.zeros((7, 10, NUMBER_OF_ACTIONS))  # number of rows, columns, actions
    total_number_steps = 0


def agent_start(state):
    """
    Hint: Initialize the variables that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    global previous_state, previous_action, total_number_steps

    action = choose_action(state)
    previous_action = action
    previous_state = tuple(state)
    total_number_steps += 1
    return convert_action(action)


def agent_step(reward, state):  # returns NumPy array, reward: floating point, this_observation: NumPy array
    """
    Arguments: reward: floting point, state: integer
    Returns: action: floating point
    """
    global previous_state, previous_action, Q, total_number_steps
    # select an action, based on Q
    action = choose_action(state)

    # update Q
    Q[previous_state[0], previous_state[1], previous_action] += alpha * (
        reward + gamma * Q[state[0], state[1], action] - Q[previous_state[0], previous_state[1], previous_action])
    # print(state, "\n" , action)
    previous_action = action
    previous_state = tuple(state)
    caction = convert_action(action)
    total_number_steps += 1
    # print state, caction
    return caction


def agent_end(reward):
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    # print "Final reward is ", reward
    # do learning and update Q
    global total_number_steps
    Q[previous_state[0], previous_state[1], previous_action] += alpha * (
         reward - Q[previous_state[0], previous_state[1], previous_action])
    total_number_steps += 1
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
        return pickle.dumps(np.max(Q, axis=1), protocol=0)
    elif in_message.lower() == 'total_steps'.lower():
        return total_number_steps
    else:
        return "I don't know what to return!!"


if __name__ == "__main__":
    agent_init()
    agent_start((3, 0))
    agent_step(-1, (3, 1))
