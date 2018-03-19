#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Skeleton code for Monte Carlo Exploring Starts Control Agent
           for use on A3 of Reinforcement learning course University of Alberta Fall 2017
 
"""

from __future__ import division
import numpy as np
from tiles3 import IHT, tiles

size = 1000
k = 100
gamma = 1
previous_state = None
previous_action = None
possible_actions = None
seed = 0
fourier_order = None
w = None
agent_type = None
iht = None

num_tilings = {
    'tile': 50,
    'aggregation': 1
}
tile_widths = {
    'tile': 0.2,
    'aggregation': 0.1
}
alphas = {
    'tabular': 0.1,  # 0.1
    'tile':  0.001 / num_tilings['tile'],  # 0.001 / num_tilings['tile']
    'aggregation': 0.001,  # 0.001
    'fourier': 0.005  # 0.00005
}
IHT_SIZE = 1024


def x_tab(state):
    s = int(state - 1)
    tmp = np.zeros((size, 1))
    if s in range(size):
        tmp[s] = 1.0
    return tmp


def x_tile(state):
    global iht, agent_type
    indices = tiles(iht, num_tilings[agent_type], (state - 1) / (size * tile_widths[agent_type]))
    t = np.zeros((IHT_SIZE, 1))#(num_tilings[agent_type] / tile_widths[agent_type], 1))
    t[indices] = 1.0
    return t


def x_agg(state):
    global agent_type
    t = np.zeros((num_tilings[agent_type] / tile_widths[agent_type], 1))
    t[int(np.floor((state - 1) / size / tile_widths[agent_type]))] = 1.0
    return t


def x_four(state):
    global fourier_order
    # t = np.zeros((fourier_order, 1))
    # t = np.sin((np.arange(1, size + 1) - (size + 1.0) / 2.0) / (size - 1))
    l = [1]
    l += [np.sin(i * np.pi * state / (size + 1))
                     for i in range(1, fourier_order + 1)]
    l += [np.cos(i * np.pi * state / (size + 1))
                     for i in range(1, fourier_order + 1)]
# [np.sin(i * np.pi * (state - (size + 1.0) / 2.0) / (size + 1))
#                     for i in range(1, fourier_order + 1)]
#     print("len is", len(l))

    t = np.fromiter(l, dtype=np.float64)
    t = t.reshape((2 * fourier_order + 1, 1))
    # print(t)
    return t


xs = {
    'tabular': x_tab,
    'tile': x_tile,
    'aggregation': x_agg,
    'fourier': x_four
}


def x(state):
    global agent_type
    tmp = xs[agent_type](state)
    # print(tmp)
    return tmp


# def policy(s, a):
#     # s = int(s - 1)
#     if -k <= a < 0 or 0 < a <= k:
#         return 1 / 2 / k
#     return 0


def v_hat(state):
    # print(w.shape, x(state).shape)
    dt = np.dot(w.T, x(state))
    return dt


def choose_action(state):
    action = np.random.choice(possible_actions)
    print(action)
    return action


def save_state_and_action(state, action):
    global previous_action, previous_state
    previous_state = np.copy(state)
    previous_action = np.copy(action)


def agent_init():
    global iht, agent_type, w, possible_actions, fourier_order, seed

    np.random.seed(seed)
    """
    Hint: Initialize the variables that need to be reset before each run begins
    Returns: nothing
    """
    # initialize the policy array in a smart way
    possible_actions = np.concatenate((np.arange(-k, 0), np.arange(1, k + 1)))
    if agent_type == "tabular":
        w = np.zeros((size, 1))
    elif agent_type == "tile":
        w = np.zeros((IHT_SIZE, 1))  # (num_tilings[agent_type] / tile_widths[agent_type], 1))
        iht = IHT(IHT_SIZE)  # num_tilings[agent_type] / tile_widths[agent_type])  # 13  # size * k)
    elif agent_type == "aggregation":
        w = np.zeros((num_tilings[agent_type] / tile_widths[agent_type], 1))
    elif agent_type == "fourier":
        w = np.zeros((2 * fourier_order + 1, 1))
    else:
        print("Error")
        exit(213)


def agent_start(state):
    """
    Hint: Initialize the variables that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    # pick the first action, don't forget about exploring starts
    action = choose_action(state)
    save_state_and_action(state=state, action=action)
    return action


def agent_step(reward, state):  # returns NumPy array, reward: floating point, this_observation: NumPy array
    global w, agent_type
    """
    Arguments: reward: floating point, state: integer
    Returns: action: integer
    """
    # select an action, based on Q
    # print(previous_state)
    # print(np.nonzero(x(previous_state))[0])
    # print(v_hat(previous_state))
    # print(state)
    # print(np.nonzero(x(state))[0])
    # print(v_hat(state))
    err = alphas[agent_type] * (reward + gamma * v_hat(state) - v_hat(previous_state)) * x(previous_state)
    # print('w is', w)
    # print('err is', err)
    w += err

    action = choose_action(state)
    save_state_and_action(state=state, action=action)
    return action


def agent_end(reward):
    global w, agent_type
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    # do learning and update pi
    tmpv = v_hat(previous_state)
    tmpx = x(previous_state)
    # print('tmpv is', tmpv)
    # print 'ps is ', previous_state, 'reward', reward
    err = alphas[agent_type] * (reward - tmpv) * tmpx
    print err[np.nonzero(err)]
    w += err
    return


def agent_cleanup():
    """
    This function is not used
    """
    # clean up
    return


def agent_message(in_message):  # returns string, in_message: string
    global w, agent_type, iht, fourier_order, seed
    """
    Arguments: in_message: string
    returns: The value function as a string.
    This function is complete. You do not need to add code here.
    """
    # should not need to modify this function. Modify at your own risk
    if in_message.lower() == 'ValueFunction'.lower():
        return np.fromiter((v_hat(np.asarray([state])) for state in range(1, size + 1)), dtype=np.float64)

    elif in_message.lower() == "Tabular".lower():
        agent_type = "tabular"
    elif in_message.lower() == "Aggregation".lower():
        agent_type = "aggregation"
    elif in_message.lower() == 'Tile'.lower():
        agent_type = 'tile'
    elif "Fourier".lower() in in_message.lower():
        agent_type = 'fourier'
        fourier_order = int(in_message.split()[-1])
    elif "seed" in in_message:
        seed = int(in_message.split()[-1])
        np.random.seed()
    else:
        return "I don't know what to return!!"


if __name__ == "__main__":
    choose_action(0)
