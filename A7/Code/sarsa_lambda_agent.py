#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Skeleton code for Monte Carlo Exploring Starts Control Agent
           for use on A3 of Reinforcement learning course University of Alberta Fall 2017
 
"""

from __future__ import division
import numpy as np
from tiles3 import IHT, tiles

# size = -
# k = 100
NUM_ACTIONS = 3
gamma = 1.0
previous_state = None
previous_action = None
seed = None
z = None
w = None
iht = None
epsilon = 0.0
num_tilings = 8
tiling = np.array([8, 8])
alpha = 0.1 / num_tilings
IHT_SIZE = 4096
lamb = 0.9


def my_tiles(state, action):
    # return tiles(iht, num_tilings, [(state[0] + 1.2) / (1.2 + 0.5) * tiling[0], (state[1] + 0.07) / (0.07 + 0.07)
    #                                 * tiling[1]], [action])
    # A = [action]
    pos = state[0]
    vel = state[1]
    return tiles(iht, num_tilings, [tiling[0] * pos / (0.5 + 1.2), tiling[1] * vel / (0.07 + 0.07)], [action])


def x_tile(state, action):
    global iht
    indices = my_tiles(state, action)
    # (state[0] + 1.2) / (1.2 + 0.5)
    #                 + tiling[0] * (state[1] + 0.07) / (0.07 + 0.07)
    #                 + tiling[0] * tiling[1] * action)
    t = np.zeros((IHT_SIZE, 1))
    t[indices] = 1.0
    return t


def x(state, action):
    tmp = x_tile(state, action)
    # print(tmp)
    return tmp


# def policy(s, a):
#     # s = int(s - 1)
#     if -k <= a < 0 or 0 < a <= k:
#         return 1 / 2 / k
#     return 0


def q_hat(state, action):
    # print(w.shape, x(state).shape)
    dt = np.dot(w.T, x(state, action))
    return dt


def choose_action(state):
    toss = np.random.uniform()
    if toss < epsilon:
        action = np.random.randint(NUM_ACTIONS)
    else:
        action_values = np.fromiter((q_hat(state, action) for action in range(NUM_ACTIONS)), dtype=np.float64)

        indices = np.argwhere(action_values + np.finfo(np.float32).eps >= np.amax(action_values)).T[0]
        # print indices
        # indices = np.reshape(indices, (1, indices.size))[0]
        action = np.random.choice(indices)
    # print(action)
    return action


def save_state_and_action(state, action):
    global previous_action, previous_state
    previous_state = np.copy(state)
    previous_action = action


def update_W(delta):
    global alpha, z, w
    w += alpha * delta * z
    if not np.all(np.isfinite(w)):
        print "Invalid values detected for parameters {} {} {} {}".format(alpha, lamb, epsilon, num_tilings)


def agent_init():
    global iht, w, seed

    # np.random.seed(seed)
    """
    Hint: Initialize the variables that need to be reset before each run begins
    Returns: nothing
    """
    # initialize the policy array in a smart way
    # w = np.zeros((IHT_SIZE, 1))  # (num_tilings[agent_type] / tile_widths[agent_type], 1))
    w = -0.001 * np.random.uniform(size=(IHT_SIZE, 1))
    iht = IHT(IHT_SIZE)  # num_tilings[agent_type] / tile_widths[agent_type])  # 13  # size * k)


def agent_start(state):
    global z
    """
    Hint: Initialize the variables that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    # pick the first action, don't forget about exploring starts
    action = choose_action(state)
    save_state_and_action(state=state, action=action)
    z = np.zeros((IHT_SIZE, 1))
    return action


def agent_step(reward, state):  # returns NumPy array, reward: floating point, this_observation: NumPy array
    global w, z, previous_state, previous_action
    """
    Arguments: reward: floating point, state: integer
    Returns: action: integer
    """
    delta = reward
    # for i in my_tiles(previous_state, previous_action):
    #     # z[i] += 1
    #     delta -= w[i]
    #     z[i] = 1
    til = my_tiles(previous_state, previous_action)
    delta -= np.sum(w[til])
    # z[til] += 1
    z[til] = 1.0

    action = choose_action(state)
    # for i in my_tiles(state, action):
    #     delta += gamma * w[i]
    til = my_tiles(state, action)
    delta += gamma * np.sum(w[til])

    update_W(delta)
    z = gamma * lamb * z
    save_state_and_action(state=state, action=action)
    return action


def agent_end(reward):
    global w, z, previous_state, previous_action
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    # print "Episode done"
    delta = reward
    # for i in my_tiles(previous_state, previous_action):
    #     # z[i] += 1
    #     delta -= w[i]
    #     z[i] = 1
    til = my_tiles(previous_state, previous_action)
    delta -= np.sum(w[til])
    z[til] = 1.0
    # z[til] += 1
    update_W(delta)
    return


def agent_cleanup():
    """
    This function is not used
    """
    # clean up
    return


def agent_message(in_message):  # returns string, in_message: string
    global w, iht, seed, alpha, num_tilings, epsilon, tiling, lamb, IHT_SIZE
    """
    Arguments: in_message: string
    returns: The value function as a string.
    This function is complete. You do not need to add code here.
    """
    # should not need to modify this function. Modify at your own risk
    if 'action' in in_message.lower():
        steps = np.array([50, 50])
        q_function = np.zeros((steps[0], steps[1], NUM_ACTIONS))
        for s0 in np.arange(steps[0]):
            for s1 in np.arange(steps[1]):
                for act in np.arange(NUM_ACTIONS):
                    q_function[s0, s1, act] = q_hat(np.asarray([
                        -1.2 + (s0 * 1.7 / steps[0]), -0.07 + (s1 * 0.14 / steps[1])]), act)
        return -np.amax(q_function, axis=2)
        # return np.fromiter((q_hat(np.asarray([s0, s1]), act)
        #                     for s0 in np.linspace(-1.2, 0.5, 50)
        #                     for s1 in np.linspace(-0.07, 0.07, 50)
        #                     for act in np.arange(NUM_ACTIONS)
        #                     ), dtype=np.float64)
    elif "alpha" in in_message.lower():
        alpha = float(in_message.split()[-1])
    elif "position" in in_message.lower():
        tiling[0] = int(in_message.split()[-1])
    elif "velocity" in in_message.lower():
        tiling[1] = int(in_message.split()[-1])
    elif "memory" in in_message.lower():
        IHT_SIZE = int(in_message.split()[-1])
    elif "eps" in in_message.lower():
        epsilon = float(in_message.split()[-1])
    elif "lambda" in in_message.lower():
        lamb = float(in_message.split()[-1])
    elif "numTilings" in in_message.lower():
        num_tilings = int(in_message.split()[-1])
    elif "seed" in in_message:
        seed = int(in_message.split()[-1])
        np.random.seed(seed)
    else:
        return "I don't know what to return!!"


if __name__ == "__main__":
    choose_action(0)
