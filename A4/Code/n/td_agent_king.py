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
NUMBER_OF_ACTIONS = 8
n = 4
Q = total_number_steps = number_steps = None
action_map = ((0, 1), (1, 0), (-1, 0), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
rewards = states = actions = None


def choose_action(state):
    global Q
    mx = np.amax(Q[state[0], state[1], :])
    arg = np.argwhere(Q[state[0], state[1], :] == mx)
    inds = np.reshape(arg, np.size(arg))
    greedy_action = np.random.choice(inds)

    toss = rand_un()
    if toss > epsilon:
        action = greedy_action
    else:
        action = rand_in_range(NUMBER_OF_ACTIONS)

    return action


def convert_action(action):
    return action_map[action]


def add_to_rewards(reward):
    global number_steps, rewards
    rewards[number_steps % n] = reward


def add_to_states(state):
    global number_steps, states
    states[number_steps % n, :] = np.copy(state)


def add_to_actions(action):
    global number_steps, actions
    actions[number_steps % n] = action


def gain():
    global rewards
    return np.sum(rewards)


def agent_init():
    global Q, total_number_steps, rewards, states, actions
    """
    Hint: Initialize the variables that need to be reset before each run begins
    Returns: nothing
    """
    Q = np.zeros((7, 10, NUMBER_OF_ACTIONS))  # number of rows, columns, actions
    rewards = np.zeros(n)
    states = np.zeros((n, 2))
    actions = np.zeros(n)

    total_number_steps = 0


def agent_start(state):
    """
    Hint: Initialize the variables that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    global number_steps, total_number_steps

    number_steps = 0
    action = choose_action(state)
    # first_state = np.copy(state)
    # total_number_steps += 1

    add_to_states(state)
    add_to_actions(action)
    return convert_action(action)


def agent_step(reward, state):  # returns NumPy array, reward: floating point, this_observation: NumPy array
    """
    Arguments: reward: floting point, state: integer
    Returns: action: floating point
    """
    global Q, total_number_steps, number_steps, n
    # select an action, based on Q
    # action = choose_action(state)

    # if tao >= 0:
    # else:
    #     pass

    number_steps += 1
    total_number_steps += 1
    add_to_rewards(reward)

    tao = number_steps - n
    action = choose_action(state)

    if tao >= 0:
        ind = tao % n
        # print("ind is ", ind, " states is ", states, " actions is ", actions, " action is ", action)
        Q[states[ind, 0], states[ind, 1], actions[ind]] += alpha * (gain() + Q[state[0], state[1], action]
                                                                    - Q[states[ind, 0], states[ind, 1], actions[ind]])
    add_to_states(state)
    add_to_actions(action)

    # update Q
    caction = convert_action(action)
    return caction


def agent_end(reward):
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    # print "Final reward is ", reward
    # do learning and update Q
    global total_number_steps, Q, number_steps, rewards, n
    number_steps += 1
    total_number_steps += 1
    add_to_rewards(reward)
    G = 0

    for cnt in range(n):
        G += rewards[(number_steps - cnt) % n]
        ind = (number_steps - cnt - 1) % n
        Q[states[ind][0], states[ind][1], actions[ind]] += alpha * (G
                                                                    - Q[states[ind][0], states[ind][1], actions[ind]])

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
