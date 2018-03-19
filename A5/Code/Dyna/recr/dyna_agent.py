#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Skeleton code for Monte Carlo Exploring Starts Control Agent
           for use on A3 of Reinforcement learning course University of Alberta Fall 2017
 
"""

from __future__ import division
from utils import rand_in_range, rand_un
import numpy as np
from numpy.random import RandomState
import pickle

epsilon = 0.1
alpha = 0.1
gamma = 0.95
seed = 0
n = 50
NUMBER_OF_ACTIONS = 4
Q = model = previous_state = previous_action = number_steps = None
PlanningRandomNumberGenerator = EpsilonRandomNumberGenerator = None


def choose_action(state):
    global Q, EpsilonRandomNumberGenerator
    mx = np.amax(Q[state[0], state[1], :])
    arg = np.argwhere(Q[state[0], state[1], :] == mx)
    inds = np.reshape(arg, np.size(arg))
    greedy_action = EpsilonRandomNumberGenerator.choice(inds)  # np.random.choice(inds)
    # greedy_action = np.argmax(Q[state[0], state[1], :])

    # toss = rand_un()
    toss = EpsilonRandomNumberGenerator.uniform()
    if toss > epsilon:
        action = greedy_action
    else:
        action = EpsilonRandomNumberGenerator.randint(NUMBER_OF_ACTIONS)  # rand_in_range(NUMBER_OF_ACTIONS)

    return action


def convert_action(action):
    actions = ((0, 1), (0, -1), (1, 0), (-1, 0))
    return actions[action]


def update_Q(s, a, r, sp=None):
    global Q
    s = (int(s[0]), int(s[1]))
    if sp is not None:
        sp = (int(sp[0]), int(sp[1]))
    if sp is None:
        Q[s[0], s[1], a] += alpha * (r - Q[s[0], s[1], a])
        return
    elif sp[0] < 0 or sp[1] < 0:
        Q[s[0], s[1], a] += alpha * (r - Q[s[0], s[1], a])
    else:
        Q[s[0], s[1], a] += alpha * (r + gamma * np.amax(
            Q[sp[0], sp[1], :]) - Q[s[0], s[1], a])


def add_to_model(s, a, r, sp):
    global model
    model[s[0], s[1], a] = (sp[0], sp[1], r)


def pick_from_model():
    global model, PlanningRandomNumberGenerator
    sample_state = np.empty(2)
    sample_next_state = np.empty(2)

    list_keys = list(model.keys())
    ind = PlanningRandomNumberGenerator.randint(len(list_keys)) #np.random.choice(len(list_keys))
    sample_state[0], sample_state[1], sample_action = list_keys[ind]
    # print(sample_state, sample_action)
    sample_next_state[0], sample_next_state[1], sample_reward = model[
        (sample_state[0], sample_state[1], sample_action)]
    return sample_state, sample_action, sample_reward, sample_next_state


def agent_init():
    global Q, model, PlanningRandomNumberGenerator, EpsilonRandomNumberGenerator
    """
    Hint: Initialize the variables that need to be reset before each run begins
    Returns: nothing
    """
    PlanningRandomNumberGenerator = RandomState(seed)
    EpsilonRandomNumberGenerator = RandomState(seed)
    Q = np.zeros((9, 6, NUMBER_OF_ACTIONS))  # number of rows, columns, actions
    model = {}


def agent_start(state):
    """
    Hint: Initialize the variables that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    global previous_state, previous_action, number_steps, EpsilonRandomNumberGenerator, PlanningRandomNumberGenerator


    action = choose_action(state)
    previous_action = action
    previous_state = np.copy(state)
    number_steps = 0
    return convert_action(action)


def agent_step(reward, state):  # returns NumPy array, reward: floating point, this_observation: NumPy array
    """
    Arguments: reward: floting point, state: integer
    Returns: action: floating point
    """
    global previous_state, previous_action, Q, number_steps

    # select an action, based on Q
    action = choose_action(state)

    # update Q
    update_Q(previous_state, previous_action, reward, state)
    # Q[previous_state[0], previous_state[1], previous_action] += alpha * (
    #     reward + gamma * np.amax(Q[state[0], state[1], :]) - Q[previous_state[0], previous_state[1], previous_action])

    # learn a model
    add_to_model(previous_state, previous_action, reward, state)
    # model[previous_state[0], previous_state[1], previous_action] = (state[0], state[1], reward)
    # np.asarray((state[0], state[1], reward))

    # use model
    for cnt in range(n):
        sample_state, sample_action, sample_reward, sample_next_state = pick_from_model()
        # sample_state[0], sample_state[1], sample_action = np.random.choice(model.keys())
        # sample_next_state[0], sample_next_state[1], sample_reward = model[
        #     (sample_state[0], sample_state[1], sample_action)]
        update_Q(sample_state, sample_action, sample_reward, sample_next_state)


    previous_action = action
    previous_state = np.copy(state)
    number_steps += 1
    return convert_action(action)


def agent_end(reward):
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    # print "Final reward is ", reward
    # do learning and update Q
    global number_steps, Q, model
    # Q[previous_state[0], previous_state[1], previous_action] += alpha * (
    #      reward - Q[previous_state[0], previous_state[1], previous_action])
    update_Q(previous_state, previous_action, reward)

    # model[previous_state[0], previous_state[1], previous_action] = (-1, -1, reward)
    add_to_model(previous_state, previous_action, reward, np.array([-1, -1]))

    for cnt in range(n):
        # sample_state = np.empty((1, 2))
        # sample_next_state = np.empty((1, 2))
        #
        # sample_state[0], sample_state[1], sample_action = np.random.choice(model.keys())
        # sample_next_state[0], sample_next_state[1], sample_reward = model[
        #     (sample_state[0], sample_state[1], sample_action)]
        # Q[sample_state[0], sample_state[1], sample_action] += alpha * (sample_reward + gamma * np.amax(
        #     Q[sample_next_state[0], sample_next_state[1], :]) - Q[sample_state[0], sample_state[1], sample_action])
        sample_state, sample_action, sample_reward, sample_next_state = pick_from_model()
        update_Q(sample_state, sample_action, sample_reward, sample_next_state)

    number_steps += 1
    return


def agent_cleanup():
    """
    This function is not used
    """
    # clean up
    return


def agent_message(in_message):  # returns string, in_message: string
    global Q, seed
    """
    Arguments: in_message: string
    returns: The value function as a string.
    This function is complete. You do not need to add code here.
    """
    # should not need to modify this function. Modify at your own risk
    if in_message == 'ValueFunction':
        return pickle.dumps(np.max(Q, axis=1), protocol=0)
    elif 'seed' in in_message.lower():
        seed = int(in_message.split()[-1])
    elif in_message.lower() == 'steps'.lower():
        return number_steps
    else:
        return "I don't know what to return!!"


if __name__ == "__main__":
    agent_init()
    agent_start((3, 0))
    agent_step(-1, (3, 1))
