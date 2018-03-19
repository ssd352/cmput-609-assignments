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
# from queue import PriorityQueue
from heapq import *

epsilon = 0.1
alpha = 0.1
gamma = 0.95
n = 5
NUMBER_OF_ACTIONS = 4
theta = 0.1 * alpha * (1 + gamma)  # 0.06
Q = model = reverse_model = previous_state = previous_action = number_steps = p_queue = entry_finder = None
PlanningGenerator = ActionGenerator = None


def choose_action(state):
    global Q, ActionGenerator
    mx = np.amax(Q[state[0], state[1], :])
    arg = np.argwhere(Q[state[0], state[1], :] == mx)
    inds = np.reshape(arg, np.size(arg))
    greedy_action = ActionGenerator.choice(inds)
    # greedy_action = np.argmax(Q[state[0], state[1], :])

    # toss = rand_un()
    toss = ActionGenerator.uniform()
    if toss > epsilon:
        action = greedy_action
    else:
        action = ActionGenerator.randint(NUMBER_OF_ACTIONS)

    return action


def convert_action(action):
    actions = ((0, 1), (0, -1), (1, 0), (-1, 0))
    return actions[action]


def update_Q(s, a, r, sp=None):
    global Q
    tmp = Q[s[0], s[1], a]
    if sp is None:
        Q[s[0], s[1], a] += alpha * (r - Q[s[0], s[1], a])
    elif sp[0] < 0 or sp[1] < 0:
        Q[s[0], s[1], a] += alpha * (r - Q[s[0], s[1], a])
    else:
        Q[s[0], s[1], a] += alpha * (r + gamma * np.amax(
            Q[sp[0], sp[1], :]) - Q[s[0], s[1], a])
    return np.absolute(tmp - Q[s[0], s[1], a])


def add_to_model(s, a, r, sp):
    global model, reverse_model
    s = [int(s[0]), int(s[1])]
    sp = [int(sp[0]), int(sp[1])]
    model[s[0], s[1], a] = (int(sp[0]), int(sp[1]), r)
    if (sp[0], sp[1]) not in reverse_model:
        reverse_model[(sp[0], sp[1])] = []
    reverse_model[(sp[0], sp[1])].append((s[0], s[1], a))


# TODO: check again if error encountered
def enqueue(priority, element):
    global p_queue, entry_finder
    if priority < theta:
        return
    # if element in entry_finder:
    #     entry = entry_finder[element]
    #     entry[0] = p_queue[0][0]
    #     entry[1] = p_queue[0][1]
    #     heappop(p_queue)

    entry = [-priority, element]  # minus because it's a min-heap
    entry_finder[element] = entry
    heappush(p_queue, entry)
    # assert element in entry_finder
    # assert entry in p_queue
    # return entry


def dequeue():
    global p_queue
    # print(p_queue)
    p, element = heappop(p_queue)
    # print(p, element)
    # del entry_finder[element]
    return element


def pick_from_model():
    global model
    sample_state = np.empty(2, dtype=np.int64)
    sample_next_state = np.empty(2, dtype=np.int64)
    sample_state[0], sample_state[1], sample_action = dequeue()
    sample_next_state[0], sample_next_state[1], sample_reward = model[(sample_state[0], sample_state[1], sample_action)]
    return sample_state, sample_action, sample_reward, sample_next_state


def update_interesting_states(sample_state):
    global reverse_model
    s_bar = np.empty(2, dtype=np.int64)
    for s_bar[0], s_bar[1], a_bar in reverse_model[(sample_state[0], sample_state[1])]:
        dummy1, dummy2, r_bar = model[(s_bar[0], s_bar[1], a_bar)]
        priority = np.absolute(
            r_bar + gamma * np.amax(Q[sample_state[0], sample_state[1], :]) - Q[s_bar[0], s_bar[1], a_bar])
        assert priority is not None, "Interesting Problem"
        enqueue(priority, (int(s_bar[0]), int(s_bar[1]), a_bar))


def agent_init():
    global Q, model, p_queue, reverse_model, ActionGenerator, PlanningGenerator, entry_finder
    """
    Hint: Initialize the variables that need to be reset before each run begins
    Returns: nothing
    """
    Q = np.zeros((9, 6, NUMBER_OF_ACTIONS))  # number of rows, columns, actions
    model = {}
    reverse_model = {}
    entry_finder = {}
    p_queue = []
    ActionGenerator = RandomState(0)
    PlanningGenerator = RandomState(0)


def agent_start(state):
    """
    Hint: Initialize the variables that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    global previous_state, previous_action, number_steps

    state = np.asarray(state, dtype=np.int64)
    action = choose_action(state)
    previous_action = action
    previous_state = np.copy(state)
    number_steps = 0
    return convert_action(action)


def agent_step(reward, state):  # returns NumPy array, reward: floating point, this_observation: NumPy array
    """
    Arguments: reward: floating point, state: integer
    Returns: action: floating point
    """
    global previous_state, previous_action, Q, number_steps, reverse_model
    # select an action, based on Q
    state = np.asarray(state, dtype=np.int64)
    action = choose_action(state)
    add_to_model(previous_state, previous_action, reward, state)

    # update Q
    priority = update_Q(previous_state, previous_action, reward, state)
    # Q[previous_state[0], previous_state[1], previous_action] += alpha * (
    #     reward + gamma * np.amax(Q[state[0], state[1], :]) - Q[previous_state[0], previous_state[1], previous_action])

    assert priority is not None, "Normal Problem"
    enqueue(priority, (int(previous_state[0]), int(previous_state[1]), previous_action))
    # learn a model

    # model[previous_state[0], previous_state[1], previous_action] = (state[0], state[1], reward)
    # np.asarray((state[0], state[1], reward))

    # use model
    # print len(p_queue), n
    # if len(p_queue) != 0:
    #     print p_queue[0]
    for cnt in range(n):
        if len(p_queue) == 0:
            break
        # sample_state, sample_action, sample_reward, sample_next_state = pick_from_model()

        # sample_state[0], sample_state[1], sample_action = np.random.choice(model.keys())
        # sample_next_state[0], sample_next_state[1], sample_reward = model[
        #     (sample_state[0], sample_state[1], sample_action)]
        sample_state, sample_action, sample_reward, sample_next_state = pick_from_model()
        update_Q(sample_state, sample_action, sample_reward, sample_next_state)
        update_interesting_states(sample_state)

    # save state & action
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
    add_to_model(previous_state, previous_action, reward, np.array([-1, -1]))
    priority = update_Q(previous_state, previous_action, reward)
    enqueue(priority, (previous_state[0], previous_state[1], previous_action))

    # model[previous_state[0], previous_state[1], previous_action] = (-1, -1, reward)

    for cnt in range(n):
        if len(p_queue) == 0:
            break
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
        update_interesting_states(sample_state)

    number_steps += 1
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
    elif in_message.lower() == 'steps'.lower():
        return number_steps
    else:
        return "I don't know what to return!!"


if __name__ == "__main__":
    agent_init()
    agent_start((3, 0))
    agent_step(-1, (3, 1))
