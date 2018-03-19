#!/usr/bin/env python

"""
  Author: Adam White, Mohammad M. Ajallooeian
  Purpose: for use of Reinforcement learning course University of Alberta Fall 2017
 
  env *ignores* actions: rewards are all random
"""

from __future__ import print_function, division
from utils import rand_norm, rand_in_range, rand_un
import numpy as np


class Environment:

    def _arms(self, k):
        tmp = np.zeros(k)
        for cnt in range(k):
            tmp[cnt] = rand_norm(0, 1)
        # print('q* is {}'.format(tmp))
        return tmp

    def __init__(self, k):
        self._q_array = self._arms(k)

    def reward(self, action):
        # print('Action is ' + str(action))
        index = int(action)
        return rand_norm(self._q_array[index], 1)
    
    def optimal_action(self):
        return np.where(self._q_array == max(self._q_array))[0][0]


this_reward_observation = (None, None, None)  # this_reward_observation: (floating point, NumPy array, Boolean)


def env_init():
    global this_reward_observation, env
    local_observation = np.zeros(0)  # An empty NumPy array
    env = Environment(10)

    this_reward_observation = (0.0, local_observation, False)


def env_start():  # returns NumPy array
    return this_reward_observation[1]


def env_step(this_action): # returns (floating point, NumPy array, Boolean), this_action: NumPy array
    global this_reward_observation
    # the_reward = rand_norm(0.0, 1.0) # rewards drawn from (0, 1) Gaussian
    the_reward = env.reward(this_action)
    
    this_reward_observation = (the_reward, this_reward_observation[1], False)

    return this_reward_observation


def env_cleanup():
    #
    return


def env_message(inMessage): # returns string, inMessage: string
    if inMessage.lower() == "Optimal action".lower():
        return env.optimal_action()
    # elif inMessage == "what is your name?":
    #    return "my name is skeleton_environment!"
    # else:
    #    return "I don't know how to respond to your message"


if __name__ == "__main__":
    env_init()
    print(env_start())
    a,b,c = env_step(3)
    print(a)
    print(b)
    print('q is {}'.format(env._q_array))
    print('optimal action is {}'.format(env.optimal_action()))
    env_cleanup()
