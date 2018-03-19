#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian
  Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
 
  agent does *no* learning, selects actions randomly from the set of legal actions
 
"""

from __future__ import print_function, division
from utils import rand_in_range, rand_un
import numpy as np


last_action = None # last_action: NumPy array

num_actions = 10

class GenericAgent(object):

    def __init__(self, num_actions, q1, step_size):
        # super(object, self).__init__()
        self._Q = np.array(q1, dtype=np.float64)
        self._N = np.zeros(num_actions, dtype=np.float64)
        self._step_size = step_size
        self._num_actions = num_actions
        self._last_action = None

    def pick_first_action(self):
        action = rand_in_range(self._num_actions)
        self._last_action = action
        # print('action is {}'.format(self._last_action))
        return action

    def step(self, reward):
        # print('_Q is {}'.format(self._Q))
        # print('_N is {}'.format(self._N))
        # print(self._step_size)
        index = self._last_action
        # print('index is {}'.format(index))
        self._N[index] += 1
        # print('increment is {}'.format( self._step_size * (reward - self._Q[index]) ) )
        
        self._Q[index] += (reward - self._Q[index]) / self._N[index]
        # print('_Q[index] is {}'.format(self._Q[index]))

    def _find_max(self, arr):
        # print(arr)
        q_max = max(arr)
        first_occurence = np.where(arr == q_max)[0][0]
        action = first_occurence
        return action


##class EpsGreedyAgent(GenericAgent):
##
##    def __init__(self, num_actions, q1, step_size, epsilon):
##        super(EpsGreedyAgent, self).__init__(num_actions, q1, step_size)
##        self._epsilon = epsilon
##
##    def pick_action(self):
##        if self._last_action == None:
##            self.pick_first_action()
##        arg_max = self._find_max(self._Q)
##        if rand_un() < self._epsilon:
##            action = arg_max
##        else:
##            action = rand_in_range(self._num_actions)
##        self._last_action = action
##        # print('action is {}'.format(self._last_action))
##        return action
        
class UCBAgent(GenericAgent):

    def __init__(self, num_actions, q1, step_size, c):
        super(UCBAgent, self).__init__(num_actions, q1, step_size)
        self._t = 1
        self._c = c
        self._Uq = self._update()
               
    def _update(self):
        # print('inverse is {}'.format(1 / (self._N + 1)))
        return self._Q + self._c * np.sqrt(np.log(self._t) / (self._N + 1))
        # return self._Q + self._c * np.sqrt(1 / (self._N + 1))

    def step(self, reward):
        super(UCBAgent, self).step(reward)
        self._Uq = self._update()
        # print('_Uq is {}'.format(self._Uq))
        self._t += 1


    def pick_action(self):
        if self._last_action == None:
            self.pick_first_action()
        arg_max = self._find_max(self._Uq)
        self._last_action = arg_max
        return arg_max
        

def agent_init():
    global last_action, agent, ucb

    last_action = np.zeros(1) # generates a NumPy array with size 1 equal to zero
    # agent = EpsGreedyAgent(num_actions, [5] * num_actions, 0.1, 0)
    # agent = EpsGreedyAgent(num_actions, [0] * num_actions, 0.1, 0.1)
    agent = UCBAgent(num_actions, [0] * num_actions, 0.1, 1)
    
def agent_start(this_observation): # returns NumPy array, this_observation: NumPy array
    global last_action

    last_action[0] = rand_in_range(num_actions)

    local_action = np.zeros(1)
    local_action[0] = rand_in_range(num_actions)

    # return local_action[0]
    return agent.pick_action()


def agent_step(reward, this_observation): # returns NumPy array, reward: floating point, this_observation: NumPy array
    global last_action

    local_action = np.zeros(1)
    local_action[0] = rand_in_range(num_actions)

    # might do some learning here
    agent.step(reward)
    last_action = local_action
    
    return agent.pick_action()

def agent_end(reward): # reward: floating point
    # final learning update at end of episode
    agent.step(reward)
    return

def agent_cleanup():
    # clean up
    return

def agent_message(inMessage): # returns string, inMessage: string
    # might be useful to get information from the agent

    if inMessage == "what is your name?":
        return "my name is skeleton_agent!"
  
    # else
    return "I don't know how to respond to your message"

if __name__ == "__main__":
    agent_init()
    print(agent_start(0))
    for cnt in range(10):
        print(agent_step(8, 8))
    agent_end(8)
    agent_cleanup()
