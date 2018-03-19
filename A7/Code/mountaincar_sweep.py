#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Andrew
  Jacobsen, Victor Silva, Sina Ghiassian
  Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
  Last Modified by: Mohammad M. Ajallooeian, Sina Ghiassian
  Last Modified on: 21/11/2017

"""

from __future__ import division
from rl_glue import *  # Required for RL-Glue

RLGlue("mountaincar", "sarsa_lambda_agent")
import numpy as np
from sys import argv

if __name__ == "__main__":
    num_episodes = 200
    num_runs = 50

    steps = np.zeros([num_runs, num_episodes])
    avg_rewards = np.zeros((num_runs,))

    # This part of the code is for sweeping parameters
    alpha = float(raw_input())

    epsilon = float(raw_input())
    RL_agent_message("Set epsilon to " + str(epsilon))

    lamb = float(raw_input())
    RL_agent_message("Set lambda to " + str(lamb))

    num_tilings = int(raw_input())
    RL_agent_message("Set numTilings to " + str(num_tilings))
    RL_agent_message("Set alpha to " + str(alpha / num_tilings))

    position_grid = int(raw_input())
    RL_agent_message("Set position grid to " + str(position_grid))

    velocity_grid = int(raw_input())
    RL_agent_message("Set velocity grid to " + str(velocity_grid))

    memory_size = int(raw_input())
    RL_agent_message("Set memory size to " + str(memory_size))

    for r in range(num_runs):
        print "run number : ", r
        RL_agent_message("Set seed to " + str(r))
        RL_init()
        cum_reward = 0

        for e in range(num_episodes):
            # print '\tepisode {}'.format(e+1)
            RL_episode(0)
            cum_reward += RL_return()
            steps[r, e] = RL_num_steps()

        avg_rewards[r] = cum_reward
    np.save('steps', steps)
    avg_cum_reward = np.average(avg_rewards)
    std_cum_reward = np.std(avg_rewards) / np.sqrt(num_runs)

    with open("runs/sweep.csv", mode='a') as f:
        f.write(str(alpha) + ',' + str(num_tilings) + ',' + str(epsilon) + ',' + str(position_grid) + ','
                + str(velocity_grid) + ',' + str(lamb) + ',' + str(memory_size) + ',' + str(avg_cum_reward) + ','
                + str(std_cum_reward) + '\n')
    print str(alpha) + ',' + str(num_tilings) + ',' + str(epsilon) + ',' + str(position_grid) + ',' \
          + str(velocity_grid) + ',' + str(lamb) + ',' + str(memory_size) + ',' + str(avg_cum_reward) + ',' \
          + str(std_cum_reward)
