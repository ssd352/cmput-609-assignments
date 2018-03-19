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

    # The better set of parameters

    # RL_agent_message("Set epsilon to 0.0")
    # RL_agent_message("Set lambda to 0.9")
    # RL_agent_message("Set numTilings to 9")
    # RL_agent_message("Set alpha to " + str(0.75 / 9))
    # RL_agent_message("Set position grid to 9")
    # RL_agent_message("Set velocity grid to 9")
    # RL_agent_message("Set memory size to 4096")

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

    # For 3D Q plot
    RL_agent_message("Set seed to 0")
    RL_init()
    for e in range(1000):
        RL_episode(0)
    Q = RL_agent_message("get action values")
    np.save('Q', Q)


