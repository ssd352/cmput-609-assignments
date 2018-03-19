#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Andrew
  Jacobsen, Victor Silva, Sina Ghiassian
  Purpose: Implementation of the interaction between the Gambler's problem environment
  and the Monte Carlo agent using RL_glue.
  For use in the Reinforcement Learning course, Fall 2017, University of Alberta

"""

from rl_glue import *  # Required for RL-Glue
RLGlue("maze_env", "dyna_agent")

import numpy as np
# import pickle

if __name__ == "__main__":
    print("alpha,performance")
    for alpha in [0.03125, 0.0625, 0.1, 0.125, 0.25, 0.5, 1.0]:
        RL_agent_message("alpha " + str(alpha))
        # agent.alpha = alpha
        # globals()['_alpha'] = alpha
        num_episodes = 50
        max_steps = 0
        num_runs = 10
        episodes = np.empty((num_runs, num_episodes))

        for run in range(num_runs):
            RL_agent_message("set seed to " + str(run))
            counter = 0
            # print "run number: ", run
            RL_init()
            # print(RL_agent_message("steps"))
            # print "\n"
            for episode in range(num_episodes):
                # print "episode number: ", episode
                RL_episode(max_steps)
                episodes[run][episode] = RL_num_steps()  # RL_agent_message("steps")
                # print(RL_agent_message("steps"))

            RL_cleanup()
        # avg_episodes = np.mean(episodes[:, 1:])
        avg_episodes = np.mean(episodes)
        print(str(alpha) + "," + str(avg_episodes))
        # print (avg_episodes[1])
        # print (avg_episodes[2])

