#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Andrew
  Jacobsen, Victor Silva, Sina Ghiassian
  Purpose: Implementation of the interaction between the Gambler's problem environment
  and the Monte Carlo agent using RL_glue.
  For use in the Reinforcement Learning course, Fall 2017, University of Alberta

"""

from rl_glue import *  # Required for RL-Glue
RLGlue("maze_env", "dyna_ps_agent")

import numpy as np
# import pickle

if __name__ == "__main__":
    num_episodes = 50
    max_steps = 0 
    num_runs = 10
    episodes = np.empty((num_runs, num_episodes))
    np.random.seed(2017)

    for run in range(num_runs):
        counter = 0
        # print "run number: ", run
        RL_init()
        # print(RL_agent_message("steps"))
        # print "\n"
        for episode in range(num_episodes):
            # print "episode number: ", episode
            RL_episode(max_steps)
            episodes[run][episode] = RL_agent_message("steps")
            # print(RL_agent_message("steps"))

        RL_cleanup()
    avg_episodes = np.mean(episodes, 0)
    # print (avg_episodes[1])
    # print (avg_episodes[2])
    for item in avg_episodes:
        print(item)

