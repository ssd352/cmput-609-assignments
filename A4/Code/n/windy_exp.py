#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Andrew
  Jacobsen, Victor Silva, Sina Ghiassian
  Purpose: Implementation of the interaction between the Gambler's problem environment
  and the Monte Carlon agent using RL_glue. 
  For use in the Reinforcement Learning course, Fall 2017, University of Alberta

"""

from rl_glue import *  # Required for RL-Glue
RLGlue("windy_env", "td_agent_king")

import numpy as np
import pickle

if __name__ == "__main__":
    num_episodes = 200
    max_steps = 0

    num_runs = 1

    for run in range(num_runs):
        counter = 0
        # print "run number: ", run
        RL_init()
        print(RL_agent_message("total_steps"))
        # print "\n"
        for episode in range(num_episodes):
            # print "episode number: ", episode
            RL_episode(max_steps)
            print(RL_agent_message("total_steps"))

        RL_cleanup()
      

