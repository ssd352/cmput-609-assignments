#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Andrew
  Jacobsen, Victor Silva, Sina Ghiassian
  Purpose: Implementation of the interaction between the Gambler's problem environment
  and the Monte Carlon agent using RL_glue. 
  For use in the Reinforcement Learning course, Fall 2017, University of Alberta

"""

from rl_glue import *  # Required for RL-Glue
import numpy as np
from time import time

RLGlue("rnd_walk_env", "agent")

if __name__ == "__main__":
    v_pi = np.loadtxt("True.Values.dat")

    num_episodes = 1
    max_steps = 0  # 10000
    common_seed = time()  # (time() * 1000) % 259200
    print(common_seed)

    num_runs = 1

    agents = ["Tabular", "Tile", "Aggregation", "Fourier 5"]

    for agent in agents:
        RL_agent_message(agent)
        v_over_runs = np.zeros((num_runs, num_episodes))
        for run in range(num_runs):

            # print "run number: ", run
            RL_agent_message("set seed to " + str(int(common_seed) + run))
            # RL_agent_message("Tabular")
            # RL_agent_message("Tile")
            # RL_agent_message("Aggregation")
            # RL_agent_message("Fourier 5")
            RL_init()
            # print "\n"
            for episode in range(num_episodes):
                # print "episode number: ", episode
                RL_episode(max_steps)
                # if (episode in key_episodes):
                V = RL_agent_message('ValueFunction')
                # for item in V:
                #     print item
                v_over_runs[run][episode] = np.sqrt(np.average((V - v_pi) ** 2))

                # import matplotlib.pyplot as plt
                # plt.plot(V)
                # plt.show()
        RL_cleanup()

        average_v_over_runs = np.average(v_over_runs, 0)
        for item in average_v_over_runs:
            print item
