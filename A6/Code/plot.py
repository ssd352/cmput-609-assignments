#!/usr/bin/env python

"""
 Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian, Zach Holland
 Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
"""

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    num_runs = 30
    num_episodes = 5000
    v_pi = np.loadtxt("True.Values.dat")
    run_tab = np.zeros((num_episodes, num_runs))
    run_tile = np.zeros((num_episodes, num_runs))
    run_agg = np.zeros((num_episodes, num_runs))
    run_four = np.zeros((num_episodes, num_runs))
    for i in range(num_runs):
        episode = 0
        filename = 'runs/14/run.{:02d}.dat'.format(i + 1)
        tmp = np.loadtxt(filename)
        run_tab[:, i] = tmp[0:num_episodes]
        run_tile[:, i] = tmp[num_episodes:2 * num_episodes]
        run_agg[:, i] = tmp[2 * num_episodes: 3 * num_episodes]
        run_four[:, i] = tmp[3 * num_episodes: 4 * num_episodes]
    rms1 = np.average(run_tab, 1)
    rms2 = np.average(run_tile, 1)
    rms3 = np.average(run_agg, 1)
    rms4 = np.average(run_four, 1)

    # plt.show()
    # print V.shape
    # for i, episode_num in enumerate([100, 1000, 8000]):
    plt.plot(rms1, 'k', label="Tabular")
    plt.plot(rms2, 'r', label="Tile Coding")
    plt.plot(rms3, 'g', label="State Aggregation")
    plt.plot(rms4, 'b', label="Fourier Basis")
    # plt.plot(rms4, label="Tile Coding")
    # plt.plot(rms5, label="State Aggregation")

    # plt.xlim([0,100])
    # plt.xticks([1,25,50,75,99])
    plt.xlabel('Episodes')
    plt.ylabel('RMSVE')
    plt.legend()
    plt.savefig("Comparison.all.pdf")
    plt.savefig("Comparison.all.png")
    plt.show()
