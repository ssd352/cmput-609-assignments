#!/usr/bin/env python

"""
 Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian, Zach Holland
 Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
"""
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    num_episodes = 50

    # Y1 = np.loadtxt("../Dyna/recr/dyna.0.csv")[1:]
    Y2 = np.loadtxt("../Dyna/recr/dyna.5.csv")[1:]
    # Y3 = np.loadtxt("../Dyna/recr/dyna.50.csv")[1:]
    # Y = np.loadtxt("run.csv")[1:]
    # Y = np.loadtxt("run.2.csv")[1:]
    Y = np.loadtxt("run.4.csv")[1:]
    # X = np.average(run_matrix, axis=1)
    X = np.arange(2, num_episodes + 1)
    # print(len(Y1), len(X))

    # print(X, Y)
    plt.figure()
    # plt.plot(X, Y1)
    plt.plot(X, Y2)
    # plt.plot(X, Y3)
    plt.plot(X, Y)

    plt.legend(("Uniform", "Prioritized Sweeping"))
    plt.xlabel('Episodes')
    plt.ylabel('Time steps per episode')
    plt.savefig('8.3.pdf')
    # plt.show()
