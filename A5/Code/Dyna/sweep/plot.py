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

    arr = np.genfromtxt("dyna.sweep.csv", delimiter=',')[1:]
    X = arr[:, 0]
    Y = arr[:, 1]

    plt.figure()
    plt.plot(X, Y)
    # plt.plot(X, Y2)
    # plt.plot(X, Y3)
    #
    # plt.legend(("$n=0$", "$n=5$", "$n=50$"))
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Average Time steps per episode')
    plt.savefig('sweep.pdf')

    plt.figure()
    plt.semilogx(X, Y)
    # plt.plot(X, Y2)
    # plt.plot(X, Y3)
    #
    # plt.legend(("$n=0$", "$n=5$", "$n=50$"))
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Average Time steps per episode')
    plt.savefig('sweep.log.pdf')
